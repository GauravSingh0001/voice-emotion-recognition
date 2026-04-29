"""
services/orchestrator/gemini_client.py
──────────────────────────────────────
Module to handle transcription and emotion judging in a single Gemini 1.5 call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
import time
from typing import List, Optional, Tuple

import pydantic
from google import genai
from google.genai import types

from shared.config import get_settings
from shared.schemas import (
    Emotion,
    FastPathTrigger,
    JudgeVerdict,
    TranscriptionResult,
    UtteranceComplete,
)

logger = logging.getLogger("orchestrator.gemini_client")

# ── PCM → WAV Encoding ────────────────────────────────────────────────────────

def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 16_000, channels: int = 1) -> bytes:
    """Wrap raw 16-bit PCM in a valid WAV container."""
    bits_per_sample = 16
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_len = len(pcm_bytes)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_len, b"WAVE",
        b"fmt ", 16,             # Subchunk1Size
        1,                       # AudioFormat PCM
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data", data_len,
    )
    return header + pcm_bytes

# ── Pydantic Schema for Gemini Structured Output ──────────────────────────────

class GeminiAnalysisResponse(pydantic.BaseModel):
    transcript: str
    final_emotion: str
    confidence: float
    reasoning: str
    fast_path_summary: str

# ── Gemini Integration ────────────────────────────────────────────────────────

def _build_gemini_prompt(
    fast_path_history: List[FastPathTrigger],
    rms_energy: Optional[float] = None,
    energy_level: Optional[str] = None,
    speaking_rate: Optional[float] = None,
) -> str:
    emotion_timeline = []
    for i, fp in enumerate(fast_path_history[-10:]):  # Last 10 windows max
        emotion_timeline.append(
            f"  Window {fp.window_index} (+{i * 500}ms): {fp.top_emotion} "
            f"(confidence={fp.top_confidence:.2f})"
        )
    timeline_str = "\n".join(emotion_timeline) if emotion_timeline else "  (no fast-path data)"

    prompt = (
        "You are an expert emotion analyst and transcription engine. "
        "Your task is to transcribe the provided audio accurately and determine the speaker's final emotional state.\n\n"
        "You have access to the raw audio. You must ALSO consider the acoustic fast-path history provided below.\n\n"
        f"ACOUSTIC EMOTION TIMELINE (fast-path, 500ms windows):\n{timeline_str}\n"
    )

    if rms_energy is not None or energy_level or speaking_rate is not None:
        prompt += "\n\nACOUSTIC METADATA:"
        if energy_level:
            prompt += f"\n  Energy level: {energy_level} (RMS={rms_energy:.4f})"
        if speaking_rate is not None:
            prompt += f"\n  Speaking rate: {speaking_rate} words/sec"

    prompt += (
        "\n\nBased on your direct analysis of the audio tone, inflection, and content, "
        "along with the provided metadata, provide a structured JSON response containing:\n"
        "1. transcript: The transcribed text of the audio.\n"
        "2. final_emotion: One of [Neutral, Happy, Sad, Angry, Surprised, Sarcastic, Frustrated]\n"
        "3. confidence: Float between 0.0 and 1.0\n"
        "4. reasoning: A brief explanation referencing tone, text, and acoustic signals.\n"
        "5. fast_path_summary: A short summary of the provided acoustic timeline.\n"
    )

    return prompt

async def gemini_analyze_audio(
    utterance: UtteranceComplete,
    rms_energy: Optional[float] = None,
    energy_level: Optional[str] = None,
    speaking_rate: Optional[float] = None,
) -> Tuple[TranscriptionResult, JudgeVerdict]:
    """
    Analyzes audio multimodally using Gemini to get both transcription and judge verdict.
    """
    cfg = get_settings()
    
    if cfg.MOCK_APIS:
        await asyncio.sleep(0.1)
        t_res = TranscriptionResult(session_id=utterance.session_id, text="Mock transcript from Gemini", latency_ms=100)
        v_res = JudgeVerdict(final_emotion=Emotion.NEUTRAL, confidence=0.9, reasoning="Mock mode", fast_path_summary="Mocked")
        return t_res, v_res

    client = genai.Client(api_key=cfg.GEMINI_API_KEY)

    wav_bytes = _pcm_to_wav(utterance.pcm_bytes, sample_rate=utterance.sample_rate)
    prompt = _build_gemini_prompt(
        fast_path_history=utterance.window_emotions,
        rms_energy=rms_energy,
        energy_level=energy_level,
        speaking_rate=speaking_rate,
    )

    t0 = time.perf_counter()
    try:
        # Run synchronous generate_content in a thread pool to not block asyncio
        # google-genai supports asyncio with client.aio.models.generate_content
        response = await client.aio.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                types.Part.from_bytes(data=wav_bytes, mime_type='audio/wav'),
                prompt
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=GeminiAnalysisResponse,
            )
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        
        if response.text:
            data = json.loads(response.text)
            transcript_text = data.get("transcript", "")
            final_emotion_str = data.get("final_emotion", "Neutral")
            
            # Map string to Emotion enum safely
            try:
                final_emotion = Emotion(final_emotion_str.capitalize())
            except ValueError:
                final_emotion = Emotion.NEUTRAL
                
            t_res = TranscriptionResult(
                session_id=utterance.session_id,
                text=transcript_text,
                latency_ms=latency_ms / 2.0,  # Approximate split
            )
            
            v_res = JudgeVerdict(
                final_emotion=final_emotion,
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", ""),
                fast_path_summary=data.get("fast_path_summary", ""),
            )
            logger.info("Gemini Analysis: %.0f ms | '%s' -> %s", latency_ms, transcript_text[:40], final_emotion.value)
            return t_res, v_res
            
    except Exception as exc:
        logger.error("Gemini API error: %s", exc, exc_info=True)
    
    latency_ms = (time.perf_counter() - t0) * 1000
    t_fallback = TranscriptionResult(session_id=utterance.session_id, text="(unintelligible)", latency_ms=latency_ms)
    v_fallback = JudgeVerdict(final_emotion=Emotion.NEUTRAL, confidence=0.1, reasoning="Gemini API failed", fast_path_summary="N/A")
    return t_fallback, v_fallback
