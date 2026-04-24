"""
services/orchestrator/groq_client.py
──────────────────────────────────────
Module 5 — Groq Cloud Integration (Free Tier AI)

Wraps:
  5A: whisper-large-v3-turbo  → audio transcription
  5B: llama-3.3-70b-versatile → emotion judge

Features:
  • Token bucket rate counter (14,400 req/day free limit)
  • Exponential backoff on HTTP 429
  • Rule-based judge fallback when Groq is unavailable
  • Mock mode returning canned responses
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import struct
import time
from typing import List, Optional

from shared.config import get_settings
from shared.schemas import (
    Emotion,
    FastPathTrigger,
    JudgeVerdict,
    TranscriptionResult,
    UtteranceComplete,
)

logger = logging.getLogger("orchestrator.groq_client")

# ── Token Bucket Rate Counter ─────────────────────────────────────────────────

class _RateLimiter:
    """Simple in-memory daily request counter for the Groq free tier."""

    def __init__(self, daily_limit: int = 14_400):
        self.daily_limit = daily_limit
        self._count = 0
        self._day = time.strftime("%Y-%m-%d")

    def _reset_if_new_day(self) -> None:
        today = time.strftime("%Y-%m-%d")
        if today != self._day:
            self._count = 0
            self._day = today
            logger.info("Groq daily counter reset for %s", today)

    def consume(self, n: int = 1) -> bool:
        """Deduct n requests. Returns False if limit would be exceeded."""
        self._reset_if_new_day()
        if self._count + n > self.daily_limit:
            logger.error("Groq daily limit reached (%d / %d).", self._count, self.daily_limit)
            return False
        self._count += n
        remaining = self.daily_limit - self._count
        threshold = int(self.daily_limit * get_settings().GROQ_WARN_THRESHOLD)
        if self._count >= threshold:
            logger.warning(
                "Groq usage high: %d / %d requests used today (%d remaining).",
                self._count, self.daily_limit, remaining,
            )
        return True

    @property
    def requests_today(self) -> int:
        self._reset_if_new_day()
        return self._count

    @property
    def requests_remaining(self) -> int:
        self._reset_if_new_day()
        return max(0, self.daily_limit - self._count)


_rate_limiter = _RateLimiter()


def get_rate_limiter() -> _RateLimiter:
    return _rate_limiter


# ── Mock Responses ────────────────────────────────────────────────────────────

_MOCK_TRANSCRIPTION = "This is a mock transcription for offline testing."

_MOCK_VERDICT = JudgeVerdict(
    final_emotion=Emotion.NEUTRAL,
    confidence=0.85,
    reasoning=(
        "Mock mode: acoustic features suggest neutral emotional state. "
        "No real Groq API call was made."
    ),
    fast_path_summary="All mock windows returned Neutral.",
)


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


# ── Transcription (Groq Whisper) ──────────────────────────────────────────────

async def transcribe_utterance(utterance: UtteranceComplete) -> TranscriptionResult:
    """
    Send utterance audio to Groq Whisper and return a TranscriptionResult.
    Falls back to empty text on repeated failures.
    """
    cfg = get_settings()

    if cfg.MOCK_APIS:
        await asyncio.sleep(0.05)  # Simulate network latency
        return TranscriptionResult(
            session_id=utterance.session_id,
            text=_MOCK_TRANSCRIPTION,
            latency_ms=50.0,
        )

    if not _rate_limiter.consume(1):
        return TranscriptionResult(
            session_id=utterance.session_id,
            text="",
            latency_ms=0.0,
        )

    wav_bytes = _pcm_to_wav(utterance.pcm_bytes, sample_rate=utterance.sample_rate)

    max_retries = 2
    for attempt in range(max_retries):
        try:
            import aiohttp  # type: ignore

            t0 = time.perf_counter()
            form = aiohttp.FormData()
            form.add_field("file", wav_bytes, filename="utterance.wav", content_type="audio/wav")
            form.add_field("model", "whisper-large-v3-turbo")
            form.add_field("response_format", "json")

            headers = {"Authorization": f"Bearer {cfg.GROQ_API_KEY}"}

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    data=form,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=5.0),
                ) as resp:
                    latency_ms = (time.perf_counter() - t0) * 1000

                    if resp.status == 200:
                        data = await resp.json()
                        text = data.get("text", "").strip()
                        logger.info("Groq Whisper: %.0f ms | '%s'", latency_ms, text[:80])
                        return TranscriptionResult(
                            session_id=utterance.session_id,
                            text=text,
                            latency_ms=latency_ms,
                        )

                    elif resp.status == 429:
                        wait = 5 * (2 ** attempt)
                        logger.warning(
                            "Groq rate limit (429) on attempt %d. Waiting %ds.",
                            attempt + 1, wait,
                        )
                        await asyncio.sleep(wait)

                    else:
                        body = await resp.text()
                        logger.error("Groq Whisper HTTP %d: %s", resp.status, body[:200])
                        break

        except asyncio.TimeoutError:
            logger.warning("Groq Whisper timeout on attempt %d.", attempt + 1)
        except Exception as exc:
            logger.error("Groq Whisper error: %s", exc, exc_info=True)
            break

    return TranscriptionResult(session_id=utterance.session_id, text="", latency_ms=0.0)


# ── Judge (Groq Llama) ────────────────────────────────────────────────────────

_JUDGE_SYSTEM_PROMPT = """You are an expert emotion analyst. Your task is to determine
the final emotional state from speech, resolving any mismatch between acoustic signals
and spoken content.

You will receive:
1. A transcription of spoken words
2. A history of acoustic emotion detections (fast-path, every 500ms)

Respond ONLY with a valid JSON object with exactly these keys:
{
  "final_emotion": "<one of: Neutral, Happy, Sad, Angry, Surprised>",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<brief explanation of your decision>",
  "fast_path_summary": "<summary of acoustic signal pattern>"
}

Do not include any text outside the JSON object."""


def _build_judge_prompt(
    transcript: str,
    fast_path_history: List[FastPathTrigger],
) -> str:
    emotion_timeline = []
    for i, fp in enumerate(fast_path_history[-10:]):  # Last 10 windows max
        emotion_timeline.append(
            f"  Window {fp.window_index} (+{i * 500}ms): {fp.top_emotion} "
            f"(confidence={fp.top_confidence:.2f})"
        )
    timeline_str = "\n".join(emotion_timeline) if emotion_timeline else "  (no fast-path data)"

    return (
        f"TRANSCRIPT:\n\"{transcript}\"\n\n"
        f"ACOUSTIC EMOTION TIMELINE (fast-path, 500ms windows):\n{timeline_str}"
    )


def _rule_based_fallback(
    session_id: str,
    fast_path_history: List[FastPathTrigger],
) -> JudgeVerdict:
    """Simple majority-vote fallback when Groq is unavailable."""
    if not fast_path_history:
        return JudgeVerdict(
            final_emotion=Emotion.UNKNOWN,
            confidence=0.0,
            reasoning="No fast-path data available and Groq API unavailable.",
        )

    from collections import Counter
    counts = Counter(fp.top_emotion for fp in fast_path_history)
    top_emotion, count = counts.most_common(1)[0]
    confidence = count / len(fast_path_history)

    return JudgeVerdict(
        final_emotion=top_emotion,
        confidence=round(confidence, 2),
        reasoning=(
            f"Rule-based fallback (Groq unavailable): majority vote from "
            f"{len(fast_path_history)} fast-path windows."
        ),
        fast_path_summary=f"Dominant: {top_emotion} ({count}/{len(fast_path_history)} windows)",
    )


async def judge_emotion(
    session_id: str,
    transcript: str,
    fast_path_history: List[FastPathTrigger],
) -> JudgeVerdict:
    """
    Call Groq Llama 3.3 70B to produce a final emotion verdict.
    Falls back to rule-based verdict on API failure.
    """
    cfg = get_settings()

    if cfg.MOCK_APIS:
        await asyncio.sleep(0.08)
        return _MOCK_VERDICT

    if not transcript.strip():
        logger.info("Empty transcript, using rule-based fallback.")
        return _rule_based_fallback(session_id, fast_path_history)

    if not _rate_limiter.consume(1):
        return _rule_based_fallback(session_id, fast_path_history)

    user_prompt = _build_judge_prompt(transcript, fast_path_history)

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 256,
        "response_format": {"type": "json_object"},
    }

    max_retries = 2
    for attempt in range(max_retries):
        try:
            import aiohttp  # type: ignore

            t0 = time.perf_counter()
            headers = {
                "Authorization": f"Bearer {cfg.GROQ_API_KEY}",
                "Content-Type": "application/json",
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10.0),
                ) as resp:
                    latency_ms = (time.perf_counter() - t0) * 1000

                    if resp.status == 200:
                        data = await resp.json()
                        content = data["choices"][0]["message"]["content"]
                        verdict_data = json.loads(content)

                        verdict = JudgeVerdict(
                            final_emotion=Emotion(verdict_data.get("final_emotion", "Unknown")),
                            confidence=float(verdict_data.get("confidence", 0.0)),
                            reasoning=verdict_data.get("reasoning", ""),
                            fast_path_summary=verdict_data.get("fast_path_summary"),
                        )
                        logger.info(
                            "Groq Judge: %.0f ms | %s (%.2f) | %s",
                            latency_ms,
                            verdict.final_emotion,
                            verdict.confidence,
                            verdict.reasoning[:60],
                        )
                        return verdict

                    elif resp.status == 429:
                        wait = 5 * (2 ** attempt)
                        logger.warning(
                            "Groq Judge rate limit (429) on attempt %d. Waiting %ds.",
                            attempt + 1, wait,
                        )
                        await asyncio.sleep(wait)

                    else:
                        body = await resp.text()
                        logger.error("Groq Judge HTTP %d: %s", resp.status, body[:200])
                        break

        except (asyncio.TimeoutError, json.JSONDecodeError, KeyError) as exc:
            logger.warning("Groq Judge error on attempt %d: %s", attempt + 1, exc)
        except Exception as exc:
            logger.error("Groq Judge unexpected error: %s", exc, exc_info=True)
            break

    logger.info("Falling back to rule-based judge for session %s.", session_id)
    return _rule_based_fallback(session_id, fast_path_history)
