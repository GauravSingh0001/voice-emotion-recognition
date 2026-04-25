"""
services/orchestrator/main.py
──────────────────────────────
Module 3 — Orchestrator (Local FastAPI + asyncio Coordination Hub)

Responsibilities:
  1. Exposes REST endpoints (/health, /session/start, /session/end, /session/{id}/report)
  2. Spawns the LiveKit bot listener and signal processor as background tasks
  3. Consumes events from the signal processor via a multiprocessing Queue
  4. Routes FastPathTrigger events → fast-path result → Supabase Realtime publish
  5. Routes UtteranceComplete events → Groq Whisper + Llama → Supabase INSERT
  6. Manages backpressure: queue up to 5 utterances, exponential backoff on 429
  7. Optionally saves utterance WAV files to ./recordings/

Run with:
  uvicorn services.orchestrator.main:app --host 0.0.0.0 --port 8000 --reload
Or via Makefile:
  make run
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
import struct
import time
import uuid
from collections import Counter, deque
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Project path bootstrap ──────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared.config import get_settings
from shared.schemas import (
    FastPathTrigger,
    ServiceHealth,
    SessionContext,
    TranscriptionResult,
    UtteranceComplete,
    VADState,
    VADStateChange,
)
from services.orchestrator.groq_client import (
    judge_emotion,
    transcribe_utterance,
    get_rate_limiter,
)
from services.orchestrator.supabase_client import (
    get_session_history,
    store_emotion_result,
    verify_user_token,
)
from services.orchestrator.livekit_bot import start_livekit_bot
from services.livekit_gateway.main import run_gateway
from services.signal_processor.vad_engine import run_signal_processor

logger = logging.getLogger("orchestrator")
cfg = get_settings()

# ── FastAPI Application ────────────────────────────────────────────────────────
app = FastAPI(
    title="Real-Time Emotion Intelligence API",
    description="$0 budget emotion detection: LiveKit + SenseVoice + Groq + Supabase",
    version="1.0.0",
)

# ── CORS Middleware (Task 1.4) ─────────────────────────────────────────────────
# allow_origins=["*"] is intentional for the demo.
# In production, restrict to your frontend URL, e.g. ["https://your-app.vercel.app"].
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Session Registry ──────────────────────────────────────────────────────────
_sessions: Dict[str, SessionContext] = {}

# ── Utterance Processing Queue (max 5 for backpressure) ──────────────────────
_utterance_queue: asyncio.Queue[UtteranceComplete] = asyncio.Queue(maxsize=5)

# ── Child Process Tracking ────────────────────────────────────────────────────
_gateway_processes: Dict[str, multiprocessing.Process] = {}
_signal_processes: Dict[str, multiprocessing.Process] = {}
_bot_tasks: Dict[str, asyncio.Task] = {}
_event_queues: Dict[str, multiprocessing.Queue] = {}

# ── Recordings ────────────────────────────────────────────────────────────────
_RECORDINGS_DIR = Path(__file__).resolve().parent.parent.parent / "recordings"


def _save_utterance_wav(session_id: str, utterance: UtteranceComplete) -> None:
    """Write utterance PCM to a WAV file in ./recordings/."""
    try:
        _RECORDINGS_DIR.mkdir(exist_ok=True)
        ts = int(utterance.timestamp_ms)
        filename = _RECORDINGS_DIR / f"{session_id}_{ts}.wav"
        bits = 16
        channels = 1
        sr = utterance.sample_rate
        data = utterance.pcm_bytes
        byte_rate = sr * channels * bits // 8
        block_align = channels * bits // 8
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", 36 + len(data), b"WAVE",
            b"fmt ", 16, 1, channels, sr,
            byte_rate, block_align, bits,
            b"data", len(data),
        )
        filename.write_bytes(header + data)
        logger.debug("Saved utterance to %s", filename)
    except Exception as exc:
        logger.warning("Could not save WAV: %s", exc)


# ── Event Consumer Task ───────────────────────────────────────────────────────

async def _poll_event_queues():
    """
    Continuously poll all active session event queues and dispatch events.
    Runs as a background asyncio task.
    """
    while True:
        for session_id, eq in list(_event_queues.items()):
            while not eq.empty():
                try:
                    event = eq.get_nowait()
                    await _dispatch_event(session_id, event)
                except Exception as exc:
                    logger.debug("Event dispatch error: %s", exc)
        await asyncio.sleep(0.02)  # 20ms polling interval


async def _dispatch_event(session_id: str, event) -> None:
    """Route an event to the appropriate handler."""
    ctx = _sessions.get(session_id)
    if ctx is None:
        return

    if isinstance(event, VADStateChange):
        ctx.vad_state = event.new_state
        logger.debug(
            "Session %s VAD: %s → %s", session_id, event.previous_state, event.new_state
        )

    elif isinstance(event, FastPathTrigger):
        ctx.fast_path_history.append(event)
        if len(ctx.fast_path_history) > 30:
            ctx.fast_path_history = ctx.fast_path_history[-30:]
        logger.info(
            "Fast-path [%s] window %d: %s (%.2f)",
            session_id[:8], event.window_index, event.top_emotion, event.top_confidence,
        )

    elif isinstance(event, UtteranceComplete):
        if cfg.RECORD_UTTERANCES:
            _save_utterance_wav(session_id, event)
        try:
            _utterance_queue.put_nowait(event)
        except asyncio.QueueFull:
            try:
                dropped = _utterance_queue.get_nowait()
                logger.error(
                    "Utterance queue full (5 items): dropped utterance %s.",
                    dropped.session_id[:8],
                )
            except asyncio.QueueEmpty:
                pass
            _utterance_queue.put_nowait(event)


async def _slow_path_worker():
    """
    Consume utterances from the queue and run the Groq slow path.
    Serialised to respect rate limits gracefully.
    """
    while True:
        utterance = await _utterance_queue.get()
        session_id = utterance.session_id
        ctx = _sessions.get(session_id)

        try:
            logger.info(
                "Slow path starting for session %s (%d ms).",
                session_id[:8], utterance.audio_duration_ms,
            )

            # 5A: Transcription
            transcript = await transcribe_utterance(utterance)

            # 5B: Judge
            fast_path_history = ctx.fast_path_history.copy() if ctx else []
            verdict = await judge_emotion(session_id, transcript.text, fast_path_history)

            # 6: Store in Supabase
            user_id = ctx.user_id if ctx else None
            row_id = await store_emotion_result(utterance, transcript, verdict, user_id)

            logger.info(
                "Slow path complete [%s]: '%s' → %s (%.2f) | row=%s",
                session_id[:8],
                transcript.text[:60],
                verdict.final_emotion,
                verdict.confidence,
                row_id,
            )

        except Exception as exc:
            logger.error(
                "Slow path error for session %s: %s", session_id[:8], exc, exc_info=True
            )
        finally:
            _utterance_queue.task_done()


# ── Application Lifecycle ─────────────────────────────────────────────────────

@app.on_event("startup")
async def _startup():
    logging.basicConfig(
        level=cfg.LOG_LEVEL,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logger.info("Orchestrator starting (mock=%s).", cfg.MOCK_APIS)
    asyncio.create_task(_poll_event_queues())
    asyncio.create_task(_slow_path_worker())


@app.on_event("shutdown")
async def _shutdown():
    logger.info("Orchestrator shutting down — terminating child processes.")
    # Cancel bot tasks
    for task in list(_bot_tasks.values()):
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    # Terminate gateway and signal processes
    for proc in list(_gateway_processes.values()) + list(_signal_processes.values()):
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=3)


# ── REST Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check():
    """
    Simple health check endpoint for deployment monitoring.
    Returns: {"status": "ok", ...Groq quota and session stats}
    """
    rl = get_rate_limiter()
    return {
        "status": "ok",
        "active_sessions": len(_sessions),
        "groq_requests_today": rl.requests_today,
        "groq_daily_limit": rl.daily_limit,
        "groq_requests_remaining": rl.requests_remaining,
        "mock_mode": cfg.MOCK_APIS,
    }


@app.post("/session/analyze-file", tags=["System"])
async def analyze_file(file: UploadFile = File(...)):
    """
    Analyze a WAV file — hardened pipeline that mirrors the live microphone flow.

    Pipeline:
      Phase 1: Pre-flight silence check (RMS energy)
      Phase 2: VAD segmentation + SenseVoice fast-path triggers
      Phase 3: Speech-only concatenation → Groq Whisper (no hallucination)
      Phase 4: Acoustic-enriched Groq Llama Judge
      Phase 5: Store complete analysis in Supabase
    """
    import io
    import time as _time

    import numpy as np

    from services.signal_processor.cleaner import TARGET_SAMPLE_RATE
    from services.signal_processor.embedder import (
        run_local_inference,
        get_top_emotion,
        mock_inference,
        EMOTION_LABELS,
    )

    session_id = str(uuid.uuid4())
    wav_bytes = await file.read()

    # ── Phase 1: Load WAV → float32 numpy array ──────────────────────────────
    try:
        import librosa  # type: ignore

        audio_f32, orig_sr = librosa.load(
            io.BytesIO(wav_bytes), sr=None, mono=True
        )
    except Exception as exc:
        logger.warning("[AnalyzeFile] librosa load failed: %s", exc)
        raise HTTPException(status_code=400, detail=f"Invalid audio file: {exc}")

    duration_s = len(audio_f32) / max(orig_sr, 1)
    duration_ms = duration_s * 1000

    logger.info(
        "[AnalyzeFile] %s | %.1f ms audio @ %d Hz (%d samples)",
        session_id[:8], duration_ms, orig_sr, len(audio_f32),
    )

    # ── Phase 1: RMS silence check ────────────────────────────────────────────
    rms = float(np.sqrt(np.mean(audio_f32 ** 2)))
    logger.info("[AnalyzeFile] %s | RMS energy = %.6f", session_id[:8], rms)

    RMS_SILENCE_THRESHOLD = 0.002
    if rms < RMS_SILENCE_THRESHOLD:
        logger.warning(
            "[AnalyzeFile] %s | REJECTED — audio is silent (RMS %.6f < %.4f)",
            session_id[:8], rms, RMS_SILENCE_THRESHOLD,
        )
        raise HTTPException(
            status_code=400,
            detail="Audio is silent — no meaningful content detected.",
        )

    # ── Phase 2: Resample to 16kHz for VAD + inference ────────────────────────
    if orig_sr != TARGET_SAMPLE_RATE:
        import librosa as _lr

        audio_16k = _lr.resample(
            audio_f32, orig_sr=orig_sr, target_sr=TARGET_SAMPLE_RATE
        )
        logger.info(
            "[AnalyzeFile] %s | Resampled %d Hz → %d Hz",
            session_id[:8], orig_sr, TARGET_SAMPLE_RATE,
        )
    else:
        audio_16k = audio_f32

    # ── Phase 2: VAD — detect speech segments ─────────────────────────────────
    speech_segments = _vad_detect_speech(audio_16k, TARGET_SAMPLE_RATE, session_id)
    logger.info(
        "[AnalyzeFile] %s | VAD found %d speech segment(s)",
        session_id[:8], len(speech_segments),
    )

    if not speech_segments:
        logger.warning(
            "[AnalyzeFile] %s | REJECTED — no speech detected by VAD",
            session_id[:8],
        )
        raise HTTPException(
            status_code=400,
            detail="No speech detected in the audio file.",
        )

    # ── Phase 2: Run SenseVoice ONNX on 500ms windows → FastPathTriggers ─────
    WINDOW_MS = 500
    window_samples = int(TARGET_SAMPLE_RATE * WINDOW_MS / 1000)  # 8000
    fast_path_history: list[FastPathTrigger] = []
    window_index = 0

    for seg_start, seg_end in speech_segments:
        segment_audio = audio_16k[seg_start:seg_end]
        offset = 0
        while offset < len(segment_audio):
            chunk = segment_audio[offset : offset + window_samples]
            if len(chunk) < window_samples // 4:
                break  # Skip tiny tail fragments

            if cfg.MOCK_APIS:
                probs = mock_inference(len(chunk))
            else:
                probs = run_local_inference(chunk, TARGET_SAMPLE_RATE)
                if probs is None:
                    probs = mock_inference(len(chunk))

            top_emotion, confidence = get_top_emotion(probs)
            trigger = FastPathTrigger(
                session_id=session_id,
                window_index=window_index,
                emotion_logits=probs,
                top_emotion=top_emotion,
                top_confidence=confidence,
                audio_duration_ms=float(WINDOW_MS),
            )
            fast_path_history.append(trigger)
            window_index += 1
            offset += window_samples

    logger.info(
        "[AnalyzeFile] %s | Generated %d fast-path windows",
        session_id[:8], len(fast_path_history),
    )

    # ── Phase 3: Concatenate speech-only audio for Groq Whisper ───────────────
    speech_chunks = [audio_16k[s:e] for s, e in speech_segments]
    speech_only = np.concatenate(speech_chunks)
    speech_duration_ms = len(speech_only) / TARGET_SAMPLE_RATE * 1000

    # Convert to 16-bit PCM bytes for the Whisper transport
    pcm_int16 = (speech_only * 32767).clip(-32768, 32767).astype(np.int16)
    speech_pcm_bytes = pcm_int16.tobytes()

    logger.info(
        "[AnalyzeFile] %s | Speech-only audio: %.0f ms (trimmed from %.0f ms)",
        session_id[:8], speech_duration_ms, duration_ms,
    )

    utterance = UtteranceComplete(
        session_id=session_id,
        pcm_bytes=speech_pcm_bytes,
        sample_rate=TARGET_SAMPLE_RATE,
        audio_duration_ms=speech_duration_ms,
        timestamp_ms=_time.time() * 1000,
        window_emotions=fast_path_history,
    )

    t0 = _time.perf_counter()
    transcript = await transcribe_utterance(utterance)
    transcription_ms = (_time.perf_counter() - t0) * 1000

    # Phase 3: Anti-hallucination guard on the transcript
    cleaned_text = transcript.text.strip().strip(".,!?;:")
    if len(cleaned_text) <= 1:
        logger.info(
            "[AnalyzeFile] %s | Transcript too short/empty → '(unintelligible)'",
            session_id[:8],
        )
        transcript = TranscriptionResult(
            session_id=session_id,
            text="(unintelligible)",
            latency_ms=transcript.latency_ms,
        )

    logger.info(
        "[AnalyzeFile] %s | Whisper transcript: '%s' (%.0f ms)",
        session_id[:8], transcript.text[:80], transcription_ms,
    )

    # ── Phase 4: Acoustic-enriched judge ──────────────────────────────────────
    # Compute acoustic metadata for the judge
    energy_level = "low" if rms < 0.01 else ("medium" if rms < 0.05 else "high")
    word_count = len(transcript.text.split())
    speaking_rate = (
        round(word_count / (speech_duration_ms / 1000), 1)
        if speech_duration_ms > 0 else 0.0
    )

    verdict = await judge_emotion(
        session_id,
        transcript.text,
        fast_path_history,
        rms_energy=rms,
        energy_level=energy_level,
        speaking_rate=speaking_rate,
    )

    # ── Phase 5: Store in Supabase ────────────────────────────────────────────
    try:
        row_id = await store_emotion_result(
            utterance, transcript, verdict, user_id=None
        )
    except Exception as exc:
        logger.warning("[AnalyzeFile] Supabase store failed: %s", exc)
        row_id = None

    logger.info(
        "[AnalyzeFile] %s | COMPLETE: '%s' → %s (%.2f) | %d windows | row=%s",
        session_id[:8],
        transcript.text[:60],
        verdict.final_emotion,
        verdict.confidence,
        len(fast_path_history),
        row_id,
    )

    return {
        "session_id": session_id,
        "filename": file.filename,
        "duration_ms": round(duration_ms),
        "speech_duration_ms": round(speech_duration_ms),
        "sample_rate": orig_sr,
        "rms_energy": round(rms, 6),
        "speech_segments": len(speech_segments),
        "fast_path_windows": len(fast_path_history),
        "transcript": transcript.text,
        "final_emotion": verdict.final_emotion.value,
        "confidence": verdict.confidence,
        "reasoning": verdict.reasoning,
        "fast_path_summary": verdict.fast_path_summary,
        "emotion_timeline": [
            {
                "window": fp.window_index,
                "emotion": fp.top_emotion.value,
                "confidence": round(fp.top_confidence, 3),
            }
            for fp in fast_path_history
        ],
        "supabase_row": row_id,
        "transcription_latency_ms": round(transcription_ms),
    }


# ── Static VAD helper for file analysis ──────────────────────────────────────

def _vad_detect_speech(
    audio_16k: "np.ndarray",
    sample_rate: int,
    session_id: str,
    threshold: float = 0.5,
    min_speech_ms: int = 250,
) -> list[tuple[int, int]]:
    """
    Run Silero VAD over a complete 16kHz float32 audio array and return
    a list of (start_sample, end_sample) tuples for each speech segment.

    This is a static (non-streaming) wrapper around the same Silero model
    used by the live VAD engine, designed for offline file analysis.
    """
    import numpy as np

    from services.signal_processor.vad_engine import (
        _silero_speech_prob,
        SILERO_CHUNK_FRAMES,
    )

    chunk_size = SILERO_CHUNK_FRAMES  # 512 samples = 32ms @ 16kHz
    n_samples = len(audio_16k)
    merge_gap_samples = int(sample_rate * 0.3)  # merge segments < 300ms apart
    min_speech_samples = int(sample_rate * min_speech_ms / 1000)

    # Step 1: Classify each 32ms chunk as speech or silence
    is_speech: list[bool] = []
    for start in range(0, n_samples, chunk_size):
        chunk = audio_16k[start : start + chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        prob = _silero_speech_prob(chunk)
        is_speech.append(prob >= threshold)

    # Step 2: Build raw segments from consecutive speech chunks
    raw_segments: list[tuple[int, int]] = []
    in_speech = False
    seg_start = 0
    for i, speech in enumerate(is_speech):
        sample_pos = i * chunk_size
        if speech and not in_speech:
            seg_start = sample_pos
            in_speech = True
        elif not speech and in_speech:
            raw_segments.append((seg_start, sample_pos))
            in_speech = False
    if in_speech:
        raw_segments.append((seg_start, n_samples))

    # Step 3: Merge segments that are close together
    merged: list[tuple[int, int]] = []
    for seg in raw_segments:
        if merged and (seg[0] - merged[-1][1]) < merge_gap_samples:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(seg)

    # Step 4: Filter out segments shorter than min_speech_ms
    final = [(s, e) for s, e in merged if (e - s) >= min_speech_samples]

    logger.info(
        "[AnalyzeFile] %s | VAD: %d raw → %d merged → %d final segments",
        session_id[:8], len(raw_segments), len(merged), len(final),
    )
    return final


@app.post("/session/start", tags=["Session"])
async def session_start(request: Request, authorization: Optional[str] = Header(None)):
    """
    Initialize a new emotion detection session.

    1. Generates a unique session_id and LiveKit room name.
    2. Creates a user-facing LiveKit access token (publish rights).
    3. Spawns the LiveKit bot listener as a background task.
    4. Spawns the signal processor subprocess.

    Returns:
      - session_id: unique identifier for this session
      - livekit_token: JWT for the browser to connect to the LiveKit room
      - livekit_url: LiveKit WebSocket URL the browser should connect to
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    session_id = str(uuid.uuid4())
    room_name = f"emotion-room-{session_id}"

    # Validate Supabase JWT if provided
    user_id: Optional[str] = None
    if authorization and authorization.startswith("Bearer "):
        token_str = authorization[7:]
        user_id = await verify_user_token(token_str)
        if user_id:
            logger.info(
                "Authenticated user %s started session %s.", user_id[:8], session_id[:8]
            )

    # ── Create session context ────────────────────────────────────────────────
    ctx = SessionContext(
        session_id=session_id,
        user_id=user_id or body.get("user_id"),
        livekit_room=room_name,
    )
    _sessions[session_id] = ctx

    # ── Generate user token (can publish, cannot subscribe) ───────────────────
    livekit_token = _generate_user_livekit_token(session_id, room_name)
    livekit_url = cfg.LIVEKIT_URL or os.getenv("LIVEKIT_WS_URL", "")

    # ── Signal processor subprocess ───────────────────────────────────────────
    audio_queue: multiprocessing.Queue = multiprocessing.Queue(maxsize=500)
    event_queue: multiprocessing.Queue = multiprocessing.Queue(maxsize=200)
    _event_queues[session_id] = event_queue

    sig_proc = multiprocessing.Process(
        target=run_signal_processor,
        args=(session_id, audio_queue, event_queue, cfg.MOCK_APIS),
        daemon=True,
        name=f"signal-processor-{session_id[:8]}",
    )
    sig_proc.start()
    _signal_processes[session_id] = sig_proc

    # ── LiveKit bot (background asyncio task) ─────────────────────────────────
    if not cfg.MOCK_APIS:
        bot_task = asyncio.create_task(
            start_livekit_bot(room_name, session_id, audio_queue),
            name=f"livekit-bot-{session_id[:8]}",
        )
        _bot_tasks[session_id] = bot_task

        # Also start the legacy gateway in a subprocess (handles audio routing)
        gateway_proc = multiprocessing.Process(
            target=_run_gateway_sync,
            args=(session_id, audio_queue),
            daemon=True,
            name=f"livekit-gateway-{session_id[:8]}",
        )
        gateway_proc.start()
        _gateway_processes[session_id] = gateway_proc
    else:
        # In mock mode, start the gateway in an async task only
        asyncio.create_task(run_gateway(session_id, audio_queue))

    logger.info(
        "Session %s started. Room=%s. Bot + signal processor launched.",
        session_id[:8], room_name,
    )

    return {
        "session_id": session_id,
        "livekit_token": livekit_token,
        "livekit_url": livekit_url,
        "room_name": room_name,
        "mock_mode": cfg.MOCK_APIS,
    }


@app.post("/session/end", tags=["Session"])
async def session_end(request: Request):
    """Terminate a session and release all resources."""
    body = await request.json()
    session_id = body.get("session_id", "")
    return await _finalize_session_logic(session_id)


@app.post("/session/{session_id}/finalize", tags=["Session"])
async def session_finalize(session_id: str):
    """Compatibility endpoint for session finalization via path parameter."""
    return await _finalize_session_logic(session_id)


async def _finalize_session_logic(session_id: str):
    if not session_id or session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    # Cancel bot task
    bot_task = _bot_tasks.pop(session_id, None)
    if bot_task and not bot_task.done():
        bot_task.cancel()

    # Terminate child processes
    for proc_dict in [_gateway_processes, _signal_processes]:
        proc = proc_dict.pop(session_id, None)
        if proc and proc.is_alive():
            proc.terminate()
            proc.join(timeout=3)

    _event_queues.pop(session_id, None)
    ctx = _sessions.pop(session_id, None)

    logger.info(
        "Session %s ended. Processed %d fast-path windows.",
        session_id[:8],
        len(ctx.fast_path_history) if ctx else 0,
    )
    return {"status": "ended", "session_id": session_id}


@app.get("/session/{session_id}/status", tags=["Session"])
async def session_status(session_id: str):
    """Get current session state including VAD state and latest fast-path emotion."""
    ctx = _sessions.get(session_id)
    if not ctx:
        raise HTTPException(status_code=404, detail="Session not found.")

    last_emotion = None
    if ctx.fast_path_history:
        last = ctx.fast_path_history[-1]
        last_emotion = {
            "emotion": last.top_emotion,
            "confidence": last.top_confidence,
            "window_index": last.window_index,
        }

    return {
        "session_id": session_id,
        "vad_state": ctx.vad_state,
        "fast_path_windows": len(ctx.fast_path_history),
        "latest_emotion": last_emotion,
        "user_id": ctx.user_id,
    }


@app.get("/session/{session_id}/report", tags=["Report"])
async def session_report(session_id: str):
    """
    Return the full analysis data for a completed session.
    Consumed by result_page.html to populate the Neubrutalist dashboard.

    Returns a JSON object with fields matching the frontend's expected schema.
    """
    record = await get_session_history(session_id)

    if not record:
        raise HTTPException(
            status_code=404,
            detail=f"No session data found for session_id={session_id}",
        )

    # ── Normalise DB record → frontend schema ──────────────────────────────────
    duration_ms: int = record.get("audio_duration_ms", 0) or record.get("duration_ms", 0)
    duration_s: float = round(duration_ms / 1000, 1)

    transcript: str = record.get("utterance_text", record.get("transcript", ""))
    final_emotion: str = record.get("final_emotion", "Neutral")
    confidence: float = float(record.get("confidence", 0.0))
    reasoning: str = record.get("judge_reasoning", record.get("reasoning", ""))

    # Breakdown / timeline (stored as JSONB; fall back to derived values)
    emotion_breakdown: dict = record.get("emotion_breakdown") or {
        final_emotion: 100
    }
    timeline: list = record.get("emotion_timeline") or [
        {"time": 0, final_emotion: confidence, "Neutral": 1 - confidence},
        {"time": duration_s, final_emotion: confidence, "Neutral": 1 - confidence},
    ]

    peak_emotion_data: dict = record.get("peak_emotion") or {
        "emotion": final_emotion,
        "confidence": confidence,
        "timestamp": f"0:{int(duration_s):02d}",
    }

    insights_list: list = record.get("insights") or [
        f"Primary emotion detected: {final_emotion}",
        reasoning[:120] if reasoning else "Analysis completed.",
    ]

    # Sarcasm flags
    sarcasm_detected: bool = bool(
        record.get("sarcasm_detected", False)
        or final_emotion in ("Sarcastic", "Passive-Aggressive")
    )
    sarcasm_note: str = record.get("sarcasm_note", "")
    if sarcasm_detected and not sarcasm_note and reasoning:
        sarcasm_note = reasoning

    created_at = record.get("created_at", "")
    # Format as human-readable timestamp
    from datetime import datetime, timezone
    try:
        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        day = dt.day
        month_year = dt.strftime("%b %Y")
        hour_min = dt.strftime("%I:%M %p").lstrip("0")
        ts_label = f"{day} {month_year}, {hour_min}"
    except Exception:
        ts_label = created_at or "—"

    word_count = len(transcript.split()) if transcript else 0

    return {
        "session_id": session_id,
        "timestamp": ts_label,
        "duration": duration_s,
        "transcript": transcript,
        "language": record.get("language", "English"),
        "wordCount": word_count,
        "overallEmotion": final_emotion,
        "overallConfidence": confidence,
        "peakEmotion": peak_emotion_data,
        "emotions": emotion_breakdown,
        "timeline": timeline,
        "sarcasmDetected": sarcasm_detected,
        "sarcasmNote": sarcasm_note,
        "insights": insights_list,
        "segments": record.get("speech_segments", 1),
        "avgConfidence": float(record.get("avg_confidence", confidence)),
        "dominantEmotion": record.get("dominant_emotion", final_emotion),
        "variability": record.get("variability", "Moderate"),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _generate_user_livekit_token(session_id: str, room_name: str) -> str:
    """
    Generate a LiveKit access token for the browser participant.
    The user can publish (microphone) but not subscribe to other tracks.
    """
    try:
        from livekit import api as lk_api  # type: ignore

        token = (
            lk_api.AccessToken(cfg.LIVEKIT_API_KEY, cfg.LIVEKIT_API_SECRET)
            .with_identity(f"user-{session_id[:8]}")
            .with_name("Demo User")
            .with_grants(
                lk_api.VideoGrants(
                    room_join=True,
                    room=room_name,
                    can_publish=True,
                    can_subscribe=False,
                )
            )
            .to_jwt()
        )
        return token
    except Exception as exc:
        logger.warning("Could not generate LiveKit user token: %s", exc)
        return "mock-token" if cfg.MOCK_APIS else "unavailable"


def _run_gateway_sync(session_id: str, audio_queue: multiprocessing.Queue) -> None:
    """Synchronous entry point for gateway subprocess."""
    import asyncio
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from services.livekit_gateway.main import run_gateway

    asyncio.run(run_gateway(session_id, audio_queue))


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "services.orchestrator.main:app",
        host="0.0.0.0",
        port=cfg.ORCHESTRATOR_PORT,
        log_level=cfg.LOG_LEVEL.lower(),
        reload=False,
    )
