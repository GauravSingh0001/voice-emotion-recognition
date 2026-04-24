"""
services/livekit_gateway/main.py
────────────────────────────────
Module 1 — LiveKit Cloud Integration (WebRTC Gateway)

Responsibilities:
  1. Connect to a LiveKit room using the server-side Python SDK.
  2. Subscribe to the participant's audio track.
  3. Decode incoming Opus frames to 16-bit PCM.
  4. Write raw PCM chunks onto a multiprocessing.Queue for Module 2.

Run this as its own process (spawned by orchestrator/main.py or directly).
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from multiprocessing import Queue
from pathlib import Path

# Ensure shared/ is on the path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared.config import get_settings
from shared.schemas import AudioChunk

logger = logging.getLogger("livekit_gateway")

# ── LiveKit Monthly Usage Tracker ────────────────────────────────────────────
# Persisted in a simple JSON sidecar file.
_USAGE_FILE = Path(__file__).parent / ".livekit_usage.json"


def _load_livekit_usage() -> dict:
    import json
    if _USAGE_FILE.exists():
        try:
            return json.loads(_USAGE_FILE.read_text())
        except Exception:
            pass
    return {"minutes_used": 0.0, "month": time.strftime("%Y-%m")}


def _save_livekit_usage(data: dict) -> None:
    import json
    _USAGE_FILE.write_text(json.dumps(data))


def _check_livekit_limit(minutes_used: float, limit: float) -> bool:
    """Return True if within limit; log warning if approaching."""
    remaining = limit - minutes_used
    if remaining <= 0:
        logger.error(
            "LiveKit free-tier minute limit reached (%.1f / %.1f min). "
            "Will enter offline/mock mode.",
            minutes_used, limit,
        )
        return False
    if remaining < limit * 0.1:  # < 10% remaining
        logger.warning(
            "LiveKit minutes running low: %.1f used / %.1f limit (%.1f remaining).",
            minutes_used, limit, remaining,
        )
    return True


# ── Mock Audio Generator ──────────────────────────────────────────────────────

async def _mock_audio_producer(session_id: str, audio_queue: Queue) -> None:
    """Inject WAV file audio for offline testing."""
    import struct
    import math
    import wave

    wav_path = Path("c:/Users/gs667/Documents/Visual/ser/n.wav")
    
    if wav_path.exists():
        logger.info("MOCK MODE: Streaming %s into session %s", wav_path.name, session_id)
        with wave.open(str(wav_path), 'rb') as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            chunk_frames = sample_rate // 10  # 100ms chunks

            while True:
                frames = wf.readframes(chunk_frames)
                if not frames:
                    logger.info("Finished streaming %s", wav_path.name)
                    break
                
                chunk = AudioChunk(
                    session_id=session_id,
                    participant_sid="mock-participant",
                    sample_rate=sample_rate,
                    channels=channels,
                    pcm_bytes=frames,
                )
                audio_queue.put_nowait(chunk)
                await asyncio.sleep(0.1)

    # Keep alive with silence after wav finishes (or if it doesn't exist)
    logger.info("MOCK MODE: Playing background silence...")
    while True:
        await asyncio.sleep(1)


# ── Real LiveKit Audio Receiver ───────────────────────────────────────────────

async def _real_audio_producer(
    session_id: str,
    room_name: str,
    audio_queue: Queue,
) -> None:
    """
    Connect to LiveKit room and stream decoded PCM to audio_queue.
    Uses livekit-api + livekit-rtc Python packages.
    """
    try:
        from livekit import rtc, api  # type: ignore
    except ImportError:
        logger.error(
            "livekit-rtc package not found. "
            "Install it: pip install livekit-rtc livekit-api"
        )
        raise

    cfg = get_settings()
    usage = _load_livekit_usage()

    # Reset monthly counter when month changes
    current_month = time.strftime("%Y-%m")
    if usage.get("month") != current_month:
        usage = {"minutes_used": 0.0, "month": current_month}

    if not _check_livekit_limit(usage["minutes_used"], cfg.LIVEKIT_MONTHLY_MINUTE_LIMIT):
        logger.warning("Falling back to mock audio due to LiveKit limit exhaustion.")
        await _mock_audio_producer(session_id, audio_queue)
        return

    # Generate a short-lived token for the backend bot participant
    token = (
        api.AccessToken(cfg.LIVEKIT_API_KEY, cfg.LIVEKIT_API_SECRET)
        .with_identity("emotion-bot")
        .with_name("Emotion Bot")
        .with_grants(api.VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )

    room = rtc.Room()

    @room.on("track_subscribed")
    def on_track_subscribed(track, publication, participant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info(
                "Subscribed to audio track from participant %s",
                participant.sid,
            )
            asyncio.ensure_future(_stream_audio(track, participant.sid, session_id, audio_queue))

    session_start = time.monotonic()

    try:
        await room.connect(cfg.LIVEKIT_URL, token)
        logger.info("LiveKit gateway connected to room '%s'", room_name)

        # Keep running until cancelled
        while True:
            elapsed_minutes = (time.monotonic() - session_start) / 60.0
            total_used = usage["minutes_used"] + elapsed_minutes
            if not _check_livekit_limit(total_used, cfg.LIVEKIT_MONTHLY_MINUTE_LIMIT):
                break
            await asyncio.sleep(30)  # Check every 30 seconds

    finally:
        elapsed_minutes = (time.monotonic() - session_start) / 60.0
        usage["minutes_used"] += elapsed_minutes
        _save_livekit_usage(usage)
        await room.disconnect()
        logger.info(
            "LiveKit gateway disconnected. Session duration: %.2f minutes. "
            "Total this month: %.2f / %.2f minutes.",
            elapsed_minutes,
            usage["minutes_used"],
            cfg.LIVEKIT_MONTHLY_MINUTE_LIMIT,
        )


async def _stream_audio(
    track,
    participant_sid: str,
    session_id: str,
    audio_queue: Queue,
) -> None:
    """Read audio frames from a LiveKit AudioTrack and enqueue PCM chunks."""
    from livekit import rtc  # type: ignore

    audio_stream = rtc.AudioStream(track)
    async for frame_event in audio_stream:
        frame = frame_event.frame
        chunk = AudioChunk(
            session_id=session_id,
            participant_sid=participant_sid,
            sample_rate=frame.sample_rate,
            channels=frame.num_channels,
            pcm_bytes=bytes(frame.data),
        )
        try:
            audio_queue.put_nowait(chunk)
        except Exception:
            logger.debug("Audio queue full, dropping frame.")


# ── Public Entry Point ────────────────────────────────────────────────────────

async def run_gateway(
    session_id: str,
    audio_queue: Queue,
    room_name: str | None = None,
) -> None:
    """
    Start the LiveKit gateway for a session.
    Call this from orchestrator or run as standalone process.
    """
    cfg = get_settings()
    room = room_name or cfg.LIVEKIT_ROOM_NAME

    if cfg.MOCK_APIS:
        await _mock_audio_producer(session_id, audio_queue)
    else:
        await _real_audio_producer(session_id, room, audio_queue)


# ── Standalone Runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import multiprocessing

    logging.basicConfig(
        level=get_settings().LOG_LEVEL,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    q: Queue = multiprocessing.Queue(maxsize=500)
    asyncio.run(run_gateway(session_id="standalone-test", audio_queue=q))
