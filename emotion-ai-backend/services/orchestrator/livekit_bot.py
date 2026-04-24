"""
services/orchestrator/livekit_bot.py
──────────────────────────────────────
Module 1 (Bot Side) — LiveKit Bot Participant

Connects to a LiveKit room as a bot listener, subscribes to the user's
audio track, and pushes raw PCM frames into the audio_processor queue
so the signal processor pipeline can consume them.

Usage:
    from services.orchestrator.livekit_bot import start_livekit_bot
    asyncio.create_task(start_livekit_bot(room_name, session_id, audio_queue))
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

logger = logging.getLogger("orchestrator.livekit_bot")


async def start_livekit_bot(
    room_name: str,
    session_id: str,
    audio_queue: asyncio.Queue,
) -> None:
    """
    Connect to a LiveKit room as a non-publishing bot participant.

    Subscribes to the user's audio track and pushes each PCM frame
    as a dict into audio_queue:
        {
            "session_id": str,
            "pcm": bytes,
            "sample_rate": int,
            "channels": int,
        }

    Args:
        room_name:   The LiveKit room to join (e.g. "emotion-room-<uuid>").
        session_id:  The active session UUID, used to tag audio frames.
        audio_queue: asyncio.Queue shared with the signal processor.
    """
    api_key = os.getenv("LIVEKIT_API_KEY", "")
    api_secret = os.getenv("LIVEKIT_API_SECRET", "")
    ws_url = os.getenv("LIVEKIT_WS_URL", os.getenv("LIVEKIT_URL", ""))

    if not all([api_key, api_secret, ws_url]):
        logger.error(
            "LiveKit credentials missing (LIVEKIT_API_KEY / LIVEKIT_API_SECRET / LIVEKIT_WS_URL). "
            "Bot will not start for session %s.",
            session_id[:8],
        )
        return

    try:
        from livekit import rtc, api as lk_api  # type: ignore
    except ImportError:
        logger.error(
            "livekit-rtc not installed. Run: pip install livekit. "
            "Bot cannot start for session %s.",
            session_id[:8],
        )
        return

    # ── Generate bot access token ──────────────────────────────────────────────
    bot_identity = f"bot-{session_id}"
    try:
        token = (
            lk_api.AccessToken(api_key, api_secret)
            .with_identity(bot_identity)
            .with_name("Emotion Bot")
            .with_grants(
                lk_api.VideoGrants(
                    room_join=True,
                    room=room_name,
                    can_publish=False,          # bot only listens
                    can_subscribe=True,
                )
            )
            .to_jwt()
        )
    except Exception as exc:
        logger.error(
            "Failed to generate LiveKit bot token for session %s: %s",
            session_id[:8], exc,
        )
        return

    # ── Create room and register event handlers ────────────────────────────────
    room = rtc.Room()

    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return

        logger.info(
            "Bot subscribed to audio track from %s in session %s.",
            participant.identity, session_id[:8],
        )
        audio_stream = rtc.AudioStream(track)
        asyncio.ensure_future(
            process_audio_stream(audio_stream, session_id, audio_queue)
        )

    @room.on("disconnected")
    def on_disconnected() -> None:
        logger.info("Bot disconnected from room %s (session %s).", room_name, session_id[:8])

    # ── Connect ────────────────────────────────────────────────────────────────
    try:
        logger.info(
            "Bot connecting to LiveKit room '%s' for session %s …",
            room_name, session_id[:8],
        )
        await room.connect(ws_url, token)
        logger.info(
            "Bot connected to room '%s' (session %s). Waiting for audio tracks.",
            room_name, session_id[:8],
        )

        # Keep the bot alive until the room closes or the task is cancelled.
        while room.connection_state != rtc.ConnectionState.CONN_DISCONNECTED:
            await asyncio.sleep(1)

    except asyncio.CancelledError:
        logger.info("Bot task cancelled for session %s.", session_id[:8])
    except Exception as exc:
        logger.error(
            "Bot connection error for session %s: %s",
            session_id[:8], exc, exc_info=True,
        )
    finally:
        try:
            await room.disconnect()
        except Exception:
            pass


async def process_audio_stream(
    stream: "rtc.AudioStream",  # noqa: F821
    session_id: str,
    audio_queue: asyncio.Queue,
) -> None:
    """
    Consume audio frames from a LiveKit AudioStream and push raw PCM
    into the shared audio_queue for the signal processor.

    Each item pushed is a plain dict:
        {
            "session_id": str,
            "pcm": bytes,           # 16-bit little-endian PCM
            "sample_rate": int,
            "channels": int,
        }
    """
    logger.debug("Audio stream reader started for session %s.", session_id[:8])
    try:
        async for frame_event in stream:
            frame = frame_event.frame
            try:
                # AudioFrame exposes .data (bytes), .sample_rate, .num_channels
                pcm_bytes = bytes(frame.data)
                sample_rate: int = frame.sample_rate
                channels: int = frame.num_channels

                payload = {
                    "session_id": session_id,
                    "pcm": pcm_bytes,
                    "sample_rate": sample_rate,
                    "channels": channels,
                }

                # Non-blocking put; drop frame if the queue is full to avoid
                # memory build-up under heavy load.
                try:
                    audio_queue.put_nowait(payload)
                except asyncio.QueueFull:
                    logger.debug(
                        "Audio queue full — dropping frame for session %s.",
                        session_id[:8],
                    )

            except Exception as frame_exc:
                logger.warning(
                    "Error processing audio frame for session %s: %s",
                    session_id[:8], frame_exc,
                )

    except asyncio.CancelledError:
        logger.info("Audio stream reader cancelled for session %s.", session_id[:8])
    except Exception as exc:
        logger.error(
            "Unexpected error in audio stream for session %s: %s",
            session_id[:8], exc, exc_info=True,
        )
    finally:
        logger.debug("Audio stream reader exiting for session %s.", session_id[:8])
