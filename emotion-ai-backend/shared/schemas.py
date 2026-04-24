"""
shared/schemas.py
─────────────────
Pydantic v2 data contracts for all inter-process and inter-module messages.
These models are the single source of truth for data flowing between:
  Module 1 (LiveKit Gateway) → Module 2 (Signal Processor)
  Module 2 → Module 3 (Orchestrator)
  Module 3 → Module 5 (Groq)
  Module 3 → Module 6 (Supabase)
"""

from __future__ import annotations

import time
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


# ── VAD State Machine States ─────────────────────────────────────────────────

class VADState(str, Enum):
    SILENCE = "silence"
    SPEECH_START = "speech_start"
    SPEAKING = "speaking"
    SPEECH_END = "speech_end"


# ── Emotion Labels ────────────────────────────────────────────────────────────

class Emotion(str, Enum):
    NEUTRAL = "Neutral"
    HAPPY = "Happy"
    SAD = "Sad"
    ANGRY = "Angry"
    SURPRISED = "Surprised"
    UNKNOWN = "Unknown"


# ── Raw PCM Audio Chunk (Module 1 → Module 2) ────────────────────────────────

class AudioChunk(BaseModel):
    """A single decoded PCM frame from the LiveKit gateway."""
    session_id: str
    participant_sid: str
    timestamp_ms: float = Field(default_factory=lambda: time.time() * 1000)
    sample_rate: int = 48_000          # Raw Opus decode rate from LiveKit
    channels: int = 1
    pcm_bytes: bytes                   # Raw 16-bit little-endian PCM


# ── Signal Processor Events (Module 2 → Module 3) ────────────────────────────

class FastPathTrigger(BaseModel):
    """
    Emitted every 500 ms while speech is active.
    Contains the SenseVoice-Small emotion logits (5 classes).
    """
    session_id: str
    window_index: int                        # 0-based count of 500ms windows
    timestamp_ms: float = Field(default_factory=lambda: time.time() * 1000)
    emotion_logits: List[float]              # [Neutral, Happy, Sad, Angry, Surprised]
    top_emotion: Emotion
    top_confidence: float
    audio_duration_ms: float = 500.0


class UtteranceComplete(BaseModel):
    """
    Emitted when VAD detects 600 ms of post-speech silence.
    Contains the full utterance PCM buffer and per-window embeddings.
    """
    session_id: str
    timestamp_ms: float = Field(default_factory=lambda: time.time() * 1000)
    pcm_bytes: bytes                         # 16kHz mono 16-bit PCM for full utterance
    audio_duration_ms: float
    window_emotions: List[FastPathTrigger]   # History of fast-path results
    sample_rate: int = 16_000


class VADStateChange(BaseModel):
    """Published by the signal processor whenever the VAD state changes."""
    session_id: str
    previous_state: VADState
    new_state: VADState
    timestamp_ms: float = Field(default_factory=lambda: time.time() * 1000)


# ── Orchestrator Session State ────────────────────────────────────────────────

class SessionContext(BaseModel):
    """In-memory session state held by the Orchestrator."""
    session_id: str
    user_id: Optional[str] = None
    participant_sid: Optional[str] = None
    vad_state: VADState = VADState.SILENCE
    fast_path_history: List[FastPathTrigger] = Field(default_factory=list)
    utterance_buffer: bytes = b""
    utterance_duration_ms: float = 0.0
    created_at: float = Field(default_factory=time.time)
    livekit_room: Optional[str] = None


# ── Groq API Contracts ────────────────────────────────────────────────────────

class TranscriptionResult(BaseModel):
    """Structured result from Groq Whisper transcription."""
    session_id: str
    text: str
    language: Optional[str] = None
    duration_seconds: Optional[float] = None
    latency_ms: float = 0.0


class JudgeVerdict(BaseModel):
    """
    Parsed output from the Groq Llama 3.3 70B judge.
    The LLM is instructed to respond with JSON matching this schema.
    """
    final_emotion: Emotion
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    fast_path_summary: Optional[str] = None


# ── Supabase Row Model ────────────────────────────────────────────────────────

class EmotionSessionRow(BaseModel):
    """Maps to the `emotion_sessions` table in Supabase."""
    id: Optional[str] = None            # uuid, set by DB
    user_id: Optional[str] = None       # uuid from Supabase Auth
    utterance_text: str = ""
    fast_path_emotion: str = Emotion.UNKNOWN
    final_emotion: str = Emotion.UNKNOWN
    confidence: float = 0.0
    judge_reasoning: str = ""
    audio_duration_ms: int = 0
    session_id: str = ""


# ── Health / Status ───────────────────────────────────────────────────────────

class ServiceHealth(BaseModel):
    """Returned by GET /health on the Orchestrator."""
    status: str = "ok"
    active_sessions: int = 0
    groq_requests_today: int = 0
    groq_daily_limit: int = 14_400
    groq_requests_remaining: int = 14_400
    mock_mode: bool = False
    livekit_minutes_used: float = 0.0
    livekit_minute_limit: float = 50.0
