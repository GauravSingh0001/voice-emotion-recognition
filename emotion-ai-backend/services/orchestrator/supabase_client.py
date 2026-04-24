"""
services/orchestrator/supabase_client.py
─────────────────────────────────────────
Module 6 — Supabase Integration (Free Backend-as-a-Service)

Responsibilities:
  • Initialize Supabase client with service role key for backend operations.
  • Store EmotionSessionRow records in the `emotion_sessions` table.
  • Broadcast real-time events for UI subscriptions via DB Realtime.
  • Handle authentication JWT validation.
  • Provide mock mode for offline development.

Database Schema (create once in Supabase SQL Editor):
  See create_schema_sql() below.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import Counter
from typing import Any, Dict, List, Optional

from shared.config import get_settings
from shared.schemas import (
    EmotionSessionRow,
    FastPathTrigger,
    JudgeVerdict,
    TranscriptionResult,
    UtteranceComplete,
)

logger = logging.getLogger("orchestrator.supabase")

# ── Mock Storage ──────────────────────────────────────────────────────────────
_mock_store: list[dict] = []

# ── Migration SQL ─────────────────────────────────────────────────────────────

def create_schema_sql() -> str:
    """
    Returns the SQL to create the emotion_sessions table and RLS policies.
    Run this once in the Supabase SQL Editor of your project.

    Steps:
      1. Go to your Supabase project → SQL Editor.
      2. Paste and run this SQL.
      3. In Table Editor → emotion_sessions → Realtime → Enable.
    """
    return """
-- ════════════════════════════════════════════════════════
-- emotion_sessions table — Real-Time Emotion Intelligence
-- ════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS public.emotion_sessions (
    id                  UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             UUID        REFERENCES auth.users(id) ON DELETE CASCADE,
    session_id          TEXT        NOT NULL,
    created_at          TIMESTAMPTZ DEFAULT NOW(),

    -- Core analysis fields
    utterance_text      TEXT        DEFAULT '',
    fast_path_emotion   TEXT        DEFAULT 'Unknown',
    final_emotion       TEXT        DEFAULT 'Unknown',
    confidence          FLOAT       DEFAULT 0.0,
    judge_reasoning     TEXT        DEFAULT '',
    audio_duration_ms   INTEGER     DEFAULT 0,
    language            TEXT        DEFAULT 'English',

    -- Extended analysis fields (JSONB for flexibility)
    emotion_breakdown   JSONB       DEFAULT '{}'::jsonb,
    emotion_timeline    JSONB       DEFAULT '[]'::jsonb,
    peak_emotion        JSONB       DEFAULT '{}'::jsonb,
    insights            JSONB       DEFAULT '[]'::jsonb,

    -- Sarcasm detection
    sarcasm_detected    BOOLEAN     DEFAULT FALSE,
    sarcasm_note        TEXT        DEFAULT '',

    -- Aggregate stats
    speech_segments     INTEGER     DEFAULT 1,
    avg_confidence      FLOAT       DEFAULT 0.0,
    dominant_emotion    TEXT        DEFAULT 'Unknown',
    variability         TEXT        DEFAULT 'Moderate'
);

-- Index for fast session-based lookups
CREATE INDEX IF NOT EXISTS idx_emotion_sessions_session_id
    ON public.emotion_sessions (session_id);

-- Enable Row Level Security
ALTER TABLE public.emotion_sessions ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only read their own rows
CREATE POLICY IF NOT EXISTS "Users read own sessions"
    ON public.emotion_sessions FOR SELECT
    USING (auth.uid() = user_id);

-- Policy: Users can insert their own rows
CREATE POLICY IF NOT EXISTS "Users insert own sessions"
    ON public.emotion_sessions FOR INSERT
    WITH CHECK (auth.uid() = user_id);

-- Grant permissions to authenticated users
GRANT SELECT, INSERT ON public.emotion_sessions TO authenticated;

-- Enable Realtime (triggers INSERT events to subscribed browsers)
ALTER PUBLICATION supabase_realtime ADD TABLE public.emotion_sessions;
"""


# ── Supabase Client Factory ───────────────────────────────────────────────────

_client = None


def _get_client():
    """Lazy-initialize and cache the Supabase Python client."""
    global _client
    if _client is not None:
        return _client

    cfg = get_settings()
    try:
        from supabase import create_client  # type: ignore

        _client = create_client(cfg.SUPABASE_URL, cfg.SUPABASE_SERVICE_ROLE_KEY)
        logger.info("Supabase client initialized for project: %s", cfg.SUPABASE_URL)
        return _client
    except ImportError:
        logger.error("supabase-py not installed. Run: pip install supabase")
        raise
    except Exception as exc:
        logger.error("Failed to initialize Supabase client: %s", exc)
        raise


# ── Database Operations ───────────────────────────────────────────────────────

async def store_emotion_result(
    utterance: UtteranceComplete,
    transcript: TranscriptionResult,
    verdict: JudgeVerdict,
    user_id: Optional[str] = None,
) -> Optional[str]:
    """
    Insert a full emotion session result into Supabase.
    Returns the generated row UUID or None on failure.

    The INSERT triggers a Supabase Realtime event that the browser
    frontend subscribes to for live verdict display.
    """
    cfg = get_settings()

    # ── Compute emotion breakdown from fast-path history ──────────────────────
    window_emotions: List[FastPathTrigger] = utterance.window_emotions or []
    emotion_breakdown: Dict[str, float] = {}
    peak_emotion_data: Dict[str, Any] = {}
    dominant_emotion: str = str(verdict.final_emotion)
    avg_confidence: float = verdict.confidence
    fast_path_emotion: str = str(verdict.final_emotion)

    if window_emotions:
        counts = Counter(fp.top_emotion for fp in window_emotions)
        total = len(window_emotions)
        emotion_breakdown = {str(em): round((cnt / total) * 100, 1) for em, cnt in counts.items()}
        dominant_emotion = str(counts.most_common(1)[0][0])
        fast_path_emotion = dominant_emotion
        avg_confidence = round(
            sum(fp.top_confidence for fp in window_emotions) / total, 4
        )
        # Peak = window with the highest confidence for the dominant emotion
        peak_window = max(
            (fp for fp in window_emotions if fp.top_emotion == dominant_emotion),
            key=lambda fp: fp.top_confidence,
            default=window_emotions[-1],
        )
        peak_time_s = peak_window.window_index * 0.5
        peak_emotion_data = {
            "emotion": str(peak_window.top_emotion),
            "confidence": round(peak_window.top_confidence, 4),
            "timestamp": f"0:{int(peak_time_s):02d}",
        }
    else:
        # Fallback using only the verdict
        emotion_breakdown = {str(verdict.final_emotion): 100.0}
        peak_emotion_data = {
            "emotion": str(verdict.final_emotion),
            "confidence": round(verdict.confidence, 4),
            "timestamp": "0:00",
        }

    # ── Build emotion timeline from window history ─────────────────────────────
    emotion_labels = ["Neutral", "Happy", "Sad", "Angry", "Surprised"]
    timeline: List[Dict[str, Any]] = []
    for fp in window_emotions:
        point: Dict[str, Any] = {"time": round(fp.window_index * 0.5, 1)}
        for label in emotion_labels:
            if str(fp.top_emotion) == label:
                point[label] = round(fp.top_confidence, 3)
            else:
                point[label] = 0.0
        timeline.append(point)

    if not timeline:
        dur_s = utterance.audio_duration_ms / 1000
        timeline = [
            {"time": 0.0, str(verdict.final_emotion): verdict.confidence, "Neutral": 1 - verdict.confidence},
            {"time": round(dur_s, 1), str(verdict.final_emotion): verdict.confidence, "Neutral": 1 - verdict.confidence},
        ]

    # ── Sarcasm detection ─────────────────────────────────────────────────────
    sarcasm_detected = str(verdict.final_emotion) in ("Sarcastic", "Passive-Aggressive")
    sarcasm_note = verdict.reasoning if sarcasm_detected else ""

    # ── Insights ──────────────────────────────────────────────────────────────
    insights: List[str] = []
    if verdict.reasoning:
        insights.append(verdict.reasoning[:200])
    if verdict.fast_path_summary:
        insights.append(verdict.fast_path_summary[:200])
    if not insights:
        insights = [f"Primary emotion: {verdict.final_emotion}"]

    # ── Variability ───────────────────────────────────────────────────────────
    unique_emotions = len(set(str(fp.top_emotion) for fp in window_emotions))
    if unique_emotions >= 3:
        variability = "High"
    elif unique_emotions == 2:
        variability = "Moderate"
    else:
        variability = "Low"

    row = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "session_id": utterance.session_id,
        "utterance_text": transcript.text,
        "fast_path_emotion": fast_path_emotion,
        "final_emotion": str(verdict.final_emotion),
        "confidence": round(verdict.confidence, 4),
        "judge_reasoning": verdict.reasoning,
        "audio_duration_ms": int(utterance.audio_duration_ms),
        # Extended fields
        "emotion_breakdown": emotion_breakdown,
        "emotion_timeline": timeline,
        "peak_emotion": peak_emotion_data,
        "insights": insights,
        "sarcasm_detected": sarcasm_detected,
        "sarcasm_note": sarcasm_note,
        "speech_segments": max(1, len(window_emotions) // 6),
        "avg_confidence": avg_confidence,
        "dominant_emotion": dominant_emotion,
        "variability": variability,
    }

    if cfg.MOCK_APIS:
        _mock_store.append(row)
        logger.info(
            "MOCK: Stored emotion result | %s → %s (%.2f)",
            row["session_id"],
            row["final_emotion"],
            row["confidence"],
        )
        return row["id"]

    try:
        client = _get_client()
        result = client.table("emotion_sessions").insert(row).execute()

        if result.data:
            row_id = result.data[0].get("id", row["id"])
            logger.info(
                "Supabase INSERT OK | session=%s emotion=%s confidence=%.2f row_id=%s",
                utterance.session_id,
                verdict.final_emotion,
                verdict.confidence,
                row_id,
            )
            return row_id
        else:
            logger.warning("Supabase INSERT returned no data: %s", result)
            return None

    except Exception as exc:
        logger.error("Supabase INSERT failed: %s", exc, exc_info=True)
        return None


async def get_session_history(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch the most recent emotion session record for a given session_id.
    Returns a dict matching the Supabase row schema, or None if not found.

    Used by GET /session/{session_id}/report.
    """
    cfg = get_settings()

    if cfg.MOCK_APIS:
        # Search mock store in reverse (most recent first)
        for record in reversed(_mock_store):
            if record.get("session_id") == session_id:
                return record
        # Return a synthetic record for the demo session
        return {
            "session_id": session_id,
            "utterance_text": "This is a mock transcription for offline testing.",
            "final_emotion": "Neutral",
            "confidence": 0.85,
            "judge_reasoning": "Mock mode: acoustic features suggest neutral emotional state.",
            "audio_duration_ms": 5000,
            "emotion_breakdown": {"Neutral": 85.0, "Happy": 10.0, "Sad": 5.0},
            "emotion_timeline": [
                {"time": 0.0, "Neutral": 0.85, "Happy": 0.1, "Sad": 0.05},
                {"time": 2.5, "Neutral": 0.80, "Happy": 0.15, "Sad": 0.05},
                {"time": 5.0, "Neutral": 0.85, "Happy": 0.1, "Sad": 0.05},
            ],
            "peak_emotion": {"emotion": "Neutral", "confidence": 0.90, "timestamp": "0:03"},
            "insights": ["Neutral emotional state throughout the utterance."],
            "sarcasm_detected": False,
            "sarcasm_note": "",
            "speech_segments": 1,
            "avg_confidence": 0.85,
            "dominant_emotion": "Neutral",
            "variability": "Low",
            "created_at": "",
        }

    try:
        client = _get_client()
        result = (
            client.table("emotion_sessions")
            .select("*")
            .eq("session_id", session_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if result.data:
            return result.data[0]
        return None
    except Exception as exc:
        logger.error("Supabase SELECT failed for session %s: %s", session_id, exc)
        return None


async def get_user_history(user_id: str, limit: int = 20) -> list[dict]:
    """Fetch recent emotion session rows for a user (ordered by most recent)."""
    cfg = get_settings()

    if cfg.MOCK_APIS:
        return [r for r in _mock_store if r.get("user_id") == user_id][-limit:]

    try:
        client = _get_client()
        result = (
            client.table("emotion_sessions")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data or []
    except Exception as exc:
        logger.error("Supabase SELECT failed: %s", exc)
        return []


async def verify_user_token(jwt_token: str) -> Optional[str]:
    """
    Validate a Supabase JWT and return the user_id (UUID string) or None.
    """
    cfg = get_settings()

    if cfg.MOCK_APIS:
        return "mock-user-id-00000000-0000-0000-0000-000000000001"

    try:
        client = _get_client()
        user_response = client.auth.get_user(jwt_token)
        if user_response and user_response.user:
            return str(user_response.user.id)
    except Exception as exc:
        logger.warning("Token verification failed: %s", exc)
    return None
