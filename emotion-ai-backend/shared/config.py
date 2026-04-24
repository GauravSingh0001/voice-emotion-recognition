"""
shared/config.py
────────────────
Centralised configuration loader.
Reads `.env` (via python-dotenv), validates required keys, and exposes a
`Settings` singleton used by every module.
"""

from __future__ import annotations

import os
import logging
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (two levels up from shared/)
_root = Path(__file__).resolve().parent.parent
load_dotenv(_root / ".env", override=False)


class Settings:
    """Parsed and validated application settings."""

    # ── LiveKit ───────────────────────────────────────────────────────────────
    LIVEKIT_URL: str = os.getenv("LIVEKIT_URL", "")
    LIVEKIT_API_KEY: str = os.getenv("LIVEKIT_API_KEY", "")
    LIVEKIT_API_SECRET: str = os.getenv("LIVEKIT_API_SECRET", "")
    LIVEKIT_ROOM_NAME: str = os.getenv("LIVEKIT_ROOM_NAME", "emotion-demo")

    # ── Groq ──────────────────────────────────────────────────────────────────
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

    # ── Supabase ──────────────────────────────────────────────────────────────
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_ANON_KEY: str = os.getenv("SUPABASE_ANON_KEY", "")
    SUPABASE_SERVICE_ROLE_KEY: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

    # ── Hugging Face (fallback) ───────────────────────────────────────────────
    HF_API_TOKEN: str = os.getenv("HF_API_TOKEN", "")
    HF_EMOTION_MODEL: str = os.getenv(
        "HF_EMOTION_MODEL",
        "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
    )

    # ── Application ───────────────────────────────────────────────────────────
    ORCHESTRATOR_PORT: int = int(os.getenv("ORCHESTRATOR_PORT", "8000"))
    MOCK_APIS: bool = os.getenv("MOCK_APIS", "false").lower() == "true"
    RECORD_UTTERANCES: bool = os.getenv("RECORD_UTTERANCES", "false").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

    # ── Groq Rate-Limit Tracker (module-level, shared across import) ──────────
    GROQ_DAILY_LIMIT: int = 14_400
    GROQ_WARN_THRESHOLD: float = 0.80  # Warn at 80% usage

    # ── LiveKit Free-Tier ─────────────────────────────────────────────────────
    LIVEKIT_MONTHLY_MINUTE_LIMIT: float = 50.0  # minutes per month

    def validate(self) -> None:
        """
        Warn about missing keys without crashing—allows mock mode to work
        even when no real keys are configured.
        """
        log = logging.getLogger("config")
        missing: list[str] = []

        required_groups = {
            "LiveKit": ["LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"],
            "Groq": ["GROQ_API_KEY"],
            "Supabase": ["SUPABASE_URL", "SUPABASE_ANON_KEY"],
        }

        for group, keys in required_groups.items():
            for key in keys:
                if not getattr(self, key):
                    missing.append(key)

        if missing:
            if self.MOCK_APIS:
                log.warning(
                    "MOCK_APIS=true — running without real credentials. "
                    "Missing keys: %s",
                    ", ".join(missing),
                )
            else:
                log.error(
                    "Missing required environment variables: %s. "
                    "Copy .env.example to .env and fill in your keys, "
                    "or set MOCK_APIS=true to run in demo mode.",
                    ", ".join(missing),
                )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings singleton."""
    settings = Settings()
    settings.validate()
    return settings
