# ════════════════════════════════════════════════════════
# Dockerfile — Real-Time Emotion Intelligence Backend
# ════════════════════════════════════════════════════════
# Base: python:3.11-slim for a small, production-ready image.
# System deps: ffmpeg (audio codec), libsndfile1 (soundfile),
#              libportaudio2 (PortAudio for PyAudio fallback).
#
# Build:   docker build -t emotion-ai-backend .
# Run:     docker run -p 8000:8000 --env-file .env emotion-ai-backend
# ════════════════════════════════════════════════════════

FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        libportaudio2 \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies (cached layer) ────────────────────────────────
COPY requirements.txt .
# Install CPU-only PyTorch first to keep image size manageable
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application source ───────────────────────────────────────────────────
COPY . .

# ── ONNX model directory (populated at runtime via make models or volume mount) ─
RUN mkdir -p /app/models /app/recordings

# ── Runtime configuration ─────────────────────────────────────────────────────
# All secrets must be supplied via environment variables or --env-file.
# Do NOT bake real keys into the image.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

EXPOSE 8000

# ── Health check for deployment platform ─────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Entry point ───────────────────────────────────────────────────────────────
CMD ["uvicorn", "services.orchestrator.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]
