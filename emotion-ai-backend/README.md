# Real-Time Emotion Intelligence — $0 Student Backend

A **cost-free, portfolio-ready** real-time emotion detection system built entirely on **free-tier cloud services** and local CPU inference. No credit card required.

---

## 🏗️ Architecture

```
Browser Mic
    │ WebRTC (Opus)
    ▼
LiveKit Cloud (Free: 50 min/month)
    │ Decoded 48kHz PCM (IPC Queue)
    ▼
Signal Processor (Separate Python Process)
  ├── Noise Reduction (noisereduce)
  ├── Resample → 16kHz mono
  ├── Silero VAD (ONNX, CPU)
  └── SenseVoice-Small (ONNX) → 5-class Fast-Path every 500ms
    │ Events (FastPathTrigger, UtteranceComplete)
    ▼
Orchestrator (FastAPI + asyncio)
  ├── Fast-Path → publish to Supabase Realtime
  └── Slow-Path (utterance end)
        ├── Groq Whisper → transcript (~200ms)
        └── Groq Llama 3.3 70B → emotion judge (~500ms)
              │
              ▼
          Supabase (Free PostgreSQL + Realtime)
              │
              ▼
         Browser UI (real-time subscription)
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone <your-repo>
cd emotion-ai-backend
pip install -r requirements.txt
```

### 2. Obtain Free API Keys

| Service | URL | Free Tier |
|---------|-----|-----------|
| **LiveKit Cloud** | https://livekit.io/cloud | 50 concurrent-minutes/month |
| **Groq Cloud** | https://console.groq.com | 14,400 requests/day |
| **Supabase** | https://app.supabase.com | 500MB DB, unlimited auth |
| **Hugging Face** | https://huggingface.co/settings/tokens | Free inference API (fallback) |

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

### 4. Set Up Supabase Database

In your Supabase project, go to **SQL Editor** and run:

```bash
make schema | pbcopy   # Copy to clipboard (macOS)
# Paste and run in Supabase SQL Editor
```

This creates the `emotion_sessions` table with proper **Row Level Security** policies and enables **Realtime**.

### 5. Download ONNX Models

```bash
make models
```

This downloads:
- `models/sensevoice_small.onnx` (~150MB) — Fast-path emotion inference
- `models/silero_vad.onnx` (~2MB) — Voice activity detection

### 6. Run the System

```bash
# With real API keys:
make run

# Development mode (hot reload):
make dev

# Fully offline (no internet / no keys needed):
make mock
```

### 7. Open the Dashboard

```bash
make dashboard
```

---

## 📁 Project Structure

```
emotion-ai-backend/
├── services/
│   ├── livekit_gateway/
│   │   ├── main.py               # Module 1: WebRTC audio receiver
│   │   └── requirements.txt
│   ├── signal_processor/
│   │   ├── cleaner.py            # Noise reduction & resampling
│   │   ├── embedder.py           # SenseVoice-Small ONNX inference
│   │   ├── vad_engine.py         # Silero VAD state machine
│   │   └── requirements.txt
│   ├── orchestrator/
│   │   ├── main.py               # Module 3: FastAPI coordination hub
│   │   ├── groq_client.py        # Module 5: Whisper + Llama wrappers
│   │   ├── supabase_client.py    # Module 6: DB + Realtime
│   │   └── requirements.txt
│   └── dashboard/
│       └── tui.py                # Module 7: Rich terminal UI
├── shared/
│   ├── schemas.py                # Pydantic inter-process contracts
│   └── config.py                 # .env loader + validation
├── scripts/
│   └── download_models.sh        # ONNX model downloader
├── models/                       # ONNX models (gitignored)
├── recordings/                   # Optional WAV archives (gitignored)
├── .env.example                  # API key template
├── requirements.txt              # All dependencies
├── Makefile                      # dev shortcuts
└── README.md
```

---

## 🔧 REST API

The Orchestrator exposes these endpoints at `http://localhost:8000`:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System status, Groq quota, active sessions |
| `POST` | `/session/start` | Start a session → returns session_id + LiveKit token |
| `POST` | `/session/end` | End session, release processes |
| `GET` | `/session/{id}/status` | Current VAD state + latest emotion |

Interactive API docs: http://localhost:8000/docs

---

## 🎭 Mock Mode

Run completely offline without any API keys:

```bash
MOCK_APIS=true make run
# Or simply:
make mock
```

In mock mode:
- Synthetic audio (440Hz sine wave) is generated locally
- Transcription returns a canned text response
- The judge returns a pre-defined `Neutral` verdict
- Supabase writes go to an in-memory list

This lets you demo the full pipeline flow without internet connectivity.

---

## 📊 Emotion Classes

The fast-path model outputs 5 classes every 500ms:

| Class | Color | Notes |
|-------|-------|-------|
| `Neutral` | Gray | Default resting state |
| `Happy` | Yellow | Elevated pitch, positive prosody |
| `Sad` | Blue | Lower energy, slower rate |
| `Angry` | Red | High intensity, harsh voice |
| `Surprised` | Magenta | Short burst, rising intonation |

---

## ⚡ Rate Limits & Free Tier Details

| Service | Limit | Handling in Code |
|---------|-------|-----------------|
| LiveKit | 50 min/month | Monthly counter in `.livekit_usage.json`, graceful offline fallback |
| Groq | 14,400 req/day | Token bucket counter, warn at 80%, retry on 429, rule-based fallback |
| Supabase | 500MB DB | Deduplication: skip write if same emotion 3× in a row |
| HuggingFace | ~50k req/day | Used only as fast-path fallback, 500ms timeout |

---

## 🪲 Troubleshooting

**`ONNX model not found`** → Run `make models`

**`Groq 429 Too Many Requests`** → You've hit the daily limit. Wait until midnight UTC or enable `MOCK_APIS=true`

**`LiveKit connection refused`** → Check `LIVEKIT_URL` in `.env`. Ensure your LiveKit project is active.

**`Supabase INSERT fails`** → Run the schema SQL via `make schema`. Check that RLS is configured.

**Signal Processor crashes** → Check that `torch` is installed (`pip install torch torchaudio`). On Windows, use WSL2.

---

## 📜 License

MIT — Free to use, modify, and share. Built for educational demo purposes.
