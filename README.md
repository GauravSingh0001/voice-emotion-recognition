# Aura · Emotion Intelligence Platform

Aura is a real-time, privacy-conscious emotion intelligence platform. It captures live audio via WebRTC, runs high-speed local inference for immediate emotional context, and utilizes cloud LLMs for deep semantic analysis of full utterances. The architecture is modular, separating the visually rich frontend client from the robust audio-processing Python backend.

---

## 🏗 System Architecture

The platform follows a split-path architecture, blending local edge inference for low-latency visual feedback with cloud APIs for complex reasoning.

```text
┌─────────────────────────────────────────────────────────────────┐
│                  Module 1: LiveKit Cloud                         │
│              WebRTC ingestion → Audio track extraction           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼ Raw PCM audio 
┌─────────────────────────────────────────────────────────────────┐
│        Module 2: Local Signal Processor (Python Multiprocess)    │
│  VAD (Silero) → Resample (16kHz) → Noise Reduction               │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼ Events + Audio Buffers
┌─────────────────────────────────────────────────────────────────┐
│            Module 3: Orchestrator (FastAPI + asyncio)            │
│      Routes Fast/Slow paths, manages backpressure, caches state  │
└─────────────────────────────────────────────────────────────────┘
                    │                               │
                    ▼ Fast-Path (500ms windows)     ▼ Slow-Path (utterance end)
┌───────────────────────────────┐   ┌─────────────────────────────────────┐
│ Module 4: Fast-Path Inference │   │     Module 5: Groq Cloud             │
│   Local ONNX SenseVoice-Small │   │ 5A: Whisper-Large-V3-Turbo (Groq)   │
│   (CPU) → 5 emotion classes   │   │ 5B: Llama 3.3 70B Judge (Groq)      │
└───────────────────────────────┘   └─────────────────────────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                Module 6: Supabase                                │
│   Auth, Real-time Database, Row Level Security, REST API         │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│         Module 7: Aura Frontend Client                           │
│     Vanilla JS + Three.js PWA with glassmorphic dashboards       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🗂 Project Structure

The repository is divided into two primary workspaces:

### 1. `aura-frontend/` (The Client)
A lightweight, high-performance Progressive Web App (PWA) built with Vanilla JavaScript, HTML5, and CSS3. 
- **Core Files:** `index.html`, `main.js`, `style.css`
- **Visualization:** `particle-aura.js` (Three.js fluid orb reacting to voice and emotion).
- **Audio Transport:** `livekit.js` (WebRTC microphone capture).
- **Dashboard:** `result.html` (Detailed emotional breakdown, transcript, and timeline charts).
- **Data Layer:** Connects directly to Supabase Realtime to receive inference verdicts instantly, bypassing the backend API.

### 2. `emotion-ai-backend/` (The Engine)
A high-throughput Python backend utilizing FastAPI, asyncio, and multiprocessing for audio signal processing.
- **`services/orchestrator/`:** The FastAPI application bridging LiveKit, inference models, and Supabase.
- **Models:** Runs `sherpa-onnx` (SenseVoice) for local, sub-second emotion classification.
- **Cloud APIs:** Integrates with Groq (Whisper + Llama 3) for deep transcription and reasoning when a full utterance (marked by silence detection) is completed.
- **Environment:** Managed via `.env` (LiveKit keys, Groq API keys, Supabase credentials).

---

## 🚀 Quick Start

### Prerequisites
- Node.js (for frontend serving and env injection)
- Python 3.10+
- Accounts/Keys for: LiveKit Cloud, Supabase, Groq.

### Backend Setup
1. Navigate to `emotion-ai-backend/`.
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your API keys.
4. Run the backend orchestrator:
   ```bash
   ./run.ps1  # Or execute the Makefile targets if on Linux/Mac
   ```

### Frontend Setup
1. Navigate to `aura-frontend/`.
2. Inject your environment variables (uses the same `.env` values or relies on system variables):
   ```bash
   node scripts/inject-env.js
   ```
3. Serve the static files:
   ```bash
   npx serve .
   # or
   python -m http.server 3000
   ```
4. Open your browser to `http://localhost:3000`.

---

## 🧠 Core Technologies
- **Frontend**: HTML5, Vanilla JS, CSS3, Three.js, Chart.js, LiveKit Client SDK, Supabase JS SDK.
- **Backend**: Python, FastAPI, asyncio, LiveKit Server SDK, Supabase Python SDK, Groq API.
- **AI/ML Models**: Silero VAD, SenseVoice (via sherpa-onnx), Llama 3.3 70B, Whisper-Large-V3-Turbo.
