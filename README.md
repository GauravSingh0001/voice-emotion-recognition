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

The repository is organized into two primary workspaces, managed by a root orchestrator:

- **`aura-frontend/`**: Vanilla JS client with Three.js visualizations and Supabase Realtime integration.
- **`emotion-ai-backend/`**: Python FastAPI backend for signal processing and local ONNX inference.
- **`run.ps1`**: Root orchestrator for unified setup, development, and maintenance.

---

## 🚀 Quick Start (Windows)

The project includes a root orchestrator script to handle environment setup and multi-process execution.

### 1. Prerequisites
- **Python 3.10+** (Added to PATH)
- **Node.js** (For frontend serving)
- **API Keys**: You will need keys for [LiveKit Cloud](https://livekit.io/), [Supabase](https://supabase.com/), and [Groq](https://groq.com/).

### 2. Initial Setup
Run the unified setup command to create the virtual environment, install dependencies, and download AI models:
```powershell
.\run.ps1 setup
```

### 3. Configuration
1. Copy `emotion-ai-backend/.env.example` to `emotion-ai-backend/.env`.
2. Fill in your credentials for LiveKit, Supabase, and Groq.
3. (Optional) Run `node aura-frontend/scripts/inject-env.js` if you change frontend-specific variables.

### 4. Run the Platform
Start both the backend and frontend in separate windows:
```powershell
.\run.ps1 dev
```

---

## 🛠 Orchestrator Commands

| Command | Description |
| :--- | :--- |
| `.\run.ps1 setup` | One-time environment creation and model download. |
| `.\run.ps1 dev` | Launches both backend and frontend for development. |
| `.\run.ps1 backend` | Starts only the FastAPI orchestrator. |
| `.\run.ps1 frontend`| Starts only the static frontend server. |
| `.\run.ps1 clean` | Removes `__pycache__`, `.pyc`, and temporary recordings. |

---

## 🧠 Core Technologies
- **Frontend**: HTML5, Vanilla JS, CSS3, Three.js, Chart.js, LiveKit Client SDK, Supabase JS SDK.
- **Backend**: Python, FastAPI, asyncio, LiveKit Server SDK, Supabase Python SDK, Groq API.
- **AI/ML Models**: Silero VAD, SenseVoice (via sherpa-onnx), Llama 3.3 70B, Whisper-Large-V3-Turbo.

