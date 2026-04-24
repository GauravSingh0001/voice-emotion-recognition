import asyncio
import sys
import os
import wave
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.config import get_settings
from shared.schemas import AudioChunk, UtteranceComplete, FastPathTrigger
from services.signal_processor.embedder import run_local_inference, get_top_emotion
from services.orchestrator.groq_client import transcribe_utterance, judge_emotion

async def test_wav(filepath: str):
    cfg = get_settings()
    
    print(f"=== Audio Analysis Setup ===")
    print(f"MOCK_APIS = {cfg.MOCK_APIS}")
    print(f"File = {filepath}\n")

    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    # 1. Read WAV file (assume 16kHz or 48kHz, we will convert)
    import librosa
    import numpy as np
    
    print(f"Loading {os.path.basename(filepath)} with Librosa...")
    y, sr = librosa.load(filepath, sr=16000)
    
    print(f"Audio loaded: {len(y)/16000:.2f} seconds.")
    
    # 2. Fast Path - Local Context (SenseVoice)
    print("\n=== Fast Path Inference (SenseVoice ONNX) ===")
    # Split into 500ms chunks to simulate live windows
    chunk_frames = 16000 // 2
    fast_path_history = []
    
    for i in range(0, len(y), chunk_frames):
        chunk = y[i:i+chunk_frames]
        if len(chunk) < chunk_frames:
            # Pad the last chunk
            chunk = np.pad(chunk, (0, chunk_frames - len(chunk)))

        probs = run_local_inference(chunk, sample_rate=16000)
        if probs:
            top_em, conf = get_top_emotion(probs)
            fp = FastPathTrigger(
                session_id="test",
                window_index=len(fast_path_history),
                top_emotion=top_em,
                top_confidence=conf,
                emotion_logits=probs,
                audio_duration_ms=500
            )
            fast_path_history.append(fp)
            print(f"Window {len(fast_path_history)}: {top_em} ({conf:.2f})")
        else:
            print(f"Window {len(fast_path_history)+1}: Inference failed (Did you run .\\run.ps1 models ?)")

    if cfg.MOCK_APIS:
        print("\n[!] Skipping Slow Path because system is in MOCK mode.")
        print("Set MOCK_APIS=False in your .env and add API keys if you want to test Groq.")
        return

    # 3. Slow Path - Groq Transcription + Llama 3
    print("\n=== Slow Path Inference (Groq Whisper + Llama 3) ===")
    # Convert back to 16-bit PCM for Groq
    pcm_int16 = (y * 32767).clip(-32768, 32767).astype(np.int16)
    pcm_bytes = pcm_int16.tobytes()

    utterance = UtteranceComplete(
        session_id="test-session",
        audio_duration_ms=(len(y) / 16000) * 1000,
        pcm_bytes=pcm_bytes,
        window_emotions=fast_path_history
    )

    t0 = time.time()
    print("Transcribing with Groq Whisper...")
    transcript = await transcribe_utterance(utterance)
    print(f"Transcript ({time.time()-t0:.2f}s): '{transcript.text}'")

    print("\nJudging final emotion with Groq Llama 3...")
    t0 = time.time()
    verdict = await judge_emotion("test-session", transcript.text, fast_path_history)
    print(f"Final Emotion ({time.time()-t0:.2f}s): {verdict.final_emotion}")
    print(f"Confidence: {verdict.confidence:.2f}")
    print(f"Reasoning: {verdict.reasoning}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        asyncio.run(test_wav(sys.argv[1]))
    else:
        print("Usage: python test_wav.py <path_to_wav>")
