"""
services/signal_processor/vad_engine.py
─────────────────────────────────────────
Module 2C — VAD Engine (Silero VAD State Machine)

This module runs as a dedicated Python process (via multiprocessing.Process)
to avoid GIL interference with audio I/O. It consumes raw AudioChunk objects
from the audio_queue (filled by Module 1) and emits three event types to
the event_queue (consumed by Module 3):

  • VADStateChange    – whenever VAD state transitions
  • FastPathTrigger   – every 500ms during speech
  • UtteranceComplete – at end of utterance (after 600ms hangover)

State Machine:
  SILENCE    ─→ SPEECH_START (energy crosses threshold)
  SPEECH_START─→ SPEAKING    (confirmed after 200ms hold)
  SPEECH_START─→ SILENCE     (false positive, dropped)
  SPEAKING   ─→ SPEECH_END   (speech stops)
  SPEECH_END ─→ SILENCE      (after 600ms hangover)
  SPEECH_END ─→ SPEAKING     (speech resumes)
"""

from __future__ import annotations

import logging
import time
from multiprocessing import Queue
from typing import List

import numpy as np

from shared.schemas import (
    AudioChunk,
    Emotion,
    FastPathTrigger,
    UtteranceComplete,
    VADState,
    VADStateChange,
)
from services.signal_processor.cleaner import clean_audio_chunk, TARGET_SAMPLE_RATE
from services.signal_processor.embedder import (
    run_local_inference,
    get_top_emotion,
    mock_inference,
)

logger = logging.getLogger("signal_processor.vad_engine")

# ── Timing Constants ──────────────────────────────────────────────────────────
SPEECH_CONFIRM_MS = 200       # Hold time before SPEECH_START → SPEAKING
HANGOVER_MS = 600             # Silence after speech before declaring end
FAST_PATH_WINDOW_MS = 500     # Emit FastPathTrigger every 500ms of speech
SILERO_CHUNK_FRAMES = 512     # Silero VAD requires exactly 512 samples @ 16kHz

# ── Silero VAD Threshold ──────────────────────────────────────────────────────
VAD_THRESHOLD = 0.5

# ── Lazy Silero Model ─────────────────────────────────────────────────────────
_silero_model = None
_silero_utils = None


def _get_silero():
    global _silero_model, _silero_utils
    if _silero_model is None:
        try:
            import torch  # type: ignore

            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=True,
                trust_repo=True,
            )
            _silero_model = model
            _silero_utils = utils
            logger.info("Silero VAD model loaded.")
        except Exception as exc:
            logger.warning("Silero VAD load failed: %s. Using energy-based VAD.", exc)
    return _silero_model, _silero_utils


def _silero_speech_prob(chunk_16k: np.ndarray) -> float:
    """Return speech probability [0, 1] for a 512-frame chunk."""
    model, _ = _get_silero()
    if model is None:
        # Energy-based fallback
        rms = float(np.sqrt(np.mean(chunk_16k ** 2)))
        return min(rms * 10.0, 1.0)

    try:
        import torch  # type: ignore

        tensor = torch.from_numpy(chunk_16k).float()
        # Silero expects shape (1, N)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        prob = float(model(tensor, TARGET_SAMPLE_RATE).item())
        return prob
    except Exception as exc:
        logger.debug("Silero inference error: %s", exc)
        rms = float(np.sqrt(np.mean(chunk_16k ** 2)))
        return min(rms * 10.0, 1.0)


# ── VAD State Machine ─────────────────────────────────────────────────────────

class VADStateMachine:
    """
    Stateful VAD processor for a single session.
    Processes 16kHz audio frames and emits events.
    """

    def __init__(self, session_id: str, event_queue: Queue, mock: bool = False):
        self.session_id = session_id
        self.event_queue = event_queue
        self.mock = mock

        self.state = VADState.SILENCE
        self._speech_start_time_ms: float = 0.0
        self._last_speech_ms: float = 0.0
        self._last_fast_path_ms: float = 0.0
        self._utterance_buffer: List[np.ndarray] = []
        self._fast_path_history: List[FastPathTrigger] = []
        self._window_index: int = 0

        # Rolling buffer for the current 500ms fast-path window
        self._window_buffer: List[np.ndarray] = []
        self._window_frames: int = 0
        self._target_window_frames = int(TARGET_SAMPLE_RATE * FAST_PATH_WINDOW_MS / 1000)

    def _emit(self, event) -> None:
        try:
            self.event_queue.put_nowait(event)
        except Exception:
            logger.warning("Event queue full, dropping event: %s", type(event).__name__)

    def _transition(self, new_state: VADState) -> None:
        if new_state == self.state:
            return
        change = VADStateChange(
            session_id=self.session_id,
            previous_state=self.state,
            new_state=new_state,
        )
        logger.info("VAD: %s → %s", self.state.value, new_state.value)
        self.state = new_state
        self._emit(change)

    def _run_fast_path(self) -> None:
        """Concatenate window buffer, run inference, emit FastPathTrigger."""
        if not self._window_buffer:
            return

        audio = np.concatenate(self._window_buffer)
        self._window_buffer.clear()
        self._window_frames = 0

        if self.mock:
            probs = mock_inference(len(audio))
        else:
            probs = run_local_inference(audio, TARGET_SAMPLE_RATE)
            if probs is None:
                probs = mock_inference(len(audio))  # Fallback within module

        top_emotion, confidence = get_top_emotion(probs)
        trigger = FastPathTrigger(
            session_id=self.session_id,
            window_index=self._window_index,
            emotion_logits=probs,
            top_emotion=top_emotion,
            top_confidence=confidence,
            audio_duration_ms=FAST_PATH_WINDOW_MS,
        )
        self._window_index += 1
        self._fast_path_history.append(trigger)
        self._emit(trigger)

    def _emit_utterance_complete(self) -> None:
        """Concatenate the full utterance buffer and emit UtteranceComplete."""
        if not self._utterance_buffer:
            return

        full_audio = np.concatenate(self._utterance_buffer)
        duration_ms = len(full_audio) / TARGET_SAMPLE_RATE * 1000

        # Convert float32 → int16 PCM bytes for transport
        pcm_int16 = (full_audio * 32767).clip(-32768, 32767).astype(np.int16)
        pcm_bytes = pcm_int16.tobytes()

        event = UtteranceComplete(
            session_id=self.session_id,
            pcm_bytes=pcm_bytes,
            audio_duration_ms=duration_ms,
            window_emotions=list(self._fast_path_history),
        )
        self._emit(event)
        logger.info(
            "Utterance complete: %.0f ms, %d fast-path windows.",
            duration_ms,
            len(self._fast_path_history),
        )

        # Reset utterance state
        self._utterance_buffer.clear()
        self._fast_path_history.clear()
        self._window_index = 0

    def process_chunk(self, chunk_16k: np.ndarray, now_ms: float) -> None:
        """
        Feed a processed 16kHz float32 chunk through the VAD state machine.
        This must be called sequentially (within the signal processor process).
        """
        n = len(chunk_16k)
        self._utterance_buffer.append(chunk_16k)
        self._window_buffer.append(chunk_16k)
        self._window_frames += n

        # --- Run Silero on 512-frame sub-chunks ---
        speech_detected = False
        for start in range(0, n, SILERO_CHUNK_FRAMES):
            sub = chunk_16k[start : start + SILERO_CHUNK_FRAMES]
            if len(sub) < SILERO_CHUNK_FRAMES:
                sub = np.pad(sub, (0, SILERO_CHUNK_FRAMES - len(sub)))
            prob = _silero_speech_prob(sub)
            if prob >= VAD_THRESHOLD:
                speech_detected = True
                break

        # --- State Transitions ---
        if self.state == VADState.SILENCE:
            if speech_detected:
                self._speech_start_time_ms = now_ms
                self._transition(VADState.SPEECH_START)
            else:
                # Not speaking — do not accumulate utterance buffer
                self._utterance_buffer.clear()
                self._window_buffer.clear()
                self._window_frames = 0

        elif self.state == VADState.SPEECH_START:
            if speech_detected:
                elapsed_since_start = now_ms - self._speech_start_time_ms
                if elapsed_since_start >= SPEECH_CONFIRM_MS:
                    self._last_speech_ms = now_ms
                    self._last_fast_path_ms = now_ms
                    self._transition(VADState.SPEAKING)
            else:
                # False positive — reset
                self._utterance_buffer.clear()
                self._window_buffer.clear()
                self._window_frames = 0
                self._transition(VADState.SILENCE)

        elif self.state == VADState.SPEAKING:
            if speech_detected:
                self._last_speech_ms = now_ms
            else:
                self._transition(VADState.SPEECH_END)

            # Emit FastPathTrigger every 500ms
            if self._window_frames >= self._target_window_frames:
                self._run_fast_path()

        elif self.state == VADState.SPEECH_END:
            if speech_detected:
                # Speech resumed
                self._last_speech_ms = now_ms
                self._transition(VADState.SPEAKING)
            else:
                silence_duration = now_ms - self._last_speech_ms
                if silence_duration >= HANGOVER_MS:
                    # Flush remaining fast-path window
                    if self._window_frames > 0:
                        self._run_fast_path()
                    self._emit_utterance_complete()
                    self._transition(VADState.SILENCE)


# ── Signal Processor Process Entry Point ─────────────────────────────────────

def run_signal_processor(
    session_id: str,
    audio_queue: Queue,
    event_queue: Queue,
    mock: bool = False,
) -> None:
    """
    Main loop for the signal processor — runs in a separate Python process.
    Reads AudioChunk from audio_queue, cleans audio, and feeds VAD engine.
    """
    import sys
    from pathlib import Path

    # Ensure project root is on path inside subprocess
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logger.info("Signal processor started for session %s (mock=%s)", session_id, mock)

    vad = VADStateMachine(session_id=session_id, event_queue=event_queue, mock=mock)

    while True:
        try:
            chunk: AudioChunk = audio_queue.get(timeout=5.0)
        except Exception:
            # Timeout — check if we should emit a hangover end
            now_ms = time.time() * 1000
            if vad.state in (VADState.SPEAKING, VADState.SPEECH_END):
                vad._last_speech_ms = now_ms - HANGOVER_MS - 1
                vad._transition(VADState.SPEECH_END)
                empty = np.zeros(0, dtype=np.float32)
                vad.process_chunk(empty, now_ms)
            continue

        try:
            audio_16k = clean_audio_chunk(
                pcm_bytes=chunk.pcm_bytes,
                orig_sr=chunk.sample_rate,
                channels=chunk.channels,
                apply_nr=not mock,
            )
            now_ms = chunk.timestamp_ms
            vad.process_chunk(audio_16k, now_ms)
        except Exception as exc:
            logger.error("Error processing audio chunk: %s", exc, exc_info=True)
