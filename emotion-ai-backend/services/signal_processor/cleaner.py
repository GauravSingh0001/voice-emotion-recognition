"""
services/signal_processor/cleaner.py
──────────────────────────────────────
Module 2A — Audio Cleaner

Responsibilities:
  1. Receive raw 48kHz stereo/mono PCM from the LiveKit gateway.
  2. Resample to 16kHz mono (required by Silero VAD and SenseVoice).
  3. Apply spectral gating noise reduction via `noisereduce`.
  4. Return clean 16kHz float32 numpy array ready for the VAD engine.
"""

from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger("signal_processor.cleaner")

# Lazy imports to avoid loading heavy libs at module level
_librosa = None
_noisereduce = None


def _get_librosa():
    global _librosa
    if _librosa is None:
        import librosa as _lib
        _librosa = _lib
    return _librosa


def _get_noisereduce():
    global _noisereduce
    if _noisereduce is None:
        import noisereduce as _nr
        _noisereduce = _nr
    return _noisereduce


TARGET_SAMPLE_RATE = 16_000  # Hz


def pcm_bytes_to_float32(pcm_bytes: bytes, channels: int = 1) -> np.ndarray:
    """
    Convert raw 16-bit little-endian PCM bytes to a float32 numpy array
    in the range [-1.0, 1.0].  If input is stereo, mix down to mono.
    """
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        # Reshape to (frames, channels) and average channels
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples


def resample(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """
    Resample audio from `orig_sr` to TARGET_SAMPLE_RATE using librosa.
    Returns float32 array at 16kHz.
    """
    if orig_sr == TARGET_SAMPLE_RATE:
        return audio
    librosa = _get_librosa()
    resampled: np.ndarray = librosa.resample(
        audio, orig_sr=orig_sr, target_sr=TARGET_SAMPLE_RATE
    )
    logger.debug("Resampled %d Hz → %d Hz (%d frames)", orig_sr, TARGET_SAMPLE_RATE, len(resampled))
    return resampled


def reduce_noise(audio: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """
    Apply spectral gating noise reduction.
    Uses the first 0.5 seconds as a noise profile estimate if available.
    """
    nr = _get_noisereduce()
    try:
        # Use first 0.5s as stationary noise estimate
        noise_sample_frames = int(sample_rate * 0.5)
        if len(audio) > noise_sample_frames:
            noise_clip = audio[:noise_sample_frames]
        else:
            noise_clip = audio

        cleaned: np.ndarray = nr.reduce_noise(
            y=audio,
            sr=sample_rate,
            y_noise=noise_clip,
            stationary=True,
            prop_decrease=0.75,
        )
        return cleaned
    except Exception as exc:
        logger.warning("Noise reduction failed (%s), using raw audio.", exc)
        return audio


def clean_audio_chunk(
    pcm_bytes: bytes,
    orig_sr: int,
    channels: int,
    apply_nr: bool = True,
) -> np.ndarray:
    """
    Full pipeline: PCM bytes → float32 mono → resample → (optional) denoise.
    Returns float32 numpy array at 16kHz.
    """
    audio = pcm_bytes_to_float32(pcm_bytes, channels=channels)
    audio = resample(audio, orig_sr=orig_sr)
    if apply_nr and len(audio) > TARGET_SAMPLE_RATE * 0.1:  # Only if > 100ms
        audio = reduce_noise(audio, sample_rate=TARGET_SAMPLE_RATE)
    return audio
