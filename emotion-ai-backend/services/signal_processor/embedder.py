"""
services/signal_processor/embedder.py
────────────────────────────────────────
Module 2B / Module 4 — SenseVoice-Small ONNX Embedder (sherpa-onnx format)

Model: csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17 (int8)
Inputs:
  • x          — (N, T, 560) float32  LFR log-mel features + CMVN normalised
  • x_length   — (N,)        int32    number of LFR frames per sample
  • language   — (N,)        int32    language id  (4 = English / auto = 0)
  • text_norm  — (N,)        int32    14=with-itn  15=without-itn
Output:
  • logits     — (N, T, 25055) float32  CTC frame logits

Emotion special tokens (from tokens.txt):
  <|HAPPY|>     25001
  <|SAD|>       25002
  <|ANGRY|>     25003
  <|NEUTRAL|>   25004
  <|FEARFUL|>   25005
  <|DISGUSTED|> 25006
  <|SURPRISED|> 25007
  <|EMO_UNKNOWN|> 25009
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from shared.schemas import Emotion

logger = logging.getLogger("signal_processor.embedder")

# ── Paths ─────────────────────────────────────────────────────────────────────
_MODELS_DIR       = Path(__file__).resolve().parent.parent.parent / "models"
_ONNX_MODEL_PATH  = _MODELS_DIR / "sensevoice_small.onnx"
_TOKENS_PATH      = _MODELS_DIR / "sensevoice_tokens.txt"

# ── Emotion token IDs ────────────────────────────────────────────────────────
# Mapped directly from tokens.txt
_EMO_TOKEN_IDS = {
    25001: Emotion.HAPPY,
    25002: Emotion.SAD,
    25003: Emotion.ANGRY,
    25004: Emotion.NEUTRAL,
    25005: Emotion.SURPRISED,   # FEARFUL → map to Surprised (closest in schema)
    25006: Emotion.NEUTRAL,     # DISGUSTED → map to Neutral (not in schema)
    25007: Emotion.SURPRISED,   # SURPRISED
    25009: Emotion.NEUTRAL,     # EMO_UNKNOWN
}

# Map from Emotion enum value → 5-class index [Neutral, Happy, Sad, Angry, Surprised]
EMOTION_LABELS: List[str] = [
    Emotion.NEUTRAL,
    Emotion.HAPPY,
    Emotion.SAD,
    Emotion.ANGRY,
    Emotion.SURPRISED,
]

_EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTION_LABELS)}

# ── Model constants (from ONNX metadata) ────────────────────────────────────
_LFR_M   = 7    # LFR window size
_LFR_N   = 6    # LFR window shift
_LANG_EN = 4    # language = English
_TEXT_NORM_WITHOUT_ITN = 15  # no inverse-text-normalisation (slightly faster)

# CMVN stats — loaded lazily from model metadata
_cmvn_neg_mean: Optional[np.ndarray] = None
_cmvn_inv_std:  Optional[np.ndarray] = None

# Runtime state
_ort_session  = None
_model_ready  = False


# ── CMVN helpers ─────────────────────────────────────────────────────────────

def _load_cmvn_from_meta(meta: dict) -> None:
    global _cmvn_neg_mean, _cmvn_inv_std
    try:
        neg_mean = np.fromstring(meta["neg_mean"],  sep=",", dtype=np.float32)
        inv_std  = np.fromstring(meta["inv_stddev"], sep=",", dtype=np.float32)
        # Each repeats 7 times (LFR stacks 7 mel frames → 80*7 = 560 dims)
        # The metadata encodes all 560 dims already
        _cmvn_neg_mean = neg_mean
        _cmvn_inv_std  = inv_std
        logger.debug("CMVN loaded: neg_mean shape=%s, inv_std shape=%s",
                     neg_mean.shape, inv_std.shape)
    except Exception as exc:
        logger.warning("Failed to parse CMVN from model metadata: %s", exc)


# ── Model loader ─────────────────────────────────────────────────────────────

def _load_onnx_model() -> bool:
    global _ort_session, _model_ready

    if _model_ready:
        return True

    if not _ONNX_MODEL_PATH.exists():
        logger.warning(
            "ONNX model not found at %s. Run '.\\run.ps1 models' to fetch it. "
            "Falling back to HF API.",
            _ONNX_MODEL_PATH,
        )
        return False

    try:
        import onnxruntime as ort  # type: ignore

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        _ort_session = ort.InferenceSession(
            str(_ONNX_MODEL_PATH),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        # Load CMVN from embedded metadata
        meta = _ort_session.get_modelmeta().custom_metadata_map
        _load_cmvn_from_meta(meta)

        _model_ready = True
        logger.info("SenseVoice-Small ONNX loaded from %s", _ONNX_MODEL_PATH)
        return True
    except Exception as exc:
        logger.error("Failed to load ONNX model: %s", exc)
        return False


# ── Audio pre-processing ─────────────────────────────────────────────────────

def _extract_fbank(audio: np.ndarray, sample_rate: int = 16_000) -> np.ndarray:
    """
    Compute 80-dim log-mel filterbank features at 10ms frame shift.
    Returns shape (T, 80) float32.
    """
    try:
        import librosa  # type: ignore

        # NOTE: model expects input in range [-32768, 32767] (normalize_samples=0)
        # librosa normalises to [-1,1] by default, so scale up
        audio_int16_scale = audio * 32768.0

        mel = librosa.feature.melspectrogram(
            y=audio_int16_scale,
            sr=sample_rate,
            n_mels=80,
            fmax=8000,
            hop_length=160,   # 10ms at 16kHz
            win_length=400,   # 25ms window
            n_fft=512,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
        # (80, T) → (T, 80)
        return log_mel.T
    except Exception as exc:
        logger.error("fbank extraction failed: %s", exc)
        raise


def _apply_lfr(feats: np.ndarray, lfr_m: int = _LFR_M, lfr_n: int = _LFR_N) -> np.ndarray:
    """
    Low Frame Rate (LFR) stacking: stack lfr_m consecutive frames with stride lfr_n.
    Input shape:  (T, 80)
    Output shape: (T', 80*lfr_m) = (T', 560)
    """
    T, D = feats.shape
    T_lfr = (T - lfr_m) // lfr_n + 1
    if T_lfr <= 0:
        # Audio too short — pad to at least 1 frame
        pad = np.zeros((lfr_m - T, D), dtype=np.float32)
        feats = np.concatenate([feats, pad], axis=0)
        T_lfr = 1

    lfr_feats = np.zeros((T_lfr, D * lfr_m), dtype=np.float32)
    for i in range(T_lfr):
        start = i * lfr_n
        end   = min(start + lfr_m, T)
        chunk = feats[start:end]
        if len(chunk) < lfr_m:
            # Repeat last frame for short tail
            chunk = np.concatenate([chunk, np.tile(chunk[-1:], (lfr_m - len(chunk), 1))])
        lfr_feats[i] = chunk.flatten()
    return lfr_feats


def _apply_cmvn(feats: np.ndarray) -> np.ndarray:
    """Apply Cepstral Mean-Variance Normalisation using metadata stats."""
    if _cmvn_neg_mean is None or _cmvn_inv_std is None:
        return feats
    # Trim/pad stats to match feature width (should be 560)
    D = feats.shape[1]
    neg_mean = _cmvn_neg_mean[:D]
    inv_std  = _cmvn_inv_std[:D]
    return (feats + neg_mean) * inv_std


def _preprocess_audio(audio: np.ndarray, sample_rate: int = 16_000) -> Tuple[np.ndarray, int]:
    """
    Full preprocessing pipeline:  audio → fbank → LFR → CMVN
    Returns:
        features: (1, T', 560) float32
        length:   T'  (int)
    """
    fbank = _extract_fbank(audio, sample_rate)  # (T, 80)
    lfr   = _apply_lfr(fbank)                  # (T', 560)
    cmvn  = _apply_cmvn(lfr)                   # (T', 560)
    features = cmvn[np.newaxis, :, :]           # (1, T', 560)
    return features, cmvn.shape[0]


# ── CTC greedy decode → emotion ───────────────────────────────────────────────

def _ctc_emotion_probs(logits: np.ndarray) -> List[float]:
    """
    CTC-greedy-decode the logits and sum log-probs for each emotion token.
    Returns 5-class probability list [Neutral, Happy, Sad, Angry, Surprised].

    logits: (1, T', 25055) float32
    """
    logits = logits[0]  # (T', 25055)

    # Softmax over vocab dim
    logits_shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_l = np.exp(logits_shifted)
    probs_frame = exp_l / exp_l.sum(axis=-1, keepdims=True)  # (T', 25055)

    # Accumulate probability mass for each emotion token over all frames
    emo_scores = {emo_id: 0.0 for emo_id in _EMO_TOKEN_IDS}
    for frame_probs in probs_frame:
        for emo_id in _EMO_TOKEN_IDS:
            if emo_id < len(frame_probs):
                emo_scores[emo_id] += float(frame_probs[emo_id])

    # Pool into 5-class output
    class_scores = [0.0] * 5
    for emo_id, emotion in _EMO_TOKEN_IDS.items():
        idx = _EMOTION_TO_IDX.get(emotion, 0)
        class_scores[idx] += emo_scores[emo_id]

    total = sum(class_scores) or 1.0
    probs = [s / total for s in class_scores]

    # Smooth towards Neutral if all scores are near-zero (silence / noise)
    if max(probs) < 0.25:
        probs[0] = max(probs[0], 0.5)
        total2 = sum(probs)
        probs = [p / total2 for p in probs]

    return probs


# ── Public inference interface ────────────────────────────────────────────────

def run_local_inference(audio: np.ndarray, sample_rate: int = 16_000) -> Optional[List[float]]:
    """
    Run SenseVoice-Small ONNX inference on 16kHz float32 audio.
    Returns list of 5 floats (softmax probabilities) over:
        [Neutral, Happy, Sad, Angry, Surprised]
    or None on failure.
    """
    if not _load_onnx_model():
        return None

    try:
        t0 = time.perf_counter()

        features, length = _preprocess_audio(audio, sample_rate)

        x_length  = np.array([length],   dtype=np.int32)
        language  = np.array([_LANG_EN], dtype=np.int32)
        text_norm = np.array([_TEXT_NORM_WITHOUT_ITN], dtype=np.int32)

        outputs = _ort_session.run(  # type: ignore
            None,
            {
                "x":         features,
                "x_length":  x_length,
                "language":  language,
                "text_norm": text_norm,
            },
        )

        logits = outputs[0]  # (1, T', 25055)
        probs  = _ctc_emotion_probs(logits)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        top_idx    = int(np.argmax(probs))
        logger.debug(
            "ONNX inference: %.1f ms | top=%s (%.2f)",
            elapsed_ms, EMOTION_LABELS[top_idx], probs[top_idx],
        )
        return probs

    except Exception as exc:
        logger.error("ONNX inference failed: %s", exc)
        return None


async def run_hf_fallback(audio: np.ndarray, sample_rate: int = 16_000) -> Optional[List[float]]:
    """
    Call Hugging Face Inference API as a cold-start-aware fallback.
    Timeout: 500ms. Returns None if unavailable or cold-starting.
    """
    import io
    import aiohttp   # type: ignore
    import soundfile as sf  # type: ignore

    from shared.config import get_settings

    cfg = get_settings()
    if not cfg.HF_API_TOKEN:
        logger.debug("No HF_API_TOKEN configured, skipping HF fallback.")
        return None

    # Fallback model path (hardcoded since HF_EMOTION_MODEL removed from env)
    model_id = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {cfg.HF_API_TOKEN}"}

    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    wav_bytes = buf.getvalue()

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                data=wav_bytes,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=0.5),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    probs = [0.0] * 5
                    label_map = {
                        "neutral": 0, "happy": 1, "sad": 2,
                        "angry": 3, "surprised": 4, "surprise": 4,
                    }
                    for item in data:
                        lbl = item.get("label", "").lower()
                        if lbl in label_map:
                            probs[label_map[lbl]] = item.get("score", 0.0)
                    logger.info("HF fallback returned response (status 200).")
                    return probs
                elif resp.status == 503:
                    logger.warning("HF model cold-starting (503). Skipping fallback.")
                else:
                    logger.warning("HF API returned status %d.", resp.status)
    except asyncio.TimeoutError:
        logger.warning("HF fallback timed out after 500ms.")
    except Exception as exc:
        logger.warning("HF fallback error: %s", exc)

    return None


def get_top_emotion(probs: List[float]) -> tuple[Emotion, float]:
    """Return the highest-probability emotion and its confidence."""
    idx = int(np.argmax(probs))
    return Emotion(EMOTION_LABELS[idx]), float(probs[idx])


def mock_inference(audio_len_frames: int) -> List[float]:
    """Return deterministic mock probabilities for offline testing."""
    bucket = (audio_len_frames // 8000) % 5
    probs = [0.1, 0.1, 0.1, 0.1, 0.1]
    probs[bucket] = 0.6
    return probs
