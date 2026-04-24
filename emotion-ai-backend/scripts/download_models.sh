#!/usr/bin/env bash
# ============================================================
# scripts/download_models.sh
# Download required ONNX model files for offline CPU inference
# ============================================================
#
# Usage:
#   chmod +x scripts/download_models.sh
#   ./scripts/download_models.sh
#
# What this downloads:
#   • SenseVoice-Small ONNX model (~150MB) from HuggingFace
#   • Silero VAD ONNX model (~2MB) from HuggingFace
#
# All models are saved to: ./models/
# ============================================================

set -euo pipefail

MODELS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/models"
mkdir -p "$MODELS_DIR"

echo "📦 Downloading models to: $MODELS_DIR"

# ── SenseVoice-Small ONNX ────────────────────────────────────────────────────
SENSEVOICE_URL="https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/model.int8.onnx"
SENSEVOICE_PATH="$MODELS_DIR/sensevoice_small.onnx"

if [ -f "$SENSEVOICE_PATH" ]; then
    echo "✅ SenseVoice-Small already exists, skipping."
else
    echo "⬇️  Downloading SenseVoice-Small ONNX (~150MB)..."
    if command -v curl &>/dev/null; then
        curl -L --progress-bar -o "$SENSEVOICE_PATH" "$SENSEVOICE_URL"
    elif command -v wget &>/dev/null; then
        wget -q --show-progress -O "$SENSEVOICE_PATH" "$SENSEVOICE_URL"
    else
        echo "❌ Neither curl nor wget found. Install one and re-run."
        exit 1
    fi
    echo "✅ SenseVoice-Small downloaded: $(du -sh "$SENSEVOICE_PATH" | cut -f1)"
fi

# ── Silero VAD ONNX ──────────────────────────────────────────────────────────
SILERO_URL="https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx"
SILERO_PATH="$MODELS_DIR/silero_vad.onnx"

if [ -f "$SILERO_PATH" ]; then
    echo "✅ Silero VAD already exists, skipping."
else
    echo "⬇️  Downloading Silero VAD ONNX (~2MB)..."
    if command -v curl &>/dev/null; then
        curl -L --progress-bar -o "$SILERO_PATH" "$SILERO_URL"
    elif command -v wget &>/dev/null; then
        wget -q --show-progress -O "$SILERO_PATH" "$SILERO_URL"
    fi
    echo "✅ Silero VAD downloaded: $(du -sh "$SILERO_PATH" | cut -f1)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All models ready in $MODELS_DIR"
echo "   Next step → copy .env.example to .env and add your API keys"
echo "   Then run: make dev   (or: make mock)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
