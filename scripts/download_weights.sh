#!/bin/bash
# Download model weights for LID-Bench
# Usage: ./scripts/download_weights.sh

set -euo pipefail

echo "=== LID-Bench Weight Setup ==="

# ECAPA-TDNN: already in repo
ECAPA_WEIGHTS="mlx/weights/ecapa_tdnn_lid107.safetensors"
if [ -f "$ECAPA_WEIGHTS" ]; then
    echo "✅ ECAPA-TDNN weights: $ECAPA_WEIGHTS ($(du -h "$ECAPA_WEIGHTS" | cut -f1))"
else
    echo "❌ ECAPA-TDNN weights missing: $ECAPA_WEIGHTS"
    echo "   Run: python mlx/convert_ecapa_weights.py"
fi

# MMS-LID-256: download from HuggingFace
echo ""
echo "Downloading MMS-LID-256 weights from facebook/mms-lid-256..."
if command -v hf &> /dev/null; then
    hf download facebook/mms-lid-256 \
        --include "model.safetensors" "config.json" "preprocessor_config.json"
    echo "✅ MMS-LID-256 weights downloaded to HF cache"
elif command -v huggingface-cli &> /dev/null; then
    huggingface-cli download facebook/mms-lid-256 \
        --include "model.safetensors" "config.json" "preprocessor_config.json"
    echo "✅ MMS-LID-256 weights downloaded to HF cache"
else
    echo "❌ Neither 'hf' nor 'huggingface-cli' found."
    echo "   Install: pip install huggingface_hub"
    echo "   Or: pipx install huggingface_hub"
    exit 1
fi

# CoreML models (optional)
echo ""
echo "CoreML models (optional — only needed for LIDBench target):"
echo "  hf download beshkenadze/mms-lid-256-coreml --local-dir models/MmsLid256"
echo "  hf download beshkenadze/lang-id-voxlingua107-ecapa-coreml --local-dir models/EcapaTdnn"
