#!/usr/bin/env python3
"""
Convert SpeechBrain ECAPA-TDNN (lang-id-voxlingua107-ecapa) weights to MLX safetensors.

SpeechBrain ECAPA-TDNN architecture (from the checkpoint):
  embedding_model (ECAPA_TDNN):
    blocks.0: TDNNBlock (Conv1d 60→512, k=5)
    blocks.1: SERes2NetBlock (512, k=3, dilation=2)
    blocks.2: SERes2NetBlock (512, k=3, dilation=3)
    blocks.3: SERes2NetBlock (512, k=3, dilation=4)
    mfa: TDNNBlock (Conv1d 1536→1536, k=1)  -- channel fusion
    asp: AttentiveStatisticsPooling (1536→1536)
    asp_bn: BatchNorm1d(3072)
    fc: Conv1d(3072→192, k=1)  -- final embedding projection
  classifier: Linear(192→107) wrapped in Classifier module

PyTorch Conv1d weight shape: (C_out, C_in/groups, K)
MLX Conv1d weight shape:     (C_out, K, C_in/groups)
→ Transpose axes 1 and 2: weight.swapaxes(1, 2)
"""

import json
import sys
from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).parent.parent / "models"
OUTPUT_DIR = Path(__file__).parent / "weights"


def convert():
    try:
        import torch
    except ImportError:
        print("ERROR: torch required for conversion. Run:")
        print("  uv pip install torch --python .venv/bin/python")
        sys.exit(1)

    try:
        from safetensors.numpy import save_file
    except ImportError:
        print("ERROR: safetensors required. Run:")
        print("  uv pip install safetensors --python .venv/bin/python")
        sys.exit(1)

    # Load SpeechBrain checkpoint from HF cache
    hf_cache = Path.home() / ".cache/huggingface/hub/models--speechbrain--lang-id-voxlingua107-ecapa"
    snapshot_dir = None
    if hf_cache.exists():
        snapshots = list((hf_cache / "snapshots").iterdir())
        if snapshots:
            snapshot_dir = snapshots[0]

    if snapshot_dir is None:
        print("ERROR: SpeechBrain model not in HF cache.")
        print("Run: huggingface-cli download speechbrain/lang-id-voxlingua107-ecapa")
        sys.exit(1)

    ckpt_path = snapshot_dir / "embedding_model.ckpt"
    classifier_ckpt_path = snapshot_dir / "classifier.ckpt"

    print(f"Loading embedding model from {ckpt_path}")
    emb_state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    print(f"  {len(emb_state)} parameters")

    print(f"Loading classifier from {classifier_ckpt_path}")
    cls_state = torch.load(classifier_ckpt_path, map_location="cpu", weights_only=True)
    print(f"  {len(cls_state)} parameters")

    state_dict = {}
    for k, v in emb_state.items():
        state_dict[f"embedding_model.{k}"] = v
    for k, v in cls_state.items():
        state_dict[f"classifier.{k}"] = v

    # Print architecture
    print("\nSpeechBrain parameter map:")
    for k, v in sorted(state_dict.items()):
        print(f"  {k}: {list(v.shape)}")

    # Convert to MLX-compatible naming and shapes
    # SpeechBrain ECAPA-TDNN → our flat MLX ECAPA-TDNN
    mlx_weights = {}
    skipped = []

    for key, tensor in state_dict.items():
        arr = tensor.detach().cpu().numpy()

        # Conv1d weights: PyTorch (C_out, C_in/g, K) → MLX (C_out, K, C_in/g)
        if "conv.weight" in key or "tdnn.weight" in key or ".weight" in key:
            if arr.ndim == 3:
                arr = np.swapaxes(arr, 1, 2)

        # BatchNorm running stats
        if "num_batches_tracked" in key:
            skipped.append(key)
            continue

        mlx_weights[key] = arr

    print(f"\nConverted {len(mlx_weights)} tensors, skipped {len(skipped)}")
    for s in skipped:
        print(f"  skipped: {s}")

    # Save as safetensors
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "ecapa_tdnn_lid107.safetensors"
    save_file(mlx_weights, str(output_path))
    print(f"\nSaved: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Also save the key mapping for debugging
    mapping_path = OUTPUT_DIR / "ecapa_tdnn_key_mapping.json"
    mapping = {k: list(v.shape) for k, v in mlx_weights.items()}
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"Key mapping: {mapping_path}")


if __name__ == "__main__":
    convert()
