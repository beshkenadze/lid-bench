#!/usr/bin/env python3
"""
Convert speechbrain/lang-id-voxlingua107-ecapa to CoreML .mlpackage

Path: PyTorch → torch.jit.trace → CoreML
Input: log-mel spectrogram [1, T, 60] float32
Output: log_probs [1, 107]

The Fbank (STFT) is NOT exported — must be computed on-device (e.g. vDSP in Swift).
SpeechBrain uses 60 mel bins with symmetric triangular filterbank (differs from HTK).
CMVN stats are all zeros (identity transform), so normalization is skipped.

Mel spectrogram parameters (SpeechBrain defaults for this model):
  n_fft=400, hop_length=160, win_length=400, n_mels=60
  window=hamming (periodic), center=True, pad_mode=constant
  log: 10 * log10(clamp(x, min=1e-10)), top_db=80 (per-sequence)
"""

import json
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_ID = "speechbrain/lang-id-voxlingua107-ecapa"
COREML_PATH = Path(__file__).parent.parent / "models" / "EcapaTdnnLid107.mlpackage"
MODELS_DIR = COREML_PATH.parent
N_MELS = 60


class LangIDWrapper(nn.Module):
    """Wraps ECAPA-TDNN + classifier, taking mel-spectrogram input."""

    def __init__(self, embedding_model, classifier_model):
        super().__init__()
        self.embedding_model = embedding_model
        self.classifier = classifier_model

    def forward(self, mel_feats: torch.Tensor) -> torch.Tensor:
        # mel_feats: [B, T, 60]
        # Pass lengths=None to avoid int() tracing errors
        embeddings = self.embedding_model(mel_feats, None)  # [B, 1, 192]
        out_prob = self.classifier(embeddings).squeeze(1)  # [B, 107]
        return out_prob


def main():
    from speechbrain.inference.classifiers import EncoderClassifier

    # Step 1: Load model
    print(f"Loading {MODEL_ID}...")
    classifier = EncoderClassifier.from_hparams(
        source=MODEL_ID,
        savedir=str(MODELS_DIR / "speechbrain_cache"),
    )
    classifier.eval()

    # Step 2: Extract label map
    label_encoder = classifier.hparams.label_encoder
    num_labels = len(label_encoder)
    id2label = {}
    for i in range(num_labels):
        lang = label_encoder.decode_ndim(torch.tensor(i))
        id2label[str(i)] = lang
    print(f"  Labels: {num_labels}")

    labels_path = MODELS_DIR / "ecapa_tdnn_lid107_labels.json"
    with open(labels_path, "w") as f:
        json.dump(id2label, f, indent=2)
    print(f"  Labels saved to {labels_path}")

    # Step 3: Verify CMVN stats are identity (all zeros)
    norm = classifier.mods.mean_var_norm
    cmvn_mean = norm.glob_mean.cpu().numpy().flatten()
    cmvn_std = norm.glob_std.cpu().numpy().flatten()
    assert cmvn_mean.sum() == 0 and cmvn_std.sum() == 0, (
        "CMVN stats are non-zero — normalization needed"
    )
    print("  CMVN stats: all zeros (identity transform, skipped)")

    # Verify n_mels from the model's Fbank
    fbank = classifier.mods.compute_features
    actual_n_mels = fbank.compute_fbanks.n_mels
    print(f"  n_mels from model: {actual_n_mels}")
    assert actual_n_mels == N_MELS, f"Expected {N_MELS} mel bins, got {actual_n_mels}"

    # Step 4: Create wrapper and trace
    print("Tracing model...")
    wrapper = LangIDWrapper(
        classifier.mods.embedding_model,
        classifier.mods.classifier,
    )
    wrapper.eval()

    T_frames = 300  # ~3 seconds of audio at 10ms hop
    example_input = torch.randn(1, T_frames, N_MELS)

    with torch.no_grad():
        test_output = wrapper(example_input)
        print(f"  Test output shape: {test_output.shape}")  # [1, 107]
        assert test_output.shape == (1, num_labels)

        traced = torch.jit.trace(wrapper, example_input)

    # Step 5: Convert to CoreML
    print("Converting to CoreML...")
    import coremltools as ct

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="mel_features",
                shape=ct.Shape(
                    shape=(
                        1,
                        ct.RangeDim(lower_bound=10, upper_bound=3000, default=300),
                        N_MELS,
                    )
                ),
                dtype=np.float32,
            )
        ],
        minimum_deployment_target=ct.target.macOS14,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT32,
    )

    mlmodel.short_description = (
        f"VoxLingua107 ECAPA-TDNN Language ID ({num_labels} languages)"
    )
    mlmodel.user_defined_metadata["source_model"] = MODEL_ID
    mlmodel.user_defined_metadata["id2label"] = json.dumps(id2label)
    mlmodel.user_defined_metadata["sampling_rate"] = "16000"
    mlmodel.user_defined_metadata["n_mels"] = str(N_MELS)
    mlmodel.user_defined_metadata["n_fft"] = "400"
    mlmodel.user_defined_metadata["hop_length"] = "160"
    mlmodel.user_defined_metadata["win_length"] = "400"

    COREML_PATH.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(COREML_PATH))
    print(f"  CoreML saved: {COREML_PATH}")

    # Step 6: Verify
    spec = mlmodel.get_spec()
    print(f"  Inputs: {[i.name for i in spec.description.input]}")
    print(f"  Outputs: {[o.name for o in spec.description.output]}")

    print(f"\n✅ ECAPA-TDNN conversion complete!")


if __name__ == "__main__":
    main()
