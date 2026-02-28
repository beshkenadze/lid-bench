#!/usr/bin/env python3
"""
Convert facebook/mms-lid-256 to CoreML .mlpackage
Path: PyTorch → torch.jit.trace → CoreML
Input: raw waveform [1, N] at 16kHz, float32
Output: logits [1, 256]
"""

import json
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

MODEL_ID = "facebook/mms-lid-256"
COREML_PATH = Path(__file__).parent.parent / "models" / "MmsLid256.mlpackage"


def main():
    from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

    # Step 1: Load model
    print(f"Loading {MODEL_ID}...")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_ID)
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model.eval()
    model.config.return_dict = False

    num_labels = model.config.num_labels
    id2label = model.config.id2label
    print(f"  Labels: {num_labels}")
    print(f"  Sample rate: {feature_extractor.sampling_rate}")

    # Save label map for Swift
    labels_path = COREML_PATH.parent / "mms_lid_256_labels.json"
    with open(labels_path, "w") as f:
        json.dump(id2label, f, indent=2)
    print(f"  Labels saved to {labels_path}")

    # Step 2: Trace model
    print("Tracing model...")
    dummy_input = torch.randn(1, 16000)  # 1s at 16kHz

    with torch.no_grad():
        test_out = model(dummy_input)
        print(f"  PyTorch output: {len(test_out)} tensors, shape {test_out[0].shape}")
        traced = torch.jit.trace(model, dummy_input)

    # Verify trace
    with torch.no_grad():
        traced_out = traced(dummy_input)
        diff = (test_out[0] - traced_out[0]).abs().max().item()
        print(f"  Trace vs eager max diff: {diff:.6f}")
        assert diff < 0.01, f"Trace diverges from eager: {diff}"

    # Step 3: Convert to CoreML
    print("Converting to CoreML...")
    import coremltools as ct

    COREML_PATH.parent.mkdir(parents=True, exist_ok=True)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="input_values",
                shape=ct.Shape(
                    shape=(
                        1,
                        ct.RangeDim(
                            lower_bound=1600, upper_bound=480000, default=160000
                        ),
                    )
                ),
                dtype=np.float32,
            )
        ],
        minimum_deployment_target=ct.target.macOS14,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
    )

    # Get output name
    spec = mlmodel.get_spec()
    output_name = spec.description.output[0].name
    print(f"  Raw output name: {output_name}")

    mlmodel.short_description = f"MMS Language ID - {num_labels} languages ({MODEL_ID})"
    mlmodel.user_defined_metadata["id2label"] = json.dumps(id2label)
    mlmodel.user_defined_metadata["sampling_rate"] = "16000"
    mlmodel.user_defined_metadata["output_name"] = output_name

    mlmodel.save(str(COREML_PATH))
    print(f"  CoreML saved: {COREML_PATH}")

    # Step 4: Verify CoreML
    print("Verifying CoreML...")
    spec = mlmodel.get_spec()
    print(
        f"  Inputs: {[(i.name, list(i.type.multiArrayType.shape)) for i in spec.description.input]}"
    )
    print(
        f"  Outputs: {[(o.name, list(o.type.multiArrayType.shape)) for o in spec.description.output]}"
    )

    print("\n✅ MMS-LID conversion complete!")


if __name__ == "__main__":
    main()
