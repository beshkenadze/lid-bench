# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportMissingTypeArgument=false, reportUntypedFunctionDecorator=false, reportAny=false, reportUnknownParameterType=false
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
MLX_DIR = ROOT_DIR / "mlx"
if str(MLX_DIR) not in sys.path:
    sys.path.insert(0, str(MLX_DIR))

from ecapa_tdnn_lid import (  # noqa: E402
    HOP_LENGTH,
    N_FFT,
    SAMPLE_RATE,
    EcapaTdnnLid,
    compute_mel_spectrogram,
    map_key,
    speechbrain_mel_filterbank,
)
from mms_lid_256 import HF_MODEL_DIR, MmsLid256  # noqa: E402


ECAPA_WEIGHTS_PATH = ROOT_DIR / "mlx" / "weights" / "ecapa_tdnn_lid107.safetensors"
MMS_CONFIG_PATH = HF_MODEL_DIR / "config.json"
MMS_WEIGHTS_PATH = HF_MODEL_DIR / "model.safetensors"


@pytest.fixture(scope="module")
def ecapa_model() -> EcapaTdnnLid:
    return EcapaTdnnLid(n_mels=60, channels=1024, embed_dim=256, num_classes=107)


@pytest.fixture(scope="module")
def mms_config() -> dict:
    with MMS_CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def mms_model(mms_config: dict) -> MmsLid256:
    num_labels = int(mms_config.get("num_labels", 256))
    return MmsLid256(num_classes=num_labels)


def test_compute_mel_spectrogram_shape_and_db_range() -> None:
    t = np.arange(SAMPLE_RATE, dtype=np.float32) / SAMPLE_RATE
    audio = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)

    mel = compute_mel_spectrogram(audio)
    mel_np = np.array(mel)

    expected_frames = 1 + ((len(audio) + N_FFT - N_FFT) // HOP_LENGTH)
    assert mel_np.shape == (1, expected_frames, 60)

    mel_rel_db = mel_np - np.max(mel_np)
    assert np.max(mel_rel_db) <= 0.0 + 1e-6
    assert np.min(mel_rel_db) >= -80.0 - 1e-6


def test_map_key_known_speechbrain_mappings() -> None:
    assert (
        map_key("embedding_model.blocks.0.conv.conv.weight")
        == "embedding_model.blocks.0.conv.weight"
    )
    assert (
        map_key("embedding_model.blocks.1.norm.norm.bias")
        == "embedding_model.blocks.1.norm.bias"
    )
    assert (
        map_key("embedding_model.blocks.1.se_block.conv1.conv.weight")
        == "embedding_model.blocks.1.se_block.conv1.weight"
    )
    assert (
        map_key("embedding_model.asp_bn.norm.weight") == "embedding_model.asp_bn.weight"
    )
    assert map_key("embedding_model.fc.conv.weight") == "embedding_model.fc.weight"
    assert map_key("classifier.norm.num_batches_tracked") is None


def test_ecapa_tdnn_model_output_shape(ecapa_model: EcapaTdnnLid) -> None:
    dummy_mel = mx.random.normal((1, 12, 60))
    logits = ecapa_model(dummy_mel)
    mx.eval(logits)
    assert logits.shape == (1, 107)


@pytest.mark.skipif(
    not ECAPA_WEIGHTS_PATH.exists(),
    reason="ECAPA weights not found at mlx/weights/ecapa_tdnn_lid107.safetensors",
)
def test_ecapa_weights_file_available_for_optional_weighted_tests() -> None:
    assert ECAPA_WEIGHTS_PATH.exists()


@pytest.mark.skipif(
    not MMS_CONFIG_PATH.exists() or not MMS_WEIGHTS_PATH.exists(),
    reason="MMS-LID-256 config/weights not found in HuggingFace cache",
)
def test_mms_lid_256_output_shape(mms_model: MmsLid256) -> None:
    dummy_waveform = mx.zeros((1, 400), dtype=mx.float32)
    logits = mms_model(dummy_waveform)
    mx.eval(logits)
    assert logits.shape == (1, 256)


def test_speechbrain_mel_filterbank_shape_nonnegative_and_peak_level() -> None:
    filterbank = speechbrain_mel_filterbank(16000, 400, 60)
    assert filterbank.shape == (60, 201)
    assert np.all(filterbank >= 0.0)

    peaks = np.max(filterbank, axis=1)
    assert np.all(peaks <= 1.0 + 1e-6)
    assert np.all(peaks >= 0.35)
