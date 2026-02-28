#!/usr/bin/env python3
"""
MLX MMS-LID-256 Language Identification (256 languages).

Reimplements facebook/mms-lid-256 (Wav2Vec2ForSequenceClassification) in pure MLX.
Loads weights directly from HuggingFace safetensors.

Architecture:
  Feature Extractor: 7 × Conv1d layers with LayerNorm + GELU
    Layer 0: Conv1d(1→512, k=10, s=5) + LayerNorm(512) + GELU
    Layers 1-4: Conv1d(512→512, k=3, s=2) + LayerNorm(512) + GELU
    Layer 5: Conv1d(512→512, k=2, s=2) + LayerNorm(512) + GELU
    Layer 6: Conv1d(512→512, k=2, s=2) + LayerNorm(512) + GELU
    Total stride: 5×2×2×2×2×2×2 = 320
  Feature Projection: LayerNorm(512) + Linear(512→1280)
  Positional Conv: Weight-normed Conv1d(1280→1280, k=128, groups=16, pad=64)
  48 × Transformer (stable layer norm):
    LayerNorm → MultiHeadAttention(1280, 16 heads) → dropout → residual
    LayerNorm → FFN(1280→5120→1280) → dropout → residual
  Encoder LayerNorm(1280)
  Classifier: mean pool → Linear(1280→1024) → Linear(1024→256)

Input: raw 16kHz waveform (float32), zero-mean unit-variance normalized
Output: language prediction with probabilities
"""

import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

MODELS_DIR = Path(__file__).parent.parent / "models"
WEIGHTS_DIR = Path(__file__).parent / "weights"
SAMPLE_RATE = 16000

# HuggingFace cached model path
HF_MODEL_DIR = (
    Path.home()
    / ".cache/huggingface/hub/models--facebook--mms-lid-256/snapshots/edc73fd00996e671dfc59d16436a29b12b10588a"
)


# ============================================================
# Feature Extractor (7 Conv1d layers)
# ============================================================


class FeatureExtractorLayer(nn.Module):
    """Single conv layer of the feature extractor: Conv1d + LayerNorm + GELU."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, bias=True
        )
        self.layer_norm = nn.LayerNorm(out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv(x)
        x = self.layer_norm(x)
        x = nn.gelu(x)
        return x


class FeatureExtractor(nn.Module):
    """7-layer convolutional feature extractor from raw waveform."""

    def __init__(self):
        super().__init__()
        conv_dims = [512] * 7
        conv_kernels = [10, 3, 3, 3, 3, 2, 2]
        conv_strides = [5, 2, 2, 2, 2, 2, 2]

        in_channels = [1] + conv_dims[:-1]
        self.conv_layers = [
            FeatureExtractorLayer(ic, oc, k, s)
            for ic, oc, k, s in zip(in_channels, conv_dims, conv_kernels, conv_strides)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, 1) — raw waveform with channel dim
        for layer in self.conv_layers:
            x = layer(x)
        return x  # (B, T', 512)


# ============================================================
# Feature Projection
# ============================================================


class FeatureProjection(nn.Module):
    """LayerNorm(512) + Linear(512→1280)."""

    def __init__(self, in_dim: int = 512, out_dim: int = 1280):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_dim)
        self.projection = nn.Linear(in_dim, out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.layer_norm(x)
        x = self.projection(x)
        return x


# ============================================================
# Positional Convolutional Embedding
# ============================================================


class PositionalConvEmbedding(nn.Module):
    """Weight-normed grouped Conv1d for positional encoding.

    Conv1d(1280→1280, kernel=128, groups=16, padding=64).
    Weights stored as weight_g (gain) and weight_v (direction) with dim=2 weight norm.
    We precompute the full weight at load time.
    """

    def __init__(
        self, hidden_size: int = 1280, kernel_size: int = 128, groups: int = 16
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size,
            groups=groups,
            padding=kernel_size // 2,
            bias=True,
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, C)
        x = self.conv(x)
        x = nn.gelu(x)
        return x


# ============================================================
# Transformer Layer (Stable Layer Norm variant)
# ============================================================


class Wav2Vec2Attention(nn.Module):
    """Multi-head attention with separate q/k/v/out projections."""

    def __init__(self, hidden_size: int = 1280, num_heads: int = 16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        B, T, _ = x.shape
        q = (
            self.q_proj(x)
            .reshape(B, T, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k_proj(x)
            .reshape(B, T, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(x)
            .reshape(B, T, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        scale = self.head_dim**-0.5
        attn = (q @ k.transpose(0, 1, 3, 2)) * scale
        attn = mx.softmax(attn, axis=-1)

        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.out_proj(out)


class Wav2Vec2FeedForward(nn.Module):
    """FFN: Linear(1280→5120) + GELU + Linear(5120→1280)."""

    def __init__(self, hidden_size: int = 1280, intermediate_size: int = 5120):
        super().__init__()
        self.intermediate_dense = nn.Linear(hidden_size, intermediate_size)
        self.output_dense = nn.Linear(intermediate_size, hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.intermediate_dense(x)
        x = nn.gelu(x)
        x = self.output_dense(x)
        return x


class Wav2Vec2EncoderLayer(nn.Module):
    """Single transformer layer with stable layer norm (pre-norm)."""

    def __init__(
        self,
        hidden_size: int = 1280,
        num_heads: int = 16,
        intermediate_size: int = 5120,
    ):
        super().__init__()
        self.attention = Wav2Vec2Attention(hidden_size, num_heads)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-5)
        self.feed_forward = Wav2Vec2FeedForward(hidden_size, intermediate_size)
        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=1e-5)

    def __call__(self, x: mx.array) -> mx.array:
        # Pre-norm (stable layer norm)
        residual = x
        x = self.layer_norm(x)
        x = self.attention(x)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = self.feed_forward(x)
        x = residual + x
        return x


# ============================================================
# Full Encoder
# ============================================================


class Wav2Vec2Encoder(nn.Module):
    """Positional conv + 48 transformer layers + final LayerNorm."""

    def __init__(
        self,
        hidden_size: int = 1280,
        num_layers: int = 48,
        num_heads: int = 16,
        intermediate_size: int = 5120,
    ):
        super().__init__()
        self.pos_conv_embed = PositionalConvEmbedding(hidden_size)
        self.layers = [
            Wav2Vec2EncoderLayer(hidden_size, num_heads, intermediate_size)
            for _ in range(num_layers)
        ]
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-5)

    def __call__(self, x: mx.array) -> mx.array:
        # Add positional embeddings
        pos = self.pos_conv_embed(x)
        # Trim to match sequence length (padding=64 with k=128 adds 1 extra)
        pos = pos[:, : x.shape[1], :]
        x = x + pos

        for layer in self.layers:
            x = layer(x)

        x = self.layer_norm(x)
        return x


# ============================================================
# Full Wav2Vec2 + Classifier
# ============================================================


class MmsLid256(nn.Module):
    """MMS-LID-256: Wav2Vec2ForSequenceClassification.

    Input: raw 16kHz waveform, zero-mean unit-variance normalized.
    Output: log-softmax over 256 language classes.
    """

    def __init__(self, num_classes: int = 256):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.feature_projection = FeatureProjection(512, 1280)
        self.encoder = Wav2Vec2Encoder(1280, 48, 16, 5120)
        self.projector = nn.Linear(1280, 1024)
        self.classifier = nn.Linear(1024, num_classes)

    def __call__(self, waveform: mx.array) -> mx.array:
        """Forward pass.

        Args:
            waveform: (B, T) raw 16kHz audio, already normalized.
        Returns:
            log_probs: (B, num_classes)
        """
        # Add channel dim for Conv1d: (B, T) → (B, T, 1)
        x = mx.expand_dims(waveform, axis=-1)

        # Feature extraction
        x = self.feature_extractor(x)  # (B, T', 512)

        # Feature projection
        x = self.feature_projection(x)  # (B, T', 1280)

        # Transformer encoder
        x = self.encoder(x)  # (B, T', 1280)

        # Mean pool over time
        x = mx.mean(x, axis=1)  # (B, 1280)

        # Classifier head
        x = self.projector(x)  # (B, 1024)
        logits = self.classifier(x)  # (B, 256)

        return nn.log_softmax(logits, axis=-1)


# ============================================================
# Weight Loading from HuggingFace
# ============================================================


def _compute_weight_norm(weight_g: mx.array, weight_v: mx.array) -> mx.array:
    """Compute weight from weight_norm parameters (dim=2).

    weight_g: [1, 1, kernel_size] — gain per output position
    weight_v: [out_channels, groups_dim, kernel_size] — direction
    result: normalized weight = g * v / ||v||
    """
    # Norm over dims 0 and 1 (out_channels and groups_dim), keeping kernel dim
    norm = mx.sqrt(mx.sum(weight_v**2, axis=(0, 1), keepdims=True) + 1e-12)
    return weight_g * weight_v / norm


def load_hf_weights(model: MmsLid256, model_dir: Path) -> None:
    """Load weights from HuggingFace safetensors, mapping keys to our model structure."""
    from safetensors import safe_open

    weights_path = model_dir / "model.safetensors"
    f = safe_open(str(weights_path), framework="numpy")

    mapped = {}

    for hf_key in sorted(f.keys()):
        tensor = f.get_tensor(hf_key)

        # Skip training-only tensors
        if "masked_spec_embed" in hf_key:
            continue
        if "adapter_layer" in hf_key:
            continue

        # Map HF key → our model key
        mlx_key = _map_hf_key(hf_key)
        if mlx_key is None:
            continue

        # Conv1d weight axis swap: HF [out, in, kernel] → MLX [out, kernel, in]
        if mlx_key.endswith(".conv.weight"):
            tensor = np.swapaxes(tensor, 1, 2)

        mapped[mlx_key] = mx.array(tensor)

    # Handle positional conv weight norm: combine weight_g + weight_v → weight
    weight_g_key = "encoder.pos_conv_embed.conv.weight_g"
    weight_v_key = "encoder.pos_conv_embed.conv.weight_v"
    if weight_g_key in mapped and weight_v_key in mapped:
        full_weight = _compute_weight_norm(mapped[weight_g_key], mapped[weight_v_key])
        # full_weight: [1280, 80, 128] in HF format → swap to MLX [1280, 128, 80]
        full_weight = mx.swapaxes(full_weight, 1, 2)
        mapped["encoder.pos_conv_embed.conv.weight"] = full_weight
        del mapped[weight_g_key]
        del mapped[weight_v_key]

    model.load_weights(list(mapped.items()), strict=False)


def _map_hf_key(hf_key: str) -> str | None:
    """Map HuggingFace Wav2Vec2 key to our model key structure."""

    # Classifier head (no wav2vec2. prefix)
    if hf_key.startswith("projector."):
        return hf_key
    if hf_key.startswith("classifier."):
        return hf_key

    # Strip wav2vec2. prefix
    if not hf_key.startswith("wav2vec2."):
        return None
    key = hf_key[len("wav2vec2.") :]

    # Feature extractor: feature_extractor.conv_layers.N.conv.weight/bias
    #                     feature_extractor.conv_layers.N.layer_norm.weight/bias
    if key.startswith("feature_extractor."):
        return key

    # Feature projection: feature_projection.layer_norm.weight/bias
    #                      feature_projection.projection.weight/bias
    if key.startswith("feature_projection."):
        return key

    # Encoder: encoder.pos_conv_embed.conv.{bias,weight_g,weight_v}
    #          encoder.layers.N.attention.{q,k,v,out}_proj.{weight,bias}
    #          encoder.layers.N.feed_forward.{intermediate_dense,output_dense}.{weight,bias}
    #          encoder.layers.N.layer_norm.{weight,bias}
    #          encoder.layers.N.final_layer_norm.{weight,bias}
    #          encoder.layer_norm.{weight,bias}
    if key.startswith("encoder."):
        return key

    return None


# ============================================================
# Inference & Benchmark
# ============================================================


def load_model(model_dir: Path | None = None, labels_path: Path | None = None) -> tuple:
    """Load MMS-LID-256 model with HF weights."""
    if model_dir is None:
        model_dir = HF_MODEL_DIR
    if labels_path is None:
        labels_path = MODELS_DIR / "mms_lid_256_labels.json"

    with open(labels_path) as f:
        labels = json.load(f)

    model = MmsLid256(num_classes=len(labels))
    model.eval()
    load_hf_weights(model, model_dir)

    return model, labels


def normalize_waveform(audio: np.ndarray) -> mx.array:
    """Zero-mean unit-variance normalization (HF preprocessor default)."""
    mean = np.mean(audio)
    std = np.std(audio)
    if std > 0:
        audio = (audio - mean) / std
    else:
        audio = audio - mean
    return mx.array(audio[np.newaxis, :])  # (1, T)


def predict(
    model: MmsLid256, audio: np.ndarray, labels: dict, top_k: int = 5
) -> list[tuple[str, float]]:
    waveform = normalize_waveform(audio)
    log_probs = model(waveform)
    mx.eval(log_probs)

    probs = mx.exp(log_probs).tolist()[0]
    indexed = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
    return [(labels.get(str(i), f"?{i}"), p) for i, p in indexed[:top_k]]


def benchmark(
    model: MmsLid256, audio: np.ndarray, n_warmup: int = 3, n_runs: int = 10
) -> dict:
    waveform = normalize_waveform(audio)

    # Compile for graph fusion (matches Swift MLX compile() usage)
    compiled_forward = mx.compile(model)

    for _ in range(n_warmup):
        mx.eval(compiled_forward(waveform))

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        mx.eval(compiled_forward(waveform))
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return {
        "mean_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "min_ms": np.min(times) * 1000,
        "max_ms": np.max(times) * 1000,
        "n_runs": n_runs,
        "audio_duration_s": len(audio) / SAMPLE_RATE,
    }


def load_audio_file(path: str) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        duration = len(audio) / sr
        n_samples = int(duration * SAMPLE_RATE)
        x_old = np.linspace(0, duration, len(audio))
        x_new = np.linspace(0, duration, n_samples)
        audio = np.interp(x_new, x_old, audio)
    return audio


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MLX MMS-LID-256 Language ID")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--model-dir", default=str(HF_MODEL_DIR))
    parser.add_argument("--labels", default=str(MODELS_DIR / "mms_lid_256_labels.json"))
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    print(f"MLX device: {mx.default_device()}")
    print("Loading model...")
    model, labels = load_model(Path(args.model_dir), Path(args.labels))

    print(f"Loading audio: {args.audio}")
    audio = load_audio_file(args.audio)
    print(f"  Duration: {len(audio) / SAMPLE_RATE:.1f}s, samples: {len(audio)}")

    print(f"\nPrediction (top {args.top_k}):")
    results = predict(model, audio, labels, top_k=args.top_k)
    for label, prob in results:
        print(f"  {label}: {prob:.4f} ({prob * 100:.1f}%)")

    if args.benchmark:
        print(f"\nBenchmark ({10} runs, {3} warmup):")
        stats = benchmark(model, audio)
        print(f"  Mean: {stats['mean_ms']:.1f}ms ± {stats['std_ms']:.1f}ms")
        print(f"  Min:  {stats['min_ms']:.1f}ms")
        print(f"  Max:  {stats['max_ms']:.1f}ms")
        print(f"  Audio: {stats['audio_duration_s']:.1f}s")
        print(f"  RTF:  {stats['mean_ms'] / (stats['audio_duration_s'] * 1000):.4f}")


if __name__ == "__main__":
    main()
