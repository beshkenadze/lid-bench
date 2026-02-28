#!/usr/bin/env python3
"""
MLX ECAPA-TDNN Language Identification (107 languages).

Reimplements SpeechBrain ECAPA-TDNN architecture in pure MLX.
Loads weights converted from speechbrain/lang-id-voxlingua107-ecapa.

Architecture (from actual checkpoint):
  blocks.0: TDNNBlock(60→1024, k=5)
  blocks.1: SERes2NetBlock(1024, scale=8, k=3, dil=2)
  blocks.2: SERes2NetBlock(1024, scale=8, k=3, dil=3)
  blocks.3: SERes2NetBlock(1024, scale=8, k=3, dil=4)
  mfa: TDNNBlock(3072→3072, k=1)
  asp: AttentiveStatisticsPooling(3072, global_context=True) → 6144
  asp_bn: BatchNorm(6144)
  fc: Conv1d(6144→256, k=1)
  classifier: BN(256) → Linear(256,512) → BN(512) → Linear(512,107)

Input: raw audio (16kHz float32)
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

# Audio params (SpeechBrain defaults)
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
WIN_LENGTH = 400
N_MELS = 60


# ============================================================
# Mel Spectrogram (SpeechBrain-compatible)
# ============================================================


def speechbrain_mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """SpeechBrain symmetric triangular filterbank."""

    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    low_mel = hz_to_mel(0.0)
    high_mel = hz_to_mel(sr / 2.0)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    fft_bins = np.linspace(0, sr / 2, n_fft // 2 + 1)

    filterbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(n_mels):
        f_left, f_center, f_right = hz_points[m], hz_points[m + 1], hz_points[m + 2]
        band_left = f_center - f_left
        band_right = f_right - f_center
        for k in range(len(fft_bins)):
            f = fft_bins[k]
            if f_left <= f <= f_center and band_left > 0:
                filterbank[m, k] = (f - f_left) / band_left
            elif f_center < f <= f_right and band_right > 0:
                filterbank[m, k] = (f_right - f) / band_right
    return filterbank


def compute_mel_spectrogram(audio: np.ndarray) -> mx.array:
    """Compute SpeechBrain-compatible log-mel spectrogram.

    Returns: mx.array shape (1, T, 60)
    """
    # Periodic Hamming window
    n = np.arange(WIN_LENGTH, dtype=np.float32)
    window = 0.54 - 0.46 * np.cos(2.0 * np.pi * n / WIN_LENGTH)

    # Pad for center=True
    pad_len = N_FFT // 2
    audio_padded = np.pad(audio, (pad_len, pad_len), mode="constant")

    # Frame
    num_frames = 1 + (len(audio_padded) - N_FFT) // HOP_LENGTH
    frames = np.zeros((num_frames, N_FFT), dtype=np.float32)
    for i in range(num_frames):
        start = i * HOP_LENGTH
        frames[i] = audio_padded[start : start + N_FFT]
    frames *= window

    # FFT → power → mel → log
    fft_result = np.fft.rfft(frames, n=N_FFT)
    power_spectrum = np.abs(fft_result) ** 2
    mel_fb = speechbrain_mel_filterbank(SAMPLE_RATE, N_FFT, N_MELS)
    mel_spec = power_spectrum @ mel_fb.T
    mel_spec = np.maximum(mel_spec, 1e-10)
    log_mel = 10.0 * np.log10(mel_spec)
    max_val = log_mel.max()
    log_mel = np.maximum(log_mel, max_val - 80.0)

    return mx.array(log_mel[np.newaxis, :, :])


# ============================================================
# ECAPA-TDNN Model (matching SpeechBrain checkpoint exactly)
# ============================================================


class TDNNBlock(nn.Module):
    """Conv1d + ReLU + BatchNorm."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        groups: int = 1,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        self.norm = nn.BatchNorm(out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        return self.norm(nn.relu(self.conv(x)))


class Res2NetBlock(nn.Module):
    """Multi-scale Res2Net block with scale=8."""

    def __init__(
        self, channels: int, kernel_size: int = 3, dilation: int = 1, scale: int = 8
    ):
        super().__init__()
        self.scale = scale
        hidden = channels // scale
        self.blocks = [
            TDNNBlock(hidden, hidden, kernel_size, dilation=dilation)
            for _ in range(scale - 1)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        chunks = mx.split(x, self.scale, axis=-1)
        y = [chunks[0]]
        for i, block in enumerate(self.blocks):
            inp = chunks[i + 1] + y[-1] if i > 0 else chunks[i + 1]
            y.append(block(inp))
        return mx.concatenate(y, axis=-1)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation via Conv1d(k=1) — matches SpeechBrain's se_block."""

    def __init__(self, in_dim: int, bottleneck: int = 128):
        super().__init__()
        # SpeechBrain uses Conv1d(k=1) for SE, stored as conv1/conv2
        self.conv1 = nn.Conv1d(in_dim, bottleneck, kernel_size=1)
        self.conv2 = nn.Conv1d(bottleneck, in_dim, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, C)
        s = mx.mean(x, axis=1, keepdims=True)  # (B, 1, C)
        s = nn.relu(self.conv1(s))
        s = mx.sigmoid(self.conv2(s))
        return x * s


class SERes2NetBlock(nn.Module):
    """SE-Res2Net block: tdnn1 → res2net → tdnn2 → SE + residual."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        res2net_scale: int = 8,
        se_channels: int = 128,
    ):
        super().__init__()
        self.tdnn1 = TDNNBlock(channels, channels, kernel_size=1)
        self.res2net_block = Res2NetBlock(
            channels, kernel_size=kernel_size, dilation=dilation, scale=res2net_scale
        )
        self.tdnn2 = TDNNBlock(channels, channels, kernel_size=1)
        self.se_block = SEBlock(channels, se_channels)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        out = self.tdnn1(x)
        out = self.res2net_block(out)
        out = self.tdnn2(out)
        out = self.se_block(out)
        return out + residual


class AttentiveStatisticsPooling(nn.Module):
    """ASP with global context (SpeechBrain style).

    Input gets concatenated with global mean and std → 3x channels for attention.
    """

    def __init__(self, channels: int, attention_channels: int = 128):
        super().__init__()
        # tdnn input = channels * 3 (frame + global_mean + global_std)
        self.tdnn = TDNNBlock(channels * 3, attention_channels, kernel_size=1)
        self.conv = nn.Conv1d(attention_channels, channels, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, C)
        # Global context
        mean = mx.mean(x, axis=1, keepdims=True)  # (B, 1, C)
        std = mx.sqrt(mx.var(x, axis=1, keepdims=True) + 1e-9)
        mean_expanded = mx.broadcast_to(mean, x.shape)
        std_expanded = mx.broadcast_to(std, x.shape)

        # Concatenate: [frame, global_mean, global_std]
        attn_input = mx.concatenate(
            [x, mean_expanded, std_expanded], axis=-1
        )  # (B, T, 3*C)

        # Attention weights
        attn = self.tdnn(attn_input)  # (B, T, att_channels)
        attn = mx.tanh(attn)
        attn = self.conv(attn)  # (B, T, C)
        attn = mx.softmax(attn, axis=1)

        # Weighted statistics
        weighted_mean = mx.sum(attn * x, axis=1)  # (B, C)
        weighted_var = mx.sum(attn * (x**2), axis=1) - weighted_mean**2
        weighted_std = mx.sqrt(mx.maximum(weighted_var, 1e-9))

        return mx.concatenate([weighted_mean, weighted_std], axis=-1)  # (B, 2*C)


class EcapaTdnnEmbedding(nn.Module):
    """ECAPA-TDNN embedding extractor (channels=1024, embed_dim=256)."""

    def __init__(self, n_mels: int = 60, channels: int = 1024, embed_dim: int = 256):
        super().__init__()
        self.blocks = [
            TDNNBlock(n_mels, channels, kernel_size=5),
            SERes2NetBlock(channels, kernel_size=3, dilation=2),
            SERes2NetBlock(channels, kernel_size=3, dilation=3),
            SERes2NetBlock(channels, kernel_size=3, dilation=4),
        ]
        self.mfa = TDNNBlock(channels * 3, channels * 3, kernel_size=1)
        self.asp = AttentiveStatisticsPooling(channels * 3, attention_channels=128)
        self.asp_bn = nn.BatchNorm(channels * 6)  # 6144
        self.fc = nn.Conv1d(channels * 6, embed_dim, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, n_mels)
        xl = []
        out = self.blocks[0](x)
        for block in self.blocks[1:]:
            out = block(out)
            xl.append(out)

        out = mx.concatenate(xl, axis=-1)  # (B, T, channels*3)
        out = self.mfa(out)
        out = self.asp(out)  # (B, channels*6)
        out = self.asp_bn(out)
        out = mx.expand_dims(out, axis=1)  # (B, 1, channels*6)
        out = self.fc(out)  # (B, 1, embed_dim)
        return out


class Classifier(nn.Module):
    """SpeechBrain classifier: BN(256) → Linear(256,512) + BN(512) → Linear(512,107)."""

    def __init__(
        self, embed_dim: int = 256, hidden_dim: int = 512, num_classes: int = 107
    ):
        super().__init__()
        self.norm = nn.BatchNorm(embed_dim)
        self.DNN = DNN(embed_dim, hidden_dim)
        self.out = ClassifierLinear(hidden_dim, num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, 1, embed_dim)
        x = x.squeeze(1)  # (B, embed_dim)
        x = self.norm(x)
        x = self.DNN(x)
        x = self.out(x)
        return nn.log_softmax(x, axis=-1)


class DNN(nn.Module):
    """Single hidden block: Linear + BN + ReLU."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.block_0 = DNNBlock(in_dim, out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.block_0(x)


class DNNBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = DNNLinear(in_dim, out_dim)
        self.norm = nn.BatchNorm(out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.relu(self.norm(self.linear(x)))


class DNNLinear(nn.Module):
    """SpeechBrain Linear layer (weight key = 'w')."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.w = nn.Linear(in_dim, out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w(x)


class ClassifierLinear(nn.Module):
    """SpeechBrain output layer (weight key = 'w')."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.w = nn.Linear(in_dim, out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w(x)


class EcapaTdnnLid(nn.Module):
    """Full ECAPA-TDNN Language ID model."""

    def __init__(
        self,
        n_mels: int = 60,
        channels: int = 1024,
        embed_dim: int = 256,
        num_classes: int = 107,
    ):
        super().__init__()
        self.embedding_model = EcapaTdnnEmbedding(n_mels, channels, embed_dim)
        self.classifier = Classifier(embed_dim, hidden_dim=512, num_classes=num_classes)

    def __call__(self, mel_features: mx.array) -> mx.array:
        embeddings = self.embedding_model(mel_features)  # (B, 1, embed_dim)
        log_probs = self.classifier(embeddings)  # (B, num_classes)
        return log_probs


# ============================================================
# Weight Loading
# ============================================================


def load_model(weights_path: Path, labels_path: Path) -> tuple:
    """Load model with converted weights."""
    with open(labels_path) as f:
        labels = json.load(f)

    model = EcapaTdnnLid(
        n_mels=N_MELS, channels=1024, embed_dim=256, num_classes=len(labels)
    )
    model.eval()

    raw_weights = mx.load(str(weights_path))

    # Map SpeechBrain keys to our model keys
    mapped = {}
    for key, value in raw_weights.items():
        mlx_key = map_key(key)
        if mlx_key is not None:
            mapped[mlx_key] = value

    model.load_weights(list(mapped.items()), strict=False)
    return model, labels


def map_key(sb_key: str) -> str | None:
    """Map SpeechBrain key to MLX model key."""
    key = sb_key

    if "num_batches_tracked" in key:
        return None

    # 1. Flatten double-nesting: .conv.conv. → .conv., .norm.norm. → .norm.
    key = key.replace(".conv.conv.", ".conv.")
    key = key.replace(".norm.norm.", ".norm.")

    # 2. SE block Conv1d wrappers: se_block.conv1.conv. → se_block.conv1.
    #    (not caught by .conv.conv. because it's conv1.conv. not conv.conv.)
    key = key.replace(".se_block.conv1.conv.", ".se_block.conv1.")
    key = key.replace(".se_block.conv2.conv.", ".se_block.conv2.")

    # 3. ASP_BN: asp_bn.norm. → asp_bn. (single .norm., not .norm.norm.)
    key = key.replace(".asp_bn.norm.", ".asp_bn.")

    # 4. FC: fc.conv. → fc. (single .conv., not .conv.conv.)
    key = key.replace(".fc.conv.", ".fc.")

    return key
# ============================================================
# Inference & Benchmark
# ============================================================


def predict(
    model: EcapaTdnnLid, audio: np.ndarray, labels: dict, top_k: int = 5
) -> list[tuple[str, float]]:
    mel = compute_mel_spectrogram(audio)
    log_probs = model(mel)
    mx.eval(log_probs)

    probs = mx.exp(log_probs).tolist()[0]
    indexed = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
    return [(labels.get(str(i), f"?{i}"), p) for i, p in indexed[:top_k]]


def benchmark(
    model: EcapaTdnnLid, audio: np.ndarray, n_warmup: int = 3, n_runs: int = 10
) -> dict:
    mel = compute_mel_spectrogram(audio)

    # Compile for graph fusion (matches Swift MLX compile() usage)
    compiled_forward = mx.compile(model)

    for _ in range(n_warmup):
        mx.eval(compiled_forward(mel))

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        mx.eval(compiled_forward(mel))
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return {
        "mean_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "min_ms": np.min(times) * 1000,
        "max_ms": np.max(times) * 1000,
        "n_runs": n_runs,
        "audio_duration_s": len(audio) / SAMPLE_RATE,
        "mel_frames": mel.shape[1],
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

    parser = argparse.ArgumentParser(description="MLX ECAPA-TDNN Language ID")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument(
        "--weights", default=str(WEIGHTS_DIR / "ecapa_tdnn_lid107.safetensors")
    )
    parser.add_argument(
        "--labels", default=str(MODELS_DIR / "ecapa_tdnn_lid107_labels.json")
    )
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    print(f"MLX device: {mx.default_device()}")
    print("Loading model...")
    model, labels = load_model(Path(args.weights), Path(args.labels))

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
        print(
            f"  Audio: {stats['audio_duration_s']:.1f}s ({stats['mel_frames']} frames)"
        )
        print(f"  RTF:  {stats['mean_ms'] / (stats['audio_duration_s'] * 1000):.4f}")


if __name__ == "__main__":
    main()
