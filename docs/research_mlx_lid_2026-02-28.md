# Research Report: MLX for Language Identification on Apple Silicon

**Date**: 2026-02-28  
**Context**: CoreML LID benchmark (MMS-LID-256 + ECAPA-TDNN), verified ANE NOT used — both models run on Metal GPU only.

## Summary

MLX is a viable alternative to CoreML for LID inference. Both frameworks use the same Metal GPU hardware. **ECAPA-TDNN architecture is already implemented in mlx-audio**, and Wav2Vec2 base encoder exists too — only classification heads are missing (~1-3 days work). Performance is expected to be within ±1.5x of CoreML GPU. The main advantage of MLX is Python/Swift flexibility, iOS via mlx-swift, and potential community contribution (issue #518 already requests MMS-LID in mlx-audio).

## Critical Finding: ANE Not Used

Empirical compute unit test (M1, 10s Russian audio):

| Compute Units | MMS-LID-256 | ECAPA-TDNN |
|---|---|---|
| cpuOnly | 0.837s | 0.055s |
| **cpuAndGPU** | **0.250s** ✅ | **0.017s** ✅ |
| all (ANE+GPU+CPU) | 3.286s ❌ 13x slower | 0.017s (ANE ignored) |

**Both models run exclusively on Metal GPU.** MMS-LID with `.all` is 13x slower due to failed ANE/GPU op splitting. ECAPA-TDNN ignores ANE entirely. This eliminates the "ANE advantage" argument for CoreML.

## MLX Ecosystem Status (Feb 2026)

### Existing Implementations

| Component | Status | Source |
|---|---|---|
| ECAPA-TDNN (full architecture) | ✅ Implemented | `mlx-audio` — `mlx_audio/tts/models/spark/modules/speaker/ecapa_tdnn.py` |
| Attentive Statistics Pooling | ✅ Implemented | `mlx-audio` — `pooling_layers.py` (ASTP, TSTP, MHASTP) |
| Wav2Vec2 encoder | ✅ Implemented | `mlx-audio` — `mlx_audio/stt/models/wav2vec/wav2vec.py` |
| Wav2Vec2 feature extractor | ✅ Implemented | `mlx-audio` — `feature_extractor.py` (7-layer CNN, group conv) |
| Audio DSP (STFT, mel, fbank) | ✅ Implemented | `mlx-audio` — `dsp.py` (Kaldi-compatible) |
| Classification head | ❌ Missing | Need `ForAudioClassification` wrapper |
| LID weight conversion | ❌ Missing | Need PyTorch→MLX safetensors script |

### MLX Layer Gap Analysis

All required ops exist in MLX:

| Required Op | MLX Support |
|---|---|
| Conv1d (+ groups) | ✅ `mlx.nn.Conv1d(groups=N)` |
| BatchNorm | ✅ `mlx.nn.BatchNorm` |
| LayerNorm | ✅ `mlx.nn.LayerNorm` |
| GroupNorm | ✅ `mlx.nn.GroupNorm` |
| Linear | ✅ `mlx.nn.Linear` |
| MultiHeadAttention | ✅ `mlx.nn.MultiHeadAttention` |
| GELU / ReLU / Sigmoid | ✅ All activations |
| Attentive pooling (mean/std/softmax) | ✅ Primitives available |
| SE block (Linear + sigmoid + multiply) | ✅ Primitives available |
| Res2Net (split/concat) | ✅ `mx.split`, `mx.concatenate` |

**Zero gaps at the primitive level.**

## Performance Projection

### Published Benchmarks (nearest proxies)

| Comparison | Source | Result |
|---|---|---|
| MLX GPU vs PyTorch/MPS (Whisper-tiny, M1) | LucasSte/MLX-vs-Pytorch | MLX **3.8x faster** |
| MLX GPU vs CoreML+ANE (Whisper large-v3-turbo, M4) | anvanvan/mac-whisper-speedtest | CoreML (ANE) 2.6x faster |
| MLX GPU vs CoreML+ANE (encoder prefill, iPhone 15) | SqueezeBits blog | CoreML ANE wins on prefill |
| BERT-base on MLX M1 GPU | arXiv:2510.18921 | ~179ms per inference |

### Projection for Our Models

| Model | CoreML GPU (measured) | MLX GPU (estimated) | Basis |
|---|---|---|---|
| MMS-LID-256 (300M params) | 0.250s | ~0.15–0.35s | BERT-base M1 = 179ms; similar encoder size |
| ECAPA-TDNN (6M params) | 0.017s | ~0.005–0.015s | Small model; MLX compile overhead ~0.4ms warm |

**Bottom line**: Within ±1.5x. Need empirical measurement to know direction.

## Implementation Paths

### Path A: Python MLX (quickest, ~1 day ECAPA / ~3 days MMS)

```
pip install mlx-audio
```

**ECAPA-TDNN**:
1. Extract `ecapa_tdnn.py` + `pooling_layers.py` from mlx-audio Spark TTS module
2. Add linear classification head (~30 lines)
3. Convert SpeechBrain weights → safetensors (transpose Conv1d: `(C_out, C_in, K)` → `(C_out, K, C_in)`)

**MMS-LID-256**:
1. Extend `wav2vec.py` with `Wav2Vec2ForAudioClassification` (~20 lines)
2. Update `sanitize()` to keep `projector.*` + `classifier.*` weights
3. HF safetensors load directly (MMS models already in safetensors format)

### Path B: Swift MLX (best for integration, ~1 week)

**mlx-swift** v0.30.6 — Apple-maintained, SPM package:

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.6"),
]
```

- Full nn layer parity with Python: Conv1d, BatchNorm, Linear, MultiHeadAttention
- `.safetensors` loading: `loadArrays(url:)`
- iOS 17+ / macOS 14+ / visionOS
- Same Metal kernels as Python MLX — zero overhead
- Reference: `mlx-audio-swift` Parakeet Conformer (Conv1d + BatchNorm + Linear)

### Path C: Contribute to mlx-audio (community impact)

Issue [#518](https://github.com/Blaizzy/mlx-audio/issues/518) "Add mms-lid" opened Feb 21, 2026. Contributing the classification head + weights would benefit the ecosystem.

## Recommendation

**Start with Path A (Python MLX) for ECAPA-TDNN** — lowest effort, fastest validation:

1. Add classification head to existing mlx-audio ECAPA-TDNN
2. Convert SpeechBrain weights
3. Run benchmark vs CoreML `.cpuAndGPU`
4. If MLX is competitive → port to Swift (Path B) for the final tool

This gives us an empirical answer in ~1 day instead of theoretical speculation.

## Key Risks

1. **SpeechBrain → mlx-audio ECAPA-TDNN mismatch**: mlx-audio's ECAPA-TDNN is from WeSpeaker, not SpeechBrain. Architecture may differ slightly (channel dims, pooling variant). Need to verify layer-by-layer.
2. **Mel spectrogram compatibility**: mlx-audio `dsp.py` has Kaldi-compatible fbank but SpeechBrain uses its own symmetric triangular filterbank. May need custom mel computation in MLX (we already have the math from our Swift implementation).
3. **First-call latency**: MLX compiles Metal shaders on first `mx.eval()`. Could add 1-2s cold start.

## References

- [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio) — 6k stars, TTS/STT/STS
- [DePasqualeOrg/mlx-swift-audio](https://github.com/DePasqualeOrg/mlx-swift-audio) — Swift audio models
- [ml-explore/mlx-swift](https://github.com/ml-explore/mlx-swift) — v0.30.6, official Swift API
- [ml-explore/mlx-examples/whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)
- [SqueezeBits blog](https://blog.squeezebits.com/disaggregated-inference-on-apple-silicon-npu-prefill-and-gpu-decode-67176) — MLX vs CoreML on iPhone
- [arXiv:2510.18921](https://arxiv.org/html/2510.18921v1) — BERT/RoBERTa on MLX
- [mlx-audio issue #518](https://github.com/Blaizzy/mlx-audio/issues/518) — "Add mms-lid" request
