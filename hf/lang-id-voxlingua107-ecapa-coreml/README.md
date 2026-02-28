---
library_name: coreml
tags:
  - coreml
  - audio
  - language-identification
  - apple-silicon
  - swift
  - ecapa-tdnn
  - speechbrain
license: apache-2.0
base_model: speechbrain/lang-id-voxlingua107-ecapa
pipeline_tag: audio-classification
---

# ECAPA-TDNN VoxLingua107 CoreML

CoreML conversion of [speechbrain/lang-id-voxlingua107-ecapa](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa) for native inference on Apple Silicon (macOS 14+ / iOS 17+).

Identifies **107 spoken languages** from log-mel spectrogram. No Python required at runtime.

## Model Details

| Property | Value |
|----------|-------|
| Source | [speechbrain/lang-id-voxlingua107-ecapa](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa) |
| Architecture | ECAPA-TDNN |
| Languages | 107 |
| Input | Log-mel spectrogram `[1, T, 60]` float32 |
| Output | Log-probabilities `[1, 107]` |
| Size | 81 MB |
| Precision | FP32 |
| Min deployment | macOS 14 / iOS 17 |
| Compute units | **CPU + GPU** (ANE not used) |

## Benchmark Results

Tested on Apple Silicon (M1, Metal GPU, `.cpuAndGPU`):

| Audio | Predicted | Confidence | Inference Time | Mel Time |
|-------|-----------|------------|----------------|----------|
| Russian (10s) | ru: Russian | 99.7% | 0.017s | 0.019s |
| English (30s) | en: English | 98.6% | 2.0s | 0.053s |

15-50x faster than MMS-LID-256 with comparable accuracy.

## Usage (Swift)

```swift
import CoreML

let compiledURL = try MLModel.compileModel(at: URL(fileURLWithPath: "EcapaTdnnLid107.mlpackage"))
let config = MLModelConfiguration()
config.computeUnits = .cpuAndGPU  // ANE provides no benefit for this model
let model = try MLModel(contentsOf: compiledURL, configuration: config)

// melFrames: [[Float]] — log-mel spectrogram [T][60]
let T = melFrames.count
let inputArray = try MLMultiArray(shape: [1, NSNumber(value: T), 60], dataType: .float32)
for t in 0..<T {
    for f in 0..<60 {
        inputArray[[0, NSNumber(value: t), NSNumber(value: f)]] = NSNumber(value: melFrames[t][f])
    }
}

let input = try MLDictionaryFeatureProvider(dictionary: [
    "mel_features": MLFeatureValue(multiArray: inputArray),
])
let output = try model.prediction(from: input)
```

## Mel Spectrogram

The model expects a log-mel spectrogram computed with SpeechBrain-compatible parameters. **This must be computed on-device** (not baked into the CoreML model).

| Parameter | Value |
|-----------|-------|
| Sample rate | 16000 Hz |
| n_fft | 400 |
| hop_length | 160 (10ms) |
| win_length | 400 (25ms) |
| n_mels | 60 |
| Window | Hamming (periodic) |
| Center padding | Yes (zero-pad) |
| Filterbank | SpeechBrain symmetric triangular |
| Log scale | `10 * log10(clamp(x, 1e-10))` |
| Dynamic range | top_db=80 per sequence |

### Implementation Notes

1. **DFT size 400 is not power-of-2** — Apple's `vDSP_fft_zrip` silently computes wrong results. Use manual DFT via `cblas_sgemv` with precomputed twiddle factors.
2. **SpeechBrain filterbank differs from HTK** — uses symmetric triangular filters: `band[m] = hz[m+1] - hz[m]` applied equally to both sides.
3. **Periodic Hamming window** — `vDSP_hamm_window` generates symmetric windows. Create N+1 symmetric, take first N for periodic.
4. **CMVN stats are identity** (all zeros) — normalization can be skipped.

## Files

- `EcapaTdnnLid107.mlpackage/` — CoreML model
- `ecapa_tdnn_lid107_labels.json` — language label mapping (index → "code: Name")

## Conversion

Converted via `torch.jit.trace` → `coremltools 9.0`. See [conversion script](https://github.com/beshkenadze/lid-bench).

## Full Inference Code

Complete Swift CLI with audio loading, mel spectrogram (Accelerate/vDSP), and inference:
**[github.com/beshkenadze/lid-bench](https://github.com/beshkenadze/lid-bench)**

## License

Apache 2.0 (same as the original model)
