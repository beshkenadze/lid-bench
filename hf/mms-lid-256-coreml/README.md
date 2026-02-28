---
library_name: coreml
tags:
  - coreml
  - audio
  - language-identification
  - apple-silicon
  - swift
  - wav2vec2
license: cc-by-nc-4.0
base_model: facebook/mms-lid-256
pipeline_tag: audio-classification
---

# MMS-LID-256 CoreML

CoreML conversion of [facebook/mms-lid-256](https://huggingface.co/facebook/mms-lid-256) for native inference on Apple Silicon (macOS 14+ / iOS 17+).

Identifies **256 spoken languages** from raw audio waveform. No Python required at runtime.

## Model Details

| Property | Value |
|----------|-------|
| Source | [facebook/mms-lid-256](https://huggingface.co/facebook/mms-lid-256) |
| Architecture | Wav2Vec2 for Sequence Classification |
| Languages | 256 |
| Input | Raw waveform `[1, N]`, 16kHz float32 |
| Output | Logits `[1, 256]` |
| Size | 1.8 GB |
| Precision | FP16 |
| Min deployment | macOS 14 / iOS 17 |
| Compute units | CPU + Neural Engine |

## Benchmark Results

Tested on Apple Silicon (M1):

| Audio | Predicted | Confidence | Inference Time |
|-------|-----------|------------|----------------|
| Russian (10s) | rus | 96.1% | ~7s |
| English (30s) | eng | 99.1% | ~34s |

## Usage (Swift)

```swift
import CoreML

let compiledURL = try MLModel.compileModel(at: URL(fileURLWithPath: "MmsLid256.mlpackage"))
let config = MLModelConfiguration()
config.computeUnits = .all
let model = try MLModel(contentsOf: compiledURL, configuration: config)

// pcm: [Float] — 16kHz mono audio samples (max 480000 = 30s)
let inputArray = try MLMultiArray(shape: [1, NSNumber(value: pcm.count)], dataType: .float32)
for (i, sample) in pcm.enumerated() {
    inputArray[[0, NSNumber(value: i)]] = NSNumber(value: sample)
}

let input = try MLDictionaryFeatureProvider(dictionary: [
    "input_values": MLFeatureValue(multiArray: inputArray),
])
let output = try model.prediction(from: input)
```

## Files

- `MmsLid256.mlpackage/` — CoreML model
- `mms_lid_256_labels.json` — language label mapping (index → ISO 639-3 code)

## Conversion

Converted via `torch.jit.trace` → `coremltools 9.0`. See [conversion script](https://github.com/beshkenadze/lid-bench).

## Full Inference Code

Complete Swift CLI with audio loading, inference, and result formatting:
**[github.com/beshkenadze/lid-bench](https://github.com/beshkenadze/lid-bench)**

## License

CC-BY-NC 4.0 (same as the original model)
