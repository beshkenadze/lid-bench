# LID-Bench: Spoken Language Identification on Apple Silicon

CoreML benchmarks for two spoken language identification models running natively on macOS/iOS via Swift. Zero Python dependencies at runtime.

## Models

| Model | HF Repo | Languages | Input | Size | Inference (M1) |
|-------|---------|-----------|-------|------|-----------------|
| **MMS-LID-256** | [beshkenadze/mms-lid-256-coreml](https://huggingface.co/beshkenadze/mms-lid-256-coreml) | 256 | Raw waveform 16kHz | 1.8 GB | ~7s (10s audio) |
| **ECAPA-TDNN** | [beshkenadze/lang-id-voxlingua107-ecapa-coreml](https://huggingface.co/beshkenadze/lang-id-voxlingua107-ecapa-coreml) | 107 | Log-mel spectrogram | 81 MB | ~0.1s (10s audio) |

Based on [facebook/mms-lid-256](https://huggingface.co/facebook/mms-lid-256) and [speechbrain/lang-id-voxlingua107-ecapa](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa).

## Results

Tested on Apple Silicon (M1, Neural Engine + CPU):

| Model | Russian (10s) | English (30s) |
|-------|---------------|---------------|
| MMS-LID-256 | 96.1% | 99.1% |
| ECAPA-TDNN | 99.7% | 98.6% |

ECAPA-TDNN is **50-150x faster** than MMS-LID with comparable or better accuracy.

## Requirements

- macOS 14+ (Sonoma)
- Xcode 16+ / Swift 6.0
- No external Swift dependencies

## Quick Start

```bash
# Download models from Hugging Face
hf download beshkenadze/mms-lid-256-coreml --local-dir models/MmsLid256
hf download beshkenadze/lang-id-voxlingua107-ecapa-coreml --local-dir models/EcapaTdnn

# Build
swift build -c release

# Run on audio file (both models)
.build/release/LIDBench path/to/audio.wav

# Run specific model
.build/release/LIDBench path/to/audio.wav mms
.build/release/LIDBench path/to/audio.wav ecapa
```

Models are loaded from `./models/` relative to the working directory. Override with:

```bash
LID_MODELS_DIR=/path/to/models .build/release/LIDBench audio.wav
```

## Project Structure

```
lid-bench/
├── Package.swift                    # SPM package (macOS 14+)
├── Sources/LIDBench/Main.swift      # Swift CLI (~430 lines)
└── convert/
    ├── convert_mms_lid.py           # MMS-LID PyTorch → CoreML
    ├── convert_ecapa_tdnn.py        # ECAPA-TDNN PyTorch → CoreML
    └── pyproject.toml               # Python deps for conversion
```

## Model Conversion

To reconvert from PyTorch (GPU recommended):

```bash
cd convert
uv venv && uv pip install -e .

# MMS-LID-256 (Wav2Vec2-based, ~1.8 GB)
python convert_mms_lid.py

# ECAPA-TDNN VoxLingua107 (~81 MB)
python convert_ecapa_tdnn.py
```

## Technical Notes

### MMS-LID-256

- Wav2Vec2-based encoder from Meta's Massively Multilingual Speech project
- Input: raw waveform `[1, N]` at 16kHz (max 30s / 480k samples)
- CoreML conversion via `torch.jit.trace` → `coremltools`
- FP16 compute precision

### ECAPA-TDNN VoxLingua107

- ECAPA-TDNN speaker/language embedding model from SpeechBrain
- Input: log-mel spectrogram `[1, T, 60]` computed on-device
- Mel spectrogram computed in Swift using Accelerate framework (vDSP + BLAS)
- FP32 compute precision

#### Mel Spectrogram (SpeechBrain-compatible)

| Parameter | Value |
|-----------|-------|
| Sample rate | 16000 Hz |
| n_fft | 400 |
| hop_length | 160 |
| win_length | 400 |
| n_mels | 60 |
| Window | Hamming (periodic) |
| Filterbank | SpeechBrain symmetric triangular |
| Log scale | `10 * log10(clamp(x, 1e-10))` |
| Dynamic range | top_db=80 (per-sequence) |

**Implementation pitfalls:**

1. `vDSP_fft_zrip` only supports power-of-2 sizes — n_fft=400 requires manual DFT via `cblas_sgemv`.
2. SpeechBrain uses symmetric triangular mel filters (differs from HTK's asymmetric).
3. PyTorch `hamming_window` is periodic; `vDSP_hamm_window` is symmetric — create N+1 and take first N.

## License

Code: MIT. Models retain original licenses:
- MMS-LID-256: CC-BY-NC 4.0
- ECAPA-TDNN VoxLingua107: Apache 2.0
