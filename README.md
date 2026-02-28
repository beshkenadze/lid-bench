# LID-Bench: Spoken Language Identification on Apple Silicon

CoreML and MLX benchmarks for spoken language identification models on Apple Silicon. Swift CLI for CoreML, Python for MLX. Both use Metal GPU.

## Models

| Model | HF Repo | Languages | Input | Size | Inference (M1) |
|-------|---------|-----------|-------|------|-----------------|
| **MMS-LID-256** | [beshkenadze/mms-lid-256-coreml](https://huggingface.co/beshkenadze/mms-lid-256-coreml) | 256 | Raw waveform 16kHz | 1.8 GB | ~0.25s (10s audio) |
| **ECAPA-TDNN** | [beshkenadze/lang-id-voxlingua107-ecapa-coreml](https://huggingface.co/beshkenadze/lang-id-voxlingua107-ecapa-coreml) | 107 | Log-mel spectrogram | 81 MB | ~0.017s (10s audio) |

Based on [facebook/mms-lid-256](https://huggingface.co/facebook/mms-lid-256) and [speechbrain/lang-id-voxlingua107-ecapa](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa).

## Results

Tested on Apple Silicon (M1, Metal GPU):

### CoreML (Swift)

| Model | Russian (10s) | English (30s) |
|-------|---------------|---------------|
| MMS-LID-256 | 89.1% (0.25s) | 99.8% (0.75s) |
| ECAPA-TDNN | 99.7% (0.017s) | 98.6% (0.05s) |

### MLX (Python)

| Model | Russian (10s) | English (30s) |
|-------|---------------|---------------|
| MMS-LID-256 | 98.8% (0.27s) | 99.8% (0.80s) |
| ECAPA-TDNN | 99.6% (0.016s) | 99.9% (0.04s) |

### CoreML vs MLX Benchmark (M1, 10s audio)

| Model | Params | CoreML GPU | MLX GPU (warm) | Ratio |
|---|---|---|---|---|
| ECAPA-TDNN | 20M | 17ms | **16.3ms** | MLX 1.04x faster |
| MMS-LID-256 | 315M | 250ms | **265ms** | CoreML 1.06x faster |

> Both frameworks use Metal GPU. Neither benefits from the Neural Engine.
> MMS-LID-256 is 13x slower with ANE enabled. Use `.cpuAndGPU`.

ECAPA-TDNN is **15x faster** than MMS-LID with comparable or better accuracy.

## Requirements

### CoreML (Swift CLI)
- macOS 14+ (Sonoma)
- Xcode 16+ / Swift 6.0

### MLX (Python)
- macOS 14+ (Apple Silicon)
- Python 3.12+
- MLX 0.31+
## Quick Start

### CoreML

```bash
# Download models from Hugging Face
hf download beshkenadze/mms-lid-256-coreml --local-dir models/MmsLid256
hf download beshkenadze/lang-id-voxlingua107-ecapa-coreml --local-dir models/EcapaTdnn

# Build & run
swift build -c release
.build/release/LIDBench path/to/audio.wav
```

### MLX

```bash
cd mlx
uv venv && uv pip install mlx numpy soundfile safetensors

# ECAPA-TDNN (107 languages, needs converted weights)
python ecapa_tdnn_lid.py path/to/audio.wav --benchmark

# MMS-LID-256 (256 languages, loads from HF cache)
# First: pip install transformers && python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/mms-lid-256')"
python mms_lid_256.py path/to/audio.wav --benchmark
```

## Project Structure

```
lid-bench/
├── Package.swift                    # SPM package (macOS 14+)
├── Sources/LIDBench/Main.swift      # Swift CLI (~430 lines)
├── convert/
│   ├── convert_mms_lid.py           # MMS-LID PyTorch → CoreML
│   ├── convert_ecapa_tdnn.py        # ECAPA-TDNN PyTorch → CoreML
│   └── pyproject.toml               # Python deps for conversion
├── mlx/
│   ├── ecapa_tdnn_lid.py            # ECAPA-TDNN in pure MLX (515 lines)
│   ├── mms_lid_256.py               # MMS-LID-256 in pure MLX (535 lines)
│   ├── convert_ecapa_weights.py     # SpeechBrain → safetensors converter
│   └── weights/                     # Converted MLX weights
└── docs/
    └── research_mlx_lid_2026-02-28.md  # MLX vs CoreML research report
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
- **Runs on Metal GPU only** — ANE causes 13x slowdown due to op splitting overhead. Use `.cpuAndGPU`.

### ECAPA-TDNN VoxLingua107

- ECAPA-TDNN speaker/language embedding model from SpeechBrain
- Input: log-mel spectrogram `[1, T, 60]` computed on-device
- Mel spectrogram computed in Swift using Accelerate framework (vDSP + BLAS)
- FP32 compute precision
- **Runs on Metal GPU** — ANE is not engaged regardless of compute unit setting

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
