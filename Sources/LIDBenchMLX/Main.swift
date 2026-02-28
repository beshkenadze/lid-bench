@preconcurrency import AVFoundation
import Foundation
import MLX
import MLXNN

// MARK: - Audio Loading

func loadAudio(path: String, sampleRate: Int = 16000) throws -> [Float] {
    let url = URL(fileURLWithPath: path)
    let file = try AVAudioFile(forReading: url)

    let targetFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: Double(sampleRate),
        channels: 1,
        interleaved: false
    )!

    let srcFormat = file.processingFormat
    let srcFrameCount = AVAudioFrameCount(file.length)
    let ratio = Double(sampleRate) / srcFormat.sampleRate
    let dstFrameCount = AVAudioFrameCount(Double(srcFrameCount) * ratio)

    guard let srcBuffer = AVAudioPCMBuffer(pcmFormat: srcFormat, frameCapacity: srcFrameCount) else {
        throw NSError(domain: "LIDBenchMLX", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to create source buffer"])
    }
    try file.read(into: srcBuffer)

    guard let converter = AVAudioConverter(from: srcFormat, to: targetFormat) else {
        throw NSError(domain: "LIDBenchMLX", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to create converter"])
    }

    guard let dstBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: dstFrameCount + 1024) else {
        throw NSError(domain: "LIDBenchMLX", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to create dest buffer"])
    }

    var error: NSError?
    let status = converter.convert(to: dstBuffer, error: &error) { _, outStatus in
        outStatus.pointee = .haveData
        return srcBuffer
    }
    if let error { throw error }
    guard status != .error else {
        throw NSError(domain: "LIDBenchMLX", code: 4, userInfo: [NSLocalizedDescriptionKey: "Conversion failed"])
    }

    let count = Int(dstBuffer.frameLength)
    guard let channelData = dstBuffer.floatChannelData else {
        throw NSError(domain: "LIDBenchMLX", code: 5, userInfo: [NSLocalizedDescriptionKey: "No channel data"])
    }
    return Array(UnsafeBufferPointer(start: channelData[0], count: count))
}

// MARK: - Prediction & Benchmark

func predict(logProbs: MLXArray, labels: [String: String], topK: Int = 5) -> [(String, Float)] {
    let probs = exp(logProbs)
    eval(probs)
    let probsFlat = probs.squeezed(axis: 0) // (numClasses,)
    let numClasses = probsFlat.dim(0)

    var indexed = [(Int, Float)]()
    for i in 0..<numClasses {
        indexed.append((i, probsFlat[i].item(Float.self)))
    }
    indexed.sort { $0.1 > $1.1 }

    return indexed.prefix(topK).map { (labels[String($0.0)] ?? "?\($0.0)", $0.1) }
}

func benchmarkModel(
    _ runInference: () -> MLXArray,
    warmup: Int = 3, runs: Int = 10
) -> (meanMs: Double, stdMs: Double, minMs: Double, maxMs: Double) {
    for _ in 0..<warmup {
        eval(runInference())
    }

    var times = [Double]()
    for _ in 0..<runs {
        let t0 = CFAbsoluteTimeGetCurrent()
        eval(runInference())
        let t1 = CFAbsoluteTimeGetCurrent()
        times.append((t1 - t0) * 1000)
    }

    let mean = times.reduce(0, +) / Double(times.count)
    let variance = times.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(times.count)
    let std = sqrt(variance)
    return (mean, std, times.min()!, times.max()!)
}

// MARK: - MMS-LID-256

func runMmsLid256(audioPath: String, modelsDir: String, doBenchmark: Bool) throws {
    print("\n========== MLX MMS-LID-256 ==========")

    // Resolve HF cached model
    let hfModelDir = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent(".cache/huggingface/hub/models--facebook--mms-lid-256/snapshots/edc73fd00996e671dfc59d16436a29b12b10588a")
    let labelsPath = URL(fileURLWithPath: modelsDir).appendingPathComponent("mms_lid_256_labels.json")

    guard FileManager.default.fileExists(atPath: hfModelDir.path) else {
        print("  Model not found: \(hfModelDir.path)")
        print("  Run: pip install transformers && python -c \"from transformers import AutoModel; AutoModel.from_pretrained('facebook/mms-lid-256')\"")
        return
    }

    print("Loading audio: \(URL(fileURLWithPath: audioPath).lastPathComponent)")
    let audio = try loadAudio(path: audioPath)
    print("  Duration: \(String(format: "%.1f", Double(audio.count) / 16000.0))s, samples: \(audio.count)")

    print("Loading model...")
    let (model, labels) = try loadMmsLid256(modelDir: hfModelDir, labelsPath: labelsPath)
    print("  Loaded \(labels.count) languages")

    // Truncate to 30s
    var pcm = audio
    let maxSamples = 480000
    if pcm.count > maxSamples { pcm = Array(pcm.prefix(maxSamples)) }

    let waveform = normalizeWaveform(pcm)

    print("Running inference...")
    let logProbs = model(waveform)
    eval(logProbs)

    let results = predict(logProbs: logProbs, labels: labels)
    print("\nTop-5 predictions:")
    for (i, (lang, prob)) in results.enumerated() {
        let marker = i == 0 ? " <- PREDICTED" : ""
        print("  \(i + 1). \(lang): \(String(format: "%.1f", prob * 100))%\(marker)")
    }

    if doBenchmark {
        print("\nBenchmark (10 runs, 3 warmup):")
        let stats = benchmarkModel({ model(waveform) })
        print("  Mean: \(String(format: "%.1f", stats.meanMs))ms +/- \(String(format: "%.1f", stats.stdMs))ms")
        print("  Min:  \(String(format: "%.1f", stats.minMs))ms")
        print("  Max:  \(String(format: "%.1f", stats.maxMs))ms")
        let duration = Double(pcm.count) / 16000.0
        print("  Audio: \(String(format: "%.1f", duration))s")
        print("  RTF:  \(String(format: "%.4f", stats.meanMs / (duration * 1000)))")
    }
}

// MARK: - ECAPA-TDNN

func runEcapaTdnn(audioPath: String, modelsDir: String, weightsDir: String, doBenchmark: Bool) throws {
    print("\n========== MLX ECAPA-TDNN VoxLingua107 ==========")

    let weightsPath = URL(fileURLWithPath: weightsDir).appendingPathComponent("ecapa_tdnn_lid107.safetensors")
    let labelsPath = URL(fileURLWithPath: modelsDir).appendingPathComponent("ecapa_tdnn_lid107_labels.json")

    guard FileManager.default.fileExists(atPath: weightsPath.path) else {
        print("  Weights not found: \(weightsPath.path)")
        return
    }

    print("Loading audio: \(URL(fileURLWithPath: audioPath).lastPathComponent)")
    let audio = try loadAudio(path: audioPath)
    print("  Duration: \(String(format: "%.1f", Double(audio.count) / 16000.0))s, samples: \(audio.count)")

    print("Computing mel spectrogram...")
    let mel = computeMelSpectrogram(audio: audio)
    print("  Mel shape: [1, \(mel.dim(1)), \(mel.dim(2))]")

    print("Loading model...")
    let (model, labels) = try loadEcapaTdnnLid(weightsPath: weightsPath, labelsPath: labelsPath)
    print("  Loaded \(labels.count) languages")

    print("Running inference...")
    let logProbs = model(mel)
    eval(logProbs)

    let results = predict(logProbs: logProbs, labels: labels)
    print("\nTop-5 predictions:")
    for (i, (lang, prob)) in results.enumerated() {
        let marker = i == 0 ? " <- PREDICTED" : ""
        print("  \(i + 1). \(lang): \(String(format: "%.1f", prob * 100))%\(marker)")
    }

    if doBenchmark {
        print("\nBenchmark (10 runs, 3 warmup):")
        let stats = benchmarkModel({ model(mel) })
        print("  Mean: \(String(format: "%.1f", stats.meanMs))ms +/- \(String(format: "%.1f", stats.stdMs))ms")
        print("  Min:  \(String(format: "%.1f", stats.minMs))ms")
        print("  Max:  \(String(format: "%.1f", stats.maxMs))ms")
        let duration = Double(audio.count) / 16000.0
        print("  Audio: \(String(format: "%.1f", duration))s (\(mel.dim(1)) frames)")
        print("  RTF:  \(String(format: "%.4f", stats.meanMs / (duration * 1000)))")
    }
}

// MARK: - Main

@main
struct LIDBenchMLX {
    static func main() throws {
        let args = CommandLine.arguments
        guard args.count >= 2 else {
            print("Usage: LIDBenchMLX <audio-path> [mms|ecapa|all] [--benchmark]")
            print("  audio-path: path to audio file (WAV/MP3)")
            print("  model: mms, ecapa, all (default: all)")
            print("  --benchmark: run timing benchmark")
            Foundation.exit(1)
        }

        let audioPath = args[1]
        let modelChoice = args.count >= 3 && !args[2].hasPrefix("-") ? args[2] : "all"
        let doBenchmark = args.contains("--benchmark")

        guard FileManager.default.fileExists(atPath: audioPath) else {
            print("ERROR: File not found: \(audioPath)")
            Foundation.exit(1)
        }

        // Resolve paths relative to project
        let projectDir: String
        if let env = ProcessInfo.processInfo.environment["LID_PROJECT_DIR"] {
            projectDir = env
        } else {
            // Walk up from executable to find project root
            var dir = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent()
            for _ in 0..<10 {
                if FileManager.default.fileExists(atPath: dir.appendingPathComponent("Package.swift").path) {
                    break
                }
                dir = dir.deletingLastPathComponent()
            }
            projectDir = dir.path
        }

        let modelsDir = ProcessInfo.processInfo.environment["LID_MODELS_DIR"]
            ?? projectDir + "/models"
        let weightsDir = ProcessInfo.processInfo.environment["LID_WEIGHTS_DIR"]
            ?? projectDir + "/mlx/weights"

        print("=== MLX Language Identification Benchmark ===")
        print("MLX device: \(Device.defaultDevice())")
        print("Audio: \(audioPath)")
        do {
            if modelChoice == "mms" || modelChoice == "all" {
                try runMmsLid256(audioPath: audioPath, modelsDir: modelsDir, doBenchmark: doBenchmark)
            }
            if modelChoice == "ecapa" || modelChoice == "all" {
                try runEcapaTdnn(audioPath: audioPath, modelsDir: modelsDir, weightsDir: weightsDir, doBenchmark: doBenchmark)
            }
        } catch {
            print("\nERROR: \(error)")
            Foundation.exit(1)
        }

        print("\nDone.")
    }
}
