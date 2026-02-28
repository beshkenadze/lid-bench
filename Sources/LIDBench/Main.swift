import AVFoundation
import Accelerate
import CoreML
import Foundation

let modelsDir: String = {
    if let env = ProcessInfo.processInfo.environment["LID_MODELS_DIR"] {
        return env
    }
    return FileManager.default.currentDirectoryPath + "/models"
}()

// MARK: - Audio Loading

func loadAudioAsPCM16kHz(path: String) throws -> [Float] {
    let url = URL(fileURLWithPath: path)
    let file = try AVAudioFile(forReading: url)

    let targetFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: 16000,
        channels: 1,
        interleaved: false
    )!

    let srcFormat = file.processingFormat
    let srcFrameCount = AVAudioFrameCount(file.length)
    let ratio = 16000.0 / srcFormat.sampleRate
    let dstFrameCount = AVAudioFrameCount(Double(srcFrameCount) * ratio)

    guard let srcBuffer = AVAudioPCMBuffer(pcmFormat: srcFormat, frameCapacity: srcFrameCount) else {
        throw NSError(domain: "LIDBench", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to create source buffer"])
    }
    try file.read(into: srcBuffer)

    guard let converter = AVAudioConverter(from: srcFormat, to: targetFormat) else {
        throw NSError(domain: "LIDBench", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to create converter"])
    }

    guard let dstBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: dstFrameCount + 1024) else {
        throw NSError(domain: "LIDBench", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to create dest buffer"])
    }

    var error: NSError?
    let status = converter.convert(to: dstBuffer, error: &error) { _, outStatus in
        outStatus.pointee = .haveData
        return srcBuffer
    }
    if let error { throw error }
    guard status != .error else {
        throw NSError(domain: "LIDBench", code: 4, userInfo: [NSLocalizedDescriptionKey: "Conversion failed"])
    }

    let count = Int(dstBuffer.frameLength)
    guard let channelData = dstBuffer.floatChannelData else {
        throw NSError(domain: "LIDBench", code: 5, userInfo: [NSLocalizedDescriptionKey: "No channel data"])
    }
    return Array(UnsafeBufferPointer(start: channelData[0], count: count))
}

// MARK: - MMS-LID Inference (raw waveform → logits)

func runMmsLid(audioPath: String) throws {
    print("\n========== MMS-LID-256 ==========")
    print("Audio: \(URL(fileURLWithPath: audioPath).lastPathComponent)")

    // Load audio
    print("Loading audio...")
    var pcm = try loadAudioAsPCM16kHz(path: audioPath)
    // Truncate to 30s max (model supports up to 480000 samples)
    let maxSamples = 480000
    if pcm.count > maxSamples { pcm = Array(pcm.prefix(maxSamples)) }
    print("  Samples: \(pcm.count) (\(String(format: "%.1f", Double(pcm.count) / 16000.0))s at 16kHz)")

    // Load model
    let mlpackagePath = "\(modelsDir)/MmsLid256.mlpackage"
    guard FileManager.default.fileExists(atPath: mlpackagePath) else {
        print("  ❌ Model not found: \(mlpackagePath)")
        return
    }

    print("Compiling model...")
    let compileStart = Date()
    let compiledURL = try MLModel.compileModel(at: URL(fileURLWithPath: mlpackagePath))
    let config = MLModelConfiguration()
    config.computeUnits = .cpuAndGPU
    let model = try MLModel(contentsOf: compiledURL, configuration: config)
    let compileTime = Date().timeIntervalSince(compileStart)
    print("  Model compiled+loaded in \(String(format: "%.2f", compileTime))s")

    // Build input
    let inputArray = try MLMultiArray(shape: [1, NSNumber(value: pcm.count)], dataType: .float32)
    for (i, sample) in pcm.enumerated() {
        inputArray[[0, NSNumber(value: i)]] = NSNumber(value: sample)
    }

    let input = try MLDictionaryFeatureProvider(dictionary: [
        "input_values": MLFeatureValue(multiArray: inputArray),
    ])

    // Predict
    print("Running inference...")
    let inferStart = Date()
    let output = try model.prediction(from: input)
    let inferTime = Date().timeIntervalSince(inferStart)

    guard let outputName = output.featureNames.first,
          let logits = output.featureValue(for: outputName)?.multiArrayValue else {
        let available = output.featureNames.joined(separator: ", ")
        print("  \u{274C} No output tensor. Available: \(available)")
        return
    }

    // Load labels
    let labelsPath = "\(modelsDir)/mms_lid_256_labels.json"
    let labelsData = try Data(contentsOf: URL(fileURLWithPath: labelsPath))
    let labels = try JSONDecoder().decode([String: String].self, from: labelsData)

    // Softmax + top-5
    let numClasses = logits.shape.last!.intValue
    var rawLogits: [Float] = []
    for i in 0..<numClasses {
        rawLogits.append(logits[[0, NSNumber(value: i)]].floatValue)
    }

    let maxLogit = rawLogits.max()!
    let expLogits = rawLogits.map { exp($0 - maxLogit) }
    let sumExp = expLogits.reduce(0, +)
    let probs = expLogits.map { $0 / sumExp }

    let sorted = probs.enumerated().sorted { $0.element > $1.element }

    print("\n--- RESULT ---")
    print("PROCESSING_TIME: \(String(format: "%.3f", inferTime))s")
    print("TOP-5 predictions:")
    for i in 0..<min(5, sorted.count) {
        let idx = sorted[i].offset
        let prob = sorted[i].element
        let lang = labels[String(idx)] ?? "unknown"
        let marker = i == 0 ? " ← PREDICTED" : ""
        print("  \(i + 1). \(lang) (\(String(format: "%.1f", prob * 100))%)\(marker)")
    }
}

// MARK: - ECAPA-TDNN Inference (mel-spectrogram → log_probs)

// CMVN stats are all zeros for this model (identity transform), so we skip CMVN normalization

func computeLogMelSpectrogram(pcm: [Float], sampleRate: Int = 16000) -> [[Float]] {
    let nFFT = 400
    let hopLength = 160
    let nMels = 60
    let fMin: Float = 0
    let fMax: Float = Float(sampleRate) / 2.0
    let amin: Float = 1e-10
    let topDb: Float = 80.0

    // Center-pad the signal (SpeechBrain STFT uses center=True)
    let padSize = nFFT / 2
    var padded = [Float](repeating: 0, count: padSize + pcm.count + padSize)
    for i in 0..<pcm.count { padded[padSize + i] = pcm[i] }

    // Hamming window (periodic, matching PyTorch's torch.hamming_window default)
    let windowSize = nFFT
    var periodicBuf = [Float](repeating: 0, count: windowSize + 1)
    vDSP_hamm_window(&periodicBuf, vDSP_Length(windowSize + 1), 0)
    let window = Array(periodicBuf.prefix(windowSize))

    let numFrames = max(0, (padded.count - nFFT) / hopLength + 1)
    if numFrames == 0 { return [] }

    let nFreqBins = nFFT / 2 + 1

    // Create mel filterbank
    let melFilters = createMelFilterbank(
        nMels: nMels, nFFT: nFFT, sampleRate: sampleRate, fMin: fMin, fMax: fMax
    )

    // Precompute DFT twiddle factors for n_fft=400 → 201 frequency bins.
    // vDSP_fft_zrip requires power-of-2 sizes (400 is not), and vDSP_DFT_zop
    // doesn't support 400. Use matrix multiply: X_re = cos_mat @ x, X_im = -sin_mat @ x
    // Accelerated by cblas_sgemv.
    var cosTable = [Float](repeating: 0, count: nFreqBins * nFFT)  // [201 x 400] row-major
    var sinTable = [Float](repeating: 0, count: nFreqBins * nFFT)
    for k in 0..<nFreqBins {
        for n in 0..<nFFT {
            let angle = 2.0 * Float.pi * Float(k) * Float(n) / Float(nFFT)
            cosTable[k * nFFT + n] = cos(angle)
            sinTable[k * nFFT + n] = sin(angle)
        }
    }

    var frames: [[Float]] = []
    var dftReal = [Float](repeating: 0, count: nFreqBins)
    var dftImag = [Float](repeating: 0, count: nFreqBins)

    for frameIdx in 0..<numFrames {
        let start = frameIdx * hopLength
        var windowed = [Float](repeating: 0, count: nFFT)
        for i in 0..<nFFT {
            if start + i < padded.count {
                windowed[i] = padded[start + i] * window[i]
            }
        }

        // DFT via matrix-vector multiply (cblas_sgemv): O(N*K) per frame
        // dftReal = cosTable @ windowed  →  X_re[k] = Σ x[n]*cos(2πkn/N)
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    Int32(nFreqBins), Int32(nFFT),
                    1.0, &cosTable, Int32(nFFT),
                    &windowed, 1,
                    0.0, &dftReal, 1)
        // dftImag = -sinTable @ windowed  →  X_im[k] = -Σ x[n]*sin(2πkn/N)
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    Int32(nFreqBins), Int32(nFFT),
                    -1.0, &sinTable, Int32(nFFT),
                    &windowed, 1,
                    0.0, &dftImag, 1)

        // Power spectrum: |X[k]|^2 = re^2 + im^2
        var powerSpec = [Float](repeating: 0, count: nFreqBins)
        vDSP_vsq(&dftReal, 1, &powerSpec, 1, vDSP_Length(nFreqBins))
        var imSq = [Float](repeating: 0, count: nFreqBins)
        vDSP_vsq(&dftImag, 1, &imSq, 1, vDSP_Length(nFreqBins))
        vDSP_vadd(powerSpec, 1, imSq, 1, &powerSpec, 1, vDSP_Length(nFreqBins))

        // Apply mel filterbank to power spectrum + log dB (matching SpeechBrain)
        // SpeechBrain: spectral_magnitude(power=1) = re^2+im^2, then matmul(power, filters), then 10*log10
        var melEnergies = [Float](repeating: 0, count: nMels)
        for m in 0..<nMels {
            var energy: Float = 0
            for k in 0..<nFreqBins {
                energy += melFilters[m][k] * powerSpec[k]
            }
            melEnergies[m] = 10.0 * log10(max(energy, amin))
        }

        frames.append(melEnergies)
    }

    // Apply top_db clipping: clamp to [max - topDb, max]
    var globalMax: Float = -Float.infinity
    for frame in frames {
        for val in frame {
            if val > globalMax { globalMax = val }
        }
    }
    let threshold = globalMax - topDb
    for i in 0..<frames.count {
        for j in 0..<frames[i].count {
            if frames[i][j] < threshold { frames[i][j] = threshold }
        }
    }

    return frames
}

func hzToMel(_ hz: Float) -> Float {
    return 2595.0 * log10(1.0 + hz / 700.0)
}

func melToHz(_ mel: Float) -> Float {
    return 700.0 * (pow(10.0, mel / 2595.0) - 1.0)
}

func createMelFilterbank(nMels: Int, nFFT: Int, sampleRate: Int, fMin: Float, fMax: Float) -> [[Float]] {
    let nFreqBins = nFFT / 2 + 1
    let melMin = hzToMel(fMin)
    let melMax = hzToMel(fMax)

    // Compute mel-spaced Hz points (n_mels + 2 points including edges)
    var melPoints = [Float](repeating: 0, count: nMels + 2)
    for i in 0..<(nMels + 2) {
        melPoints[i] = melToHz(melMin + Float(i) * (melMax - melMin) / Float(nMels + 1))
    }

    // FFT frequency bins
    var fftFreqs = [Float](repeating: 0, count: nFreqBins)
    for i in 0..<nFreqBins {
        fftFreqs[i] = Float(sampleRate) * Float(i) / Float(nFFT)
    }

    // SpeechBrain-style symmetric triangular filters:
    //   band[m] = hz[m+1] - hz[m]  (distance to previous mel point)
    //   slope = (freq - f_central) / band
    //   value = max(0, min(slope + 1, -slope + 1))
    // This differs from HTK which uses different widths on left/right sides.
    var filterbank = [[Float]](repeating: [Float](repeating: 0, count: nFreqBins), count: nMels)
    for m in 0..<nMels {
        let fCenter = melPoints[m + 1]
        let band = melPoints[m + 1] - melPoints[m]
        guard band > 0 else { continue }
        for k in 0..<nFreqBins {
            let freq = fftFreqs[k]
            let slope = (freq - fCenter) / band
            let leftSide = slope + 1.0
            let rightSide = -slope + 1.0
            filterbank[m][k] = max(0.0, min(leftSide, rightSide))
        }
    }

    return filterbank
}

func runEcapaTdnn(audioPath: String) throws {
    print("\n========== ECAPA-TDNN VoxLingua107 ==========")
    print("Audio: \(URL(fileURLWithPath: audioPath).lastPathComponent)")

    // Load audio
    print("Loading audio...")
    let pcm = try loadAudioAsPCM16kHz(path: audioPath)
    print("  Samples: \(pcm.count) (\(String(format: "%.1f", Double(pcm.count) / 16000.0))s at 16kHz)")

    // Load model
    let mlpackagePath = "\(modelsDir)/EcapaTdnnLid107.mlpackage"
    guard FileManager.default.fileExists(atPath: mlpackagePath) else {
        print("  ❌ Model not found: \(mlpackagePath)")
        return
    }

    // Compute mel spectrogram (n_mels=60 for ECAPA-TDNN)
    print("Computing mel spectrogram...")
    let melStart = Date()
    var melFrames = computeLogMelSpectrogram(pcm: pcm)
    let melTime = Date().timeIntervalSince(melStart)
    // Truncate to max 3000 frames (~30s)
    if melFrames.count > 3000 { melFrames = Array(melFrames.prefix(3000)) }
    print("  Mel frames: \(melFrames.count) x 60 (\(String(format: "%.3f", melTime))s)")

    // Compile model
    print("Compiling model...")
    let compileStart = Date()
    let compiledURL = try MLModel.compileModel(at: URL(fileURLWithPath: mlpackagePath))
    let config = MLModelConfiguration()
    config.computeUnits = .cpuAndGPU
    let model = try MLModel(contentsOf: compiledURL, configuration: config)
    let compileTime = Date().timeIntervalSince(compileStart)
    print("  Model compiled+loaded in \(String(format: "%.2f", compileTime))s")

    // Build input: [1, T, 60]
    let T = melFrames.count
    let inputArray = try MLMultiArray(
        shape: [1, NSNumber(value: T), 60],
        dataType: .float32
    )
    for t in 0..<T {
        for f in 0..<60 {
            inputArray[[0, NSNumber(value: t), NSNumber(value: f)]] = NSNumber(value: melFrames[t][f])
        }
    }

    let input = try MLDictionaryFeatureProvider(dictionary: [
        "mel_features": MLFeatureValue(multiArray: inputArray),
    ])

    // Predict
    print("Running inference...")
    let inferStart = Date()
    let output = try model.prediction(from: input)
    let inferTime = Date().timeIntervalSince(inferStart)

    guard let outputName = output.featureNames.first,
          let logProbs = output.featureValue(for: outputName)?.multiArrayValue else {
        let available = output.featureNames.joined(separator: ", ")
        print("  \u{274C} No output tensor. Available: \(available)")
        return
    }

    // Load labels
    let labelsPath = "\(modelsDir)/ecapa_tdnn_lid107_labels.json"
    let labelsData = try Data(contentsOf: URL(fileURLWithPath: labelsPath))
    let labels = try JSONDecoder().decode([String: String].self, from: labelsData)

    // Top-5 (log_probs → exp → softmax not needed, just argmax on log_probs)
    let numClasses = logProbs.shape.last!.intValue
    var rawProbs: [Float] = []
    for i in 0..<numClasses {
        rawProbs.append(logProbs[[0, NSNumber(value: i)]].floatValue)
    }

    // Apply softmax on log-probs for readable percentages
    let maxVal = rawProbs.max()!
    let expVals = rawProbs.map { exp($0 - maxVal) }
    let sumExp = expVals.reduce(0, +)
    let probs = expVals.map { $0 / sumExp }

    let sorted = probs.enumerated().sorted { $0.element > $1.element }

    print("\n--- RESULT ---")
    print("PROCESSING_TIME: \(String(format: "%.3f", inferTime))s")
    print("MEL_TIME: \(String(format: "%.3f", melTime))s")
    print("TOP-5 predictions:")
    for i in 0..<min(5, sorted.count) {
        let idx = sorted[i].offset
        let prob = sorted[i].element
        let lang = labels[String(idx)] ?? "unknown(\(idx))"
        let marker = i == 0 ? " ← PREDICTED" : ""
        print("  \(i + 1). \(lang) (\(String(format: "%.1f", prob * 100))%)\(marker)")
    }
}



// MARK: - Main

@main
struct LIDBench {
    static func main() async throws {
        let args = CommandLine.arguments
        guard args.count >= 2 else {
            print("Usage: LIDBench <audio-path> [mms|ecapa|all|compute-test]")
            print("  audio-path: path to audio file")
            print("  model: mms, ecapa, all (default: all)")
            print("  compute-test: run compute unit diagnostic")
            Foundation.exit(1)
        }

        let audioPath = args[1]
        let modelChoice = args.count >= 3 ? args[2] : "all"

        guard FileManager.default.fileExists(atPath: audioPath) else {
            print("ERROR: File not found: \(audioPath)")
            Foundation.exit(1)
        }

        print("=== Language Identification Benchmark ===")
        print("Audio: \(audioPath)")

        do {
            if modelChoice == "compute-test" {
                try runComputeUnitTest(audioPath: audioPath)
            } else {
                if modelChoice == "mms" || modelChoice == "all" {
                    try runMmsLid(audioPath: audioPath)
                }
                if modelChoice == "ecapa" || modelChoice == "all" {
                    try runEcapaTdnn(audioPath: audioPath)
                }
            }
        } catch {
            print("\n❌ Error: \(error)")
            Foundation.exit(1)
        }

        print("\nDone.")
    }
}
