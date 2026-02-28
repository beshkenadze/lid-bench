import CoreML
import Foundation
import AVFoundation

/// Quick diagnostic: run each model with different compute units to determine
/// whether ANE is actually being used or CoreML falls back to CPU/GPU.
///
/// Usage: LIDBench --compute-test <audio-path>
func runComputeUnitTest(audioPath: String) throws {
    print("=== Compute Unit Diagnostic ===\n")

    let pcm = try loadAudioAsPCM16kHz(path: audioPath)
    print("Audio loaded: \(pcm.count) samples (\(String(format: "%.1f", Double(pcm.count) / 16000.0))s)\n")

    let configs: [(String, MLComputeUnits)] = [
        ("cpuOnly",        .cpuOnly),
        ("cpuAndGPU",      .cpuAndGPU),
        ("all (ANE+GPU+CPU)", .all),
    ]

    // --- MMS-LID-256 ---
    let mmsPath = modelsDir + "/MmsLid256.mlpackage"
    if FileManager.default.fileExists(atPath: mmsPath) {
        print("── MMS-LID-256 ──")
        for (name, units) in configs {
            let config = MLModelConfiguration()
            config.computeUnits = units
            do {
                let compiledURL = try MLModel.compileModel(at: URL(fileURLWithPath: mmsPath))
                let model = try MLModel(contentsOf: compiledURL, configuration: config)

                let inputArray = try MLMultiArray(shape: [1, NSNumber(value: pcm.count)], dataType: .float32)
                for (i, sample) in pcm.enumerated() {
                    inputArray[[0, NSNumber(value: i)]] = NSNumber(value: sample)
                }
                let input = try MLDictionaryFeatureProvider(dictionary: [
                    "input_values": MLFeatureValue(multiArray: inputArray),
                ])

                // Warmup
                _ = try model.prediction(from: input)

                // Timed run (3 iterations)
                var times: [Double] = []
                for _ in 0..<3 {
                    let start = Date()
                    _ = try model.prediction(from: input)
                    times.append(Date().timeIntervalSince(start))
                }
                let avg = times.reduce(0, +) / Double(times.count)
                let timesStr = times.map { String(format: "%.3f", $0) }.joined(separator: ", ")
                print("  \(name.padding(toLength: 22, withPad: " ", startingAt: 0)) avg=\(String(format: "%.3f", avg))s  runs=[\(timesStr)]")

                try? FileManager.default.removeItem(at: compiledURL)
            } catch {
                print("  \(name.padding(toLength: 22, withPad: " ", startingAt: 0)) FAILED: \(error.localizedDescription)")
            }
        }
        print()
    } else {
        print("MMS-LID-256 not found at \(mmsPath), skipping\n")
    }

    // --- ECAPA-TDNN ---
    let ecapaPath = modelsDir + "/EcapaTdnnLid107.mlpackage"
    if FileManager.default.fileExists(atPath: ecapaPath) {
        print("── ECAPA-TDNN ──")

        let melFrames = computeLogMelSpectrogram(pcm: pcm)
        let melFeatures = melFrames.flatMap { $0 }
        let frameCount = melFrames.count

        for (name, units) in configs {
            let config = MLModelConfiguration()
            config.computeUnits = units
            do {
                let compiledURL = try MLModel.compileModel(at: URL(fileURLWithPath: ecapaPath))
                let model = try MLModel(contentsOf: compiledURL, configuration: config)

                let inputArray = try MLMultiArray(shape: [1, NSNumber(value: frameCount), 60], dataType: .float32)
                for i in 0..<melFeatures.count {
                    let t = i / 60
                    let m = i % 60
                    inputArray[[0, NSNumber(value: t), NSNumber(value: m)]] = NSNumber(value: melFeatures[i])
                }
                let input = try MLDictionaryFeatureProvider(dictionary: [
                    "mel_features": MLFeatureValue(multiArray: inputArray),
                ])

                // Warmup
                _ = try model.prediction(from: input)

                // Timed run (10 iterations for small model)
                var times: [Double] = []
                for _ in 0..<10 {
                    let start = Date()
                    _ = try model.prediction(from: input)
                    times.append(Date().timeIntervalSince(start))
                }
                let avg = times.reduce(0, +) / Double(times.count)
                let timesStr = times.map { String(format: "%.4f", $0) }.joined(separator: ", ")
                print("  \(name.padding(toLength: 22, withPad: " ", startingAt: 0)) avg=\(String(format: "%.4f", avg))s  runs=[\(timesStr)]")

                try? FileManager.default.removeItem(at: compiledURL)
            } catch {
                print("  \(name.padding(toLength: 22, withPad: " ", startingAt: 0)) FAILED: \(error.localizedDescription)")
            }
        }
        print()
    } else {
        print("ECAPA-TDNN not found at \(ecapaPath), skipping\n")
    }

    print("""
    Interpretation:
      - If "all" ≈ "cpuAndGPU" → ANE is NOT used (GPU fallback)
      - If "all" < "cpuAndGPU" significantly → ANE is engaged
      - If "all" ≈ "cpuOnly" → everything falls back to CPU
    """)
}
