import Foundation
import MLX
import MLXNN

// MARK: - Mel Spectrogram (SpeechBrain-compatible)

let kSampleRate: Int = 16000
let kNfft: Int = 400
let kHopLength: Int = 160
let kWinLength: Int = 400
let kNMels: Int = 60

func speechbrainMelFilterbank(sampleRate: Int, nfft: Int, nMels: Int) -> [[Float]] {
    func hzToMel(_ f: Float) -> Float { 2595.0 * log10(1.0 + f / 700.0) }
    func melToHz(_ m: Float) -> Float { 700.0 * (pow(10.0, m / 2595.0) - 1.0) }

    let lowMel = hzToMel(0.0)
    let highMel = hzToMel(Float(sampleRate) / 2.0)
    let melPoints = (0..<(nMels + 2)).map { i in
        melToHz(lowMel + Float(i) * (highMel - lowMel) / Float(nMels + 1))
    }
    let fftBins = (0..<(nfft / 2 + 1)).map { Float(sampleRate) * Float($0) / Float(nfft) }

    var filterbank = [[Float]](repeating: [Float](repeating: 0, count: nfft / 2 + 1), count: nMels)
    for m in 0..<nMels {
        let fLeft = melPoints[m]
        let fCenter = melPoints[m + 1]
        let fRight = melPoints[m + 2]
        let bandLeft = fCenter - fLeft
        let bandRight = fRight - fCenter
        for k in 0..<fftBins.count {
            let f = fftBins[k]
            if f >= fLeft && f <= fCenter && bandLeft > 0 {
                filterbank[m][k] = (f - fLeft) / bandLeft
            } else if f > fCenter && f <= fRight && bandRight > 0 {
                filterbank[m][k] = (fRight - f) / bandRight
            }
        }
    }
    return filterbank
}

func computeMelSpectrogram(audio: [Float]) -> MLXArray {
    // Periodic Hamming window
    let windowValues = (0..<kWinLength).map { n -> Float in
        0.54 - 0.46 * cos(2.0 * Float.pi * Float(n) / Float(kWinLength))
    }
    let window = MLXArray(windowValues)

    // Precompute mel filterbank as MLXArray [nFreqBins, nMels]
    let melFbRaw = speechbrainMelFilterbank(sampleRate: kSampleRate, nfft: kNfft, nMels: kNMels)
    let melFbT = MLXArray(melFbRaw.flatMap { $0 }).reshaped(kNMels, kNfft / 2 + 1).transposed()

    // Convert audio to MLXArray and center-pad
    let padLen = kNfft / 2
    let audioMLX = MLXArray(audio)
    let padded = concatenated([MLXArray.zeros([padLen]), audioMLX, MLXArray.zeros([padLen])])

    let totalLen = padLen + audio.count + padLen
    let numFrames = max(0, (totalLen - kNfft) / kHopLength + 1)
    if numFrames == 0 { return MLXArray.zeros([1, 0, kNMels]) }

    // Frame extraction via asStrided (zero-copy GPU view)
    let frames = asStrided(padded, [numFrames, kNfft], strides: [kHopLength, 1])

    // Apply window and compute FFT on GPU
    let fftResult = rfft(frames * window, n: kNfft, axis: -1)

    // Power spectrum: |FFT|^2
    let magnitude = abs(fftResult)
    let powerSpec = magnitude * magnitude

    // Mel filterbank: [numFrames, nFreqBins] @ [nFreqBins, nMels] -> [numFrames, nMels]
    let melSpec = matmul(powerSpec, melFbT)

    // Log scale: 10 * log10(max(x, 1e-10))
    let logMel = 10.0 * log10(maximum(melSpec, MLXArray(Float(1e-10))))

    // top_db=80 clipping
    let clipped = maximum(logMel, logMel.max() - 80.0)

    return clipped.reshaped(1, numFrames, kNMels)
}

// MARK: - ECAPA-TDNN Model Components

class TDNNBlock: Module {
    @ModuleInfo var conv: Conv1d
    @ModuleInfo var norm: BatchNorm

    init(inputChannels: Int, outputChannels: Int, kernelSize: Int, dilation: Int = 1, groups: Int = 1) {
        let padding = (kernelSize - 1) * dilation / 2
        _conv.wrappedValue = Conv1d(
            inputChannels: inputChannels, outputChannels: outputChannels,
            kernelSize: kernelSize, padding: padding, dilation: dilation,
            groups: groups, bias: true
        )
        _norm.wrappedValue = BatchNorm(featureCount: outputChannels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return norm(relu(conv(x)))
    }
}

class Res2NetBlock: Module {
    let scale: Int
    @ModuleInfo var blocks: [TDNNBlock]

    init(channels: Int, kernelSize: Int = 3, dilation: Int = 1, scale: Int = 8) {
        self.scale = scale
        let hidden = channels / scale
        _blocks.wrappedValue = (0..<(scale - 1)).map { _ in
            TDNNBlock(inputChannels: hidden, outputChannels: hidden, kernelSize: kernelSize, dilation: dilation)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let chunks = split(x, parts: scale, axis: -1)
        var y = [chunks[0]]
        for i in 0..<blocks.count {
            let inp = i > 0 ? chunks[i + 1] + y.last! : chunks[i + 1]
            y.append(blocks[i](inp))
        }
        return concatenated(y, axis: -1)
    }
}

class SEBlock: Module {
    @ModuleInfo var conv1: Conv1d
    @ModuleInfo var conv2: Conv1d

    init(inputDim: Int, bottleneck: Int = 128) {
        _conv1.wrappedValue = Conv1d(inputChannels: inputDim, outputChannels: bottleneck, kernelSize: 1)
        _conv2.wrappedValue = Conv1d(inputChannels: bottleneck, outputChannels: inputDim, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var s = mean(x, axis: 1, keepDims: true) // (B, 1, C)
        s = relu(conv1(s))
        s = sigmoid(conv2(s))
        return x * s
    }
}

class SERes2NetBlock: Module {
    @ModuleInfo var tdnn1: TDNNBlock
    @ModuleInfo(key: "res2net_block") var res2netBlock: Res2NetBlock
    @ModuleInfo var tdnn2: TDNNBlock
    @ModuleInfo(key: "se_block") var seBlock: SEBlock

    init(channels: Int, kernelSize: Int = 3, dilation: Int = 1, res2netScale: Int = 8, seChannels: Int = 128) {
        _tdnn1.wrappedValue = TDNNBlock(inputChannels: channels, outputChannels: channels, kernelSize: 1)
        _res2netBlock.wrappedValue = Res2NetBlock(channels: channels, kernelSize: kernelSize, dilation: dilation, scale: res2netScale)
        _tdnn2.wrappedValue = TDNNBlock(inputChannels: channels, outputChannels: channels, kernelSize: 1)
        _seBlock.wrappedValue = SEBlock(inputDim: channels, bottleneck: seChannels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var out = tdnn1(x)
        out = res2netBlock(out)
        out = tdnn2(out)
        out = seBlock(out)
        return out + residual
    }
}

class AttentiveStatisticsPooling: Module {
    @ModuleInfo var tdnn: TDNNBlock
    @ModuleInfo var conv: Conv1d

    init(channels: Int, attentionChannels: Int = 128) {
        _tdnn.wrappedValue = TDNNBlock(inputChannels: channels * 3, outputChannels: attentionChannels, kernelSize: 1)
        _conv.wrappedValue = Conv1d(inputChannels: attentionChannels, outputChannels: channels, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: (B, T, C)
        let m = mean(x, axis: 1, keepDims: true) // (B, 1, C)
        let v = variance(x, axis: 1, keepDims: true) // (B, 1, C)
        let s = sqrt(v + 1e-9)
        let mExpanded = broadcast(m, to: x.shape)
        let sExpanded = broadcast(s, to: x.shape)

        let attnInput = concatenated([x, mExpanded, sExpanded], axis: -1) // (B, T, 3C)
        var attn = tdnn(attnInput) // (B, T, att_channels)
        attn = tanh(attn)
        attn = conv(attn) // (B, T, C)
        attn = softmax(attn, axis: 1)

        let weightedMean = sum(attn * x, axis: 1) // (B, C)
        let weightedVar = sum(attn * (x * x), axis: 1) - weightedMean * weightedMean
        let weightedStd = sqrt(maximum(weightedVar, 1e-9))

        return concatenated([weightedMean, weightedStd], axis: -1) // (B, 2C)
    }
}

// MARK: - Heterogeneous blocks wrapper

class EcapaTdnnBlocks: Module {
    @ModuleInfo(key: "block0") var block0: TDNNBlock
    @ModuleInfo(key: "block1") var block1: SERes2NetBlock
    @ModuleInfo(key: "block2") var block2: SERes2NetBlock
    @ModuleInfo(key: "block3") var block3: SERes2NetBlock

    init(nMels: Int, channels: Int) {
        _block0.wrappedValue = TDNNBlock(inputChannels: nMels, outputChannels: channels, kernelSize: 5)
        _block1.wrappedValue = SERes2NetBlock(channels: channels, kernelSize: 3, dilation: 2)
        _block2.wrappedValue = SERes2NetBlock(channels: channels, kernelSize: 3, dilation: 3)
        _block3.wrappedValue = SERes2NetBlock(channels: channels, kernelSize: 3, dilation: 4)
    }
}

// MARK: - Embedding Model

class EcapaTdnnEmbedding: Module {
    @ModuleInfo var blocks: EcapaTdnnBlocks
    @ModuleInfo var mfa: TDNNBlock
    @ModuleInfo var asp: AttentiveStatisticsPooling
    @ModuleInfo(key: "asp_bn") var aspBn: BatchNorm
    @ModuleInfo var fc: Conv1d

    init(nMels: Int = 60, channels: Int = 1024, embedDim: Int = 256) {
        _blocks.wrappedValue = EcapaTdnnBlocks(nMels: nMels, channels: channels)
        _mfa.wrappedValue = TDNNBlock(inputChannels: channels * 3, outputChannels: channels * 3, kernelSize: 1)
        _asp.wrappedValue = AttentiveStatisticsPooling(channels: channels * 3, attentionChannels: 128)
        _aspBn.wrappedValue = BatchNorm(featureCount: channels * 6)
        _fc.wrappedValue = Conv1d(inputChannels: channels * 6, outputChannels: embedDim, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = blocks.block0(x)
        var xl = [MLXArray]()
        out = blocks.block1(out); xl.append(out)
        out = blocks.block2(out); xl.append(out)
        out = blocks.block3(out); xl.append(out)

        out = concatenated(xl, axis: -1) // (B, T, channels*3)
        out = mfa(out)
        out = asp(out) // (B, channels*6)
        out = aspBn(out)
        out = expandedDimensions(out, axis: 1) // (B, 1, channels*6)
        out = fc(out) // (B, 1, embedDim)
        return out
    }
}

// MARK: - Classifier

class DNNLinear: Module {
    @ModuleInfo var w: Linear

    init(inputDim: Int, outputDim: Int) {
        _w.wrappedValue = Linear(inputDim, outputDim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { w(x) }
}

class DNNBlock: Module {
    @ModuleInfo var linear: DNNLinear
    @ModuleInfo var norm: BatchNorm

    init(inputDim: Int, outputDim: Int) {
        _linear.wrappedValue = DNNLinear(inputDim: inputDim, outputDim: outputDim)
        _norm.wrappedValue = BatchNorm(featureCount: outputDim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return relu(norm(linear(x)))
    }
}

class DNN: Module {
    @ModuleInfo(key: "block_0") var block0: DNNBlock

    init(inputDim: Int, outputDim: Int) {
        _block0.wrappedValue = DNNBlock(inputDim: inputDim, outputDim: outputDim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { block0(x) }
}

class ClassifierLinear: Module {
    @ModuleInfo var w: Linear

    init(inputDim: Int, outputDim: Int) {
        _w.wrappedValue = Linear(inputDim, outputDim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { w(x) }
}

class EcapaClassifier: Module {
    @ModuleInfo var norm: BatchNorm
    @ModuleInfo var DNN: DNN  // Must match key "DNN" from checkpoint
    @ModuleInfo var out: ClassifierLinear

    init(embedDim: Int = 256, hiddenDim: Int = 512, numClasses: Int = 107) {
        _norm.wrappedValue = BatchNorm(featureCount: embedDim)
        _DNN.wrappedValue = .init(inputDim: embedDim, outputDim: hiddenDim)
        _out.wrappedValue = ClassifierLinear(inputDim: hiddenDim, outputDim: numClasses)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out_val = x.squeezed(axis: 1) // (B, embedDim)
        out_val = norm(out_val)
        out_val = DNN(out_val)
        out_val = out(out_val)
        return logSoftmax(out_val, axis: -1)
    }
}

// MARK: - Full Model

class EcapaTdnnLidModel: Module {
    @ModuleInfo(key: "embedding_model") var embeddingModel: EcapaTdnnEmbedding
    @ModuleInfo var classifier: EcapaClassifier

    init(nMels: Int = 60, channels: Int = 1024, embedDim: Int = 256, numClasses: Int = 107) {
        _embeddingModel.wrappedValue = EcapaTdnnEmbedding(nMels: nMels, channels: channels, embedDim: embedDim)
        _classifier.wrappedValue = EcapaClassifier(embedDim: embedDim, hiddenDim: 512, numClasses: numClasses)
    }

    func callAsFunction(_ melFeatures: MLXArray) -> MLXArray {
        let embeddings = embeddingModel(melFeatures) // (B, 1, embedDim)
        return classifier(embeddings) // (B, numClasses)
    }
}

// MARK: - Weight Loading

func loadEcapaTdnnWeights(model: EcapaTdnnLidModel, weightsURL: URL) throws {
    let rawWeights = try loadArrays(url: weightsURL)

    var mapped = [(String, MLXArray)]()
    for (key, value) in rawWeights {
        if key.contains("num_batches_tracked") { continue }
        guard let mlxKey = mapEcapaKey(key) else { continue }
        mapped.append((mlxKey, value))
    }

    try model.update(parameters: ModuleParameters.unflattened(mapped), verify: .none)
}

func mapEcapaKey(_ sbKey: String) -> String? {
    var key = sbKey

    if key.contains("num_batches_tracked") { return nil }

    // Flatten double-nesting: .conv.conv. → .conv., .norm.norm. → .norm.
    // Remap ONLY top-level embedding_model.blocks numeric indices (not res2net_block.blocks which is a real array)
    key = key.replacingOccurrences(of: "embedding_model.blocks.0.", with: "embedding_model.blocks.block0.")
    key = key.replacingOccurrences(of: "embedding_model.blocks.1.", with: "embedding_model.blocks.block1.")
    key = key.replacingOccurrences(of: "embedding_model.blocks.2.", with: "embedding_model.blocks.block2.")
    key = key.replacingOccurrences(of: "embedding_model.blocks.3.", with: "embedding_model.blocks.block3.")

    key = key.replacingOccurrences(of: ".conv.conv.", with: ".conv.")
    key = key.replacingOccurrences(of: ".norm.norm.", with: ".norm.")

    // SE block Conv1d wrappers
    key = key.replacingOccurrences(of: ".se_block.conv1.conv.", with: ".se_block.conv1.")
    key = key.replacingOccurrences(of: ".se_block.conv2.conv.", with: ".se_block.conv2.")

    // ASP_BN single .norm.
    key = key.replacingOccurrences(of: ".asp_bn.norm.", with: ".asp_bn.")

    // FC single .conv.
    key = key.replacingOccurrences(of: ".fc.conv.", with: ".fc.")

    return key
}

// MARK: - Public API

func loadEcapaTdnnLid(weightsPath: URL, labelsPath: URL) throws -> (EcapaTdnnLidModel, [String: String]) {
    let labelsData = try Data(contentsOf: labelsPath)
    let labels = try JSONDecoder().decode([String: String].self, from: labelsData)

    let model = EcapaTdnnLidModel(nMels: kNMels, channels: 1024, embedDim: 256, numClasses: labels.count)
    model.train(false)
    try loadEcapaTdnnWeights(model: model, weightsURL: weightsPath)
    eval(model)

    return (model, labels)
}
