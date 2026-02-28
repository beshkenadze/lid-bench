import Foundation
import MLX
import MLXNN

// MARK: - Feature Extractor (7 Conv1d layers)

class FeatureExtractorLayer: Module {
    @ModuleInfo var conv: Conv1d
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm

    init(inputChannels: Int, outputChannels: Int, kernelSize: Int, stride: Int) {
        _conv.wrappedValue = Conv1d(
            inputChannels: inputChannels, outputChannels: outputChannels,
            kernelSize: kernelSize, stride: stride, bias: true
        )
        _layerNorm.wrappedValue = LayerNorm(dimensions: outputChannels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = conv(x)
        out = layerNorm(out)
        out = gelu(out)
        return out
    }
}

class FeatureExtractor: Module {
    @ModuleInfo(key: "conv_layers") var convLayers: [FeatureExtractorLayer]

    override init() {
        let convDims = [512, 512, 512, 512, 512, 512, 512]
        let convKernels = [10, 3, 3, 3, 3, 2, 2]
        let convStrides = [5, 2, 2, 2, 2, 2, 2]
        let inChannels = [1] + Array(convDims.dropLast())

        _convLayers.wrappedValue = zip(zip(inChannels, convDims), zip(convKernels, convStrides))
            .map { FeatureExtractorLayer(
                inputChannels: $0.0, outputChannels: $0.1,
                kernelSize: $1.0, stride: $1.1
            ) }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        for layer in convLayers {
            out = layer(out)
        }
        return out
    }
}

// MARK: - Feature Projection

class FeatureProjection: Module {
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo var projection: Linear

    init(inputDim: Int = 512, outputDim: Int = 1280) {
        _layerNorm.wrappedValue = LayerNorm(dimensions: inputDim)
        _projection.wrappedValue = Linear(inputDim, outputDim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = layerNorm(x)
        out = projection(out)
        return out
    }
}

// MARK: - Positional Convolutional Embedding

class PositionalConvEmbedding: Module {
    @ModuleInfo var conv: Conv1d

    init(hiddenSize: Int = 1280, kernelSize: Int = 128, groups: Int = 16) {
        _conv.wrappedValue = Conv1d(
            inputChannels: hiddenSize, outputChannels: hiddenSize,
            kernelSize: kernelSize, padding: kernelSize / 2,
            groups: groups, bias: true
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = conv(x)
        out = gelu(out)
        return out
    }
}

// MARK: - Transformer Components

class Wav2Vec2Attention: Module {
    let numHeads: Int
    let headDim: Int

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(hiddenSize: Int = 1280, numHeads: Int = 16) {
        self.numHeads = numHeads
        self.headDim = hiddenSize / numHeads
        _qProj.wrappedValue = Linear(hiddenSize, hiddenSize)
        _kProj.wrappedValue = Linear(hiddenSize, hiddenSize)
        _vProj.wrappedValue = Linear(hiddenSize, hiddenSize)
        _outProj.wrappedValue = Linear(hiddenSize, hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let T = x.dim(1)

        let q = qProj(x).reshaped(B, T, numHeads, headDim).transposed(0, 2, 1, 3)
        let k = kProj(x).reshaped(B, T, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(B, T, numHeads, headDim).transposed(0, 2, 1, 3)

        let scale = Float(headDim).squareRoot()
        var attn = matmul(q, k.transposed(0, 1, 3, 2)) / scale
        attn = softmax(attn, axis: -1)

        let out = matmul(attn, v).transposed(0, 2, 1, 3).reshaped(B, T, -1)
        return outProj(out)
    }
}

class Wav2Vec2FeedForward: Module {
    @ModuleInfo(key: "intermediate_dense") var intermediateDense: Linear
    @ModuleInfo(key: "output_dense") var outputDense: Linear

    init(hiddenSize: Int = 1280, intermediateSize: Int = 5120) {
        _intermediateDense.wrappedValue = Linear(hiddenSize, intermediateSize)
        _outputDense.wrappedValue = Linear(intermediateSize, hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = intermediateDense(x)
        out = gelu(out)
        out = outputDense(out)
        return out
    }
}

class Wav2Vec2EncoderLayer: Module {
    @ModuleInfo var attention: Wav2Vec2Attention
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "feed_forward") var feedForward: Wav2Vec2FeedForward
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    init(hiddenSize: Int = 1280, numHeads: Int = 16, intermediateSize: Int = 5120) {
        _attention.wrappedValue = Wav2Vec2Attention(hiddenSize: hiddenSize, numHeads: numHeads)
        _layerNorm.wrappedValue = LayerNorm(dimensions: hiddenSize, eps: 1e-5)
        _feedForward.wrappedValue = Wav2Vec2FeedForward(hiddenSize: hiddenSize, intermediateSize: intermediateSize)
        _finalLayerNorm.wrappedValue = LayerNorm(dimensions: hiddenSize, eps: 1e-5)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Pre-norm (stable layer norm)
        var out = x
        let residual1 = out
        out = layerNorm(out)
        out = attention(out)
        out = residual1 + out

        let residual2 = out
        out = finalLayerNorm(out)
        out = feedForward(out)
        out = residual2 + out
        return out
    }
}

// MARK: - Full Encoder

class Wav2Vec2Encoder: Module {
    @ModuleInfo(key: "pos_conv_embed") var posConvEmbed: PositionalConvEmbedding
    @ModuleInfo var layers: [Wav2Vec2EncoderLayer]
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm

    init(hiddenSize: Int = 1280, numLayers: Int = 48, numHeads: Int = 16, intermediateSize: Int = 5120) {
        _posConvEmbed.wrappedValue = PositionalConvEmbedding(hiddenSize: hiddenSize)
        _layers.wrappedValue = (0..<numLayers).map { _ in
            Wav2Vec2EncoderLayer(hiddenSize: hiddenSize, numHeads: numHeads, intermediateSize: intermediateSize)
        }
        _layerNorm.wrappedValue = LayerNorm(dimensions: hiddenSize, eps: 1e-5)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        // Add positional embeddings
        var pos = posConvEmbed(out)
        // Trim to match sequence length (padding=64 with k=128 adds 1 extra)
        pos = pos[0..., ..<out.dim(1), 0...]
        out = out + pos

        for layer in layers {
            out = layer(out)
        }
        out = layerNorm(out)
        return out
    }
}

// MARK: - Full Model

class MmsLid256Model: Module {
    @ModuleInfo(key: "feature_extractor") var featureExtractor: FeatureExtractor
    @ModuleInfo(key: "feature_projection") var featureProjection: FeatureProjection
    @ModuleInfo var encoder: Wav2Vec2Encoder
    @ModuleInfo var projector: Linear
    @ModuleInfo var classifier: Linear

    init(numClasses: Int = 256) {
        _featureExtractor.wrappedValue = FeatureExtractor()
        _featureProjection.wrappedValue = FeatureProjection(inputDim: 512, outputDim: 1280)
        _encoder.wrappedValue = Wav2Vec2Encoder(hiddenSize: 1280, numLayers: 48, numHeads: 16, intermediateSize: 5120)
        _projector.wrappedValue = Linear(1280, 1024)
        _classifier.wrappedValue = Linear(1024, numClasses)
    }

    func callAsFunction(_ waveform: MLXArray) -> MLXArray {
        // waveform: (B, T) raw 16kHz audio, normalized
        var x = expandedDimensions(waveform, axis: -1) // (B, T, 1)
        x = featureExtractor(x) // (B, T', 512)
        x = featureProjection(x) // (B, T', 1280)
        x = encoder(x) // (B, T', 1280)
        x = mean(x, axis: 1) // (B, 1280)
        x = projector(x) // (B, 1024)
        let logits = classifier(x) // (B, numClasses)
        return logSoftmax(logits, axis: -1)
    }
}

// MARK: - Weight Loading

func computeWeightNorm(weightG: MLXArray, weightV: MLXArray) -> MLXArray {
    // weight_g: [1, 1, kernel_size], weight_v: [out, groups_dim, kernel_size]
    // norm over dims 0 and 1
    let norm = sqrt(sum(weightV * weightV, axes: [0, 1], keepDims: true) + 1e-12)
    return weightG * weightV / norm
}

func loadMmsLid256Weights(model: MmsLid256Model, modelDir: URL) throws {
    let weightsURL = modelDir.appendingPathComponent("model.safetensors")
    let rawWeights = try loadArrays(url: weightsURL)

    var mapped = [(String, MLXArray)]()

    // Collect weight_g and weight_v for positional conv weight norm
    var weightG: MLXArray?
    var weightV: MLXArray?

    for (hfKey, tensor) in rawWeights.sorted(by: { $0.key < $1.key }) {
        // Skip training-only tensors
        if hfKey.contains("masked_spec_embed") { continue }
        if hfKey.contains("adapter_layer") { continue }

        guard let mlxKey = mapMmsHfKey(hfKey) else { continue }

        var value = tensor

        // Handle positional conv weight_norm components
        if mlxKey == "encoder.pos_conv_embed.conv.weight_g" {
            weightG = value
            continue
        }
        if mlxKey == "encoder.pos_conv_embed.conv.weight_v" {
            weightV = value
            continue
        }

        // Conv1d weight axis swap: HF [out, in, kernel] → MLX [out, kernel, in]
        if mlxKey.hasSuffix(".conv.weight") {
            value = value.swappedAxes(1, 2)
        }

        mapped.append((mlxKey, value))
    }

    // Compute positional conv weight from weight_norm
    if let g = weightG, let v = weightV {
        var fullWeight = computeWeightNorm(weightG: g, weightV: v)
        // [1280, 80, 128] in HF → swap to MLX [1280, 128, 80]
        fullWeight = fullWeight.swappedAxes(1, 2)
        mapped.append(("encoder.pos_conv_embed.conv.weight", fullWeight))
    }

    try model.update(parameters: ModuleParameters.unflattened(mapped), verify: .none)
}

func mapMmsHfKey(_ hfKey: String) -> String? {
    // Classifier head (no wav2vec2. prefix)
    if hfKey.hasPrefix("projector.") { return hfKey }
    if hfKey.hasPrefix("classifier.") { return hfKey }

    // Strip wav2vec2. prefix
    guard hfKey.hasPrefix("wav2vec2.") else { return nil }
    let key = String(hfKey.dropFirst("wav2vec2.".count))

    if key.hasPrefix("feature_extractor.") { return key }
    if key.hasPrefix("feature_projection.") { return key }
    if key.hasPrefix("encoder.") { return key }

    return nil
}

// MARK: - Public API

func loadMmsLid256(modelDir: URL, labelsPath: URL) throws -> (MmsLid256Model, [String: String]) {
    let labelsData = try Data(contentsOf: labelsPath)
    let labels = try JSONDecoder().decode([String: String].self, from: labelsData)

    let model = MmsLid256Model(numClasses: labels.count)
    model.train(false)
    try loadMmsLid256Weights(model: model, modelDir: modelDir)
    eval(model)

    return (model, labels)
}

func normalizeWaveform(_ audio: [Float]) -> MLXArray {
    let arr = MLXArray(audio)
    let m = mean(arr)
    let s = sqrt(mean((arr - m) * (arr - m)))
    let normalized = (arr - m) / (s + 1e-7)
    return expandedDimensions(normalized, axis: 0) // (1, T)
}
