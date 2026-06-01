import Foundation

/// Configuration for jina-embeddings-v5-omni-small, parsed from the model's config.json.
/// Only the fields the MLX-Swift port consumes are kept.
public struct OmniConfig: Sendable {
    public struct Text: Sendable {
        public var hiddenSize = 1024
        public var numLayers = 28
        public var intermediateSize = 3072
        public var numHeads = 16
        public var numKVHeads = 8
        public var headDim = 128
        public var rmsNormEps: Float = 1e-6
        public var vocabSize = 151672
        public var ropeTheta: Float = 3_500_000
    }

    public struct Vision: Sendable {
        public var hiddenSize = 1024
        public var depth = 24
        public var numHeads = 16
        public var intermediateSize = 4096
        public var patchSize = 16
        public var spatialMergeSize = 2
        public var temporalPatchSize = 2
        public var inChannels = 3
        public var outHiddenSize = 1024
        public var numPositionEmbeddings = 2304
    }

    public var text = Text()
    public var vision = Vision()
    public var imageTokenId = 151655
    public var videoTokenId = 151656
    public var visionStartTokenId = 151652
    public var visionEndTokenId = 151653

    /// LoRA scale (alpha / r). Retrieval adapter is alpha=32, r=32 -> 1.0.
    public var loraScale: Float = 1.0

    public static let queryPrefix = "Query: "
    public static let passagePrefix = "Document: "

    public init() {}

    /// Parse from the model directory's config.json. Falls back to defaults for missing keys.
    public init(modelDir: URL) throws {
        var cfg = OmniConfig()
        let url = modelDir.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: url),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            self = cfg
            return
        }
        if let tc = root["text_config"] as? [String: Any] {
            cfg.text.hiddenSize = tc["hidden_size"] as? Int ?? cfg.text.hiddenSize
            cfg.text.numLayers = tc["num_hidden_layers"] as? Int ?? cfg.text.numLayers
            cfg.text.intermediateSize = tc["intermediate_size"] as? Int ?? cfg.text.intermediateSize
            cfg.text.numHeads = tc["num_attention_heads"] as? Int ?? cfg.text.numHeads
            cfg.text.numKVHeads = tc["num_key_value_heads"] as? Int ?? cfg.text.numKVHeads
            cfg.text.headDim = tc["head_dim"] as? Int ?? cfg.text.headDim
            cfg.text.vocabSize = tc["vocab_size"] as? Int ?? cfg.text.vocabSize
            if let eps = tc["rms_norm_eps"] as? Double { cfg.text.rmsNormEps = Float(eps) }
            if let rp = tc["rope_parameters"] as? [String: Any],
               let theta = rp["rope_theta"] as? Double {
                cfg.text.ropeTheta = Float(theta)
            }
        }
        if let vc = root["vision_config"] as? [String: Any] {
            cfg.vision.depth = vc["depth"] as? Int ?? cfg.vision.depth
            cfg.vision.hiddenSize = vc["hidden_size"] as? Int ?? cfg.vision.hiddenSize
            cfg.vision.numHeads = vc["num_heads"] as? Int ?? cfg.vision.numHeads
            cfg.vision.intermediateSize = vc["intermediate_size"] as? Int ?? cfg.vision.intermediateSize
            cfg.vision.numPositionEmbeddings = vc["num_position_embeddings"] as? Int ?? cfg.vision.numPositionEmbeddings
        }
        cfg.imageTokenId = root["image_token_id"] as? Int ?? cfg.imageTokenId
        cfg.videoTokenId = root["video_token_id"] as? Int ?? cfg.videoTokenId
        self = cfg
    }
}

/// Embedding task / input type. Selects the text prefix (the retrieval LoRA is always merged).
public enum OmniInputType: Sendable {
    case query
    case passage

    var prefix: String {
        switch self {
        case .query: return OmniConfig.queryPrefix
        case .passage: return OmniConfig.passagePrefix
        }
    }
}
