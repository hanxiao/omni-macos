import Foundation

/// Which modalities to index. Default: media on, text off (audio pending support).
public struct IndexSettings: Sendable, Equatable {
    public var enabledKinds: Set<FileKind>
    /// Cap the largest image/PDF-page side decoded for embedding. The vision model
    /// resizes to <= ~1.3MP anyway, so decoding larger wastes time and memory.
    public var maxImageDimension: Int = 1568
    /// Frames sampled per video.
    public var maxVideoFrames: Int = 6
    /// Longest text slice (characters) embedded as one chunk; longer text is split with overlap.
    public var maxCharsPerChunk: Int = 1800

    /// File extensions (lowercased, no dot) the user has turned off within an enabled kind, e.g.
    /// "gif" while Images stays on. Excluded from the crawl like a disabled kind.
    public var disabledExtensions: Set<String> = []

    /// Order the modalities are indexed in (user-reorderable). A uniform phase per kind lets text
    /// chunks batch across files; the order sets which modality is embedded first.
    public var kindOrder: [FileKind] = [.text, .image, .audio, .video]

    // Index-time minimums: files below these are skipped (0 = no minimum).
    public var minImageDimension: Int = 0   // largest image side, px
    public var minAudioSeconds: Double = 0
    public var minVideoSeconds: Double = 0
    public var minTextChars: Int = 0

    public init(enabledKinds: Set<FileKind> = [.image, .video, .audio]) {
        self.enabledKinds = enabledKinds
    }

    public func contains(_ k: FileKind) -> Bool { enabledKinds.contains(k) }

    public mutating func set(_ k: FileKind, _ on: Bool) {
        if on { enabledKinds.insert(k) } else { enabledKinds.remove(k) }
    }

    public static let `default` = IndexSettings()
}
