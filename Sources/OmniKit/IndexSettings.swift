import Foundation

/// Which modalities to index. Default: media on, text off (audio pending support).
public struct IndexSettings: Sendable, Equatable {
    public var enabledKinds: Set<FileKind>
    /// Cap the largest image/PDF-page side decoded for embedding. The vision model
    /// resizes to <= ~1.3MP anyway, so decoding larger wastes time and memory.
    public var maxImageDimension: Int = 1568
    /// Frames sampled per video.
    public var maxVideoFrames: Int = 6

    public init(enabledKinds: Set<FileKind> = [.image, .video, .audio]) {
        self.enabledKinds = enabledKinds
    }

    public func contains(_ k: FileKind) -> Bool { enabledKinds.contains(k) }

    public mutating func set(_ k: FileKind, _ on: Bool) {
        if on { enabledKinds.insert(k) } else { enabledKinds.remove(k) }
    }

    public static let `default` = IndexSettings()
}
