import Foundation

/// Which modalities to index. Default: media on, text off (audio pending support).
public struct IndexSettings: Sendable, Equatable {
    public var enabledKinds: Set<FileKind>

    public init(enabledKinds: Set<FileKind> = [.image, .video]) {
        self.enabledKinds = enabledKinds
    }

    public func contains(_ k: FileKind) -> Bool { enabledKinds.contains(k) }

    public mutating func set(_ k: FileKind, _ on: Bool) {
        if on { enabledKinds.insert(k) } else { enabledKinds.remove(k) }
    }

    public static let `default` = IndexSettings()
}
