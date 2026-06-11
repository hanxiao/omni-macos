import Foundation

/// Which modalities to index. Default: all four on (text, image, audio, video).
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
    /// DEPRECATED as crawl policy - kept only as a migration source for `ignore` (see OmniIgnore).
    public var disabledExtensions: Set<String> = []

    /// The single source of truth for what is EXCLUDED from indexing (gitignore semantics). The crawl
    /// indexes a file iff `FileExtractor.kind(for:) != nil` and `!ignore.isIgnored(path)`. Built by
    /// AppModel from the user's .omniignore file (which migration seeds from the legacy kind/extension
    /// settings). `.default`/`.profiling` leave it empty = index everything extractable.
    public var ignore: OmniIgnore = OmniIgnore(text: "")

    /// Order the modalities are indexed in (user-reorderable). A uniform phase per kind lets text
    /// chunks batch across files; the order sets which modality is embedded first. Text is last by
    /// default: media is slower to index, so getting it done first surfaces those results sooner.
    public var kindOrder: [FileKind] = [.image, .audio, .video, .text]

    // Index-time minimums: files below these are skipped (0 = no minimum).
    public var minImageDimension: Int = 0   // largest image side, px
    public var minAudioSeconds: Double = 0
    public var minVideoSeconds: Double = 0
    public var minTextChars: Int = 0

    /// Dataless (cloud-evicted) files: their content lives remotely, and reading it for embedding
    /// implicitly DOWNLOADS the file (iCloud Optimize Mac Storage / FileProvider). `true` (default)
    /// skips them - no surprise downloads, no disk refill, no offline stalls; they index when the
    /// user materializes them (the FSEvents reconcile picks the download up). An already-indexed
    /// file that later gets evicted KEEPS its index entry (eviction does not change content), so it
    /// stays searchable. `false` restores read-through behavior: indexing downloads as it goes.
    public var skipDataless: Bool = true

    public init(enabledKinds: Set<FileKind> = [.text, .image, .video, .audio]) {
        self.enabledKinds = enabledKinds
    }

    public func contains(_ k: FileKind) -> Bool { enabledKinds.contains(k) }

    public mutating func set(_ k: FileKind, _ on: Bool) {
        if on { enabledKinds.insert(k) } else { enabledKinds.remove(k) }
    }

    public static let `default` = IndexSettings()

    /// Fixed workload for a profiling run, so every machine indexes the IDENTICAL set of files with
    /// the IDENTICAL per-file work - the only variables left are the hardware and the app version's
    /// efficiency. All kinds on, no min thresholds (every file in the curated dataset is indexed,
    /// disregarding the user's own settings), no per-extension exclusions, and standard caps. Frozen
    /// on purpose: changing any of these moves the benchmark baseline and breaks comparability.
    public static let profiling: IndexSettings = {
        var s = IndexSettings(enabledKinds: [.text, .image, .video, .audio])
        s.maxImageDimension = 1568
        s.maxVideoFrames = 6
        s.maxCharsPerChunk = 1800
        s.minImageDimension = 0; s.minAudioSeconds = 0; s.minVideoSeconds = 0; s.minTextChars = 0
        s.disabledExtensions = []
        // Seed the well-known noise dirs the old crawl always skipped, so the workload stays identical
        // to pre-OmniIgnore profiling runs (no per-extension/kind exclusion, but noise dirs still pruned).
        s.ignore = OmniIgnore(text: FileCrawler.skipDirNames.map { "\($0)/" }.joined(separator: "\n"))
        s.kindOrder = [.image, .audio, .video, .text]
        return s
    }()
}
