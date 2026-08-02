import Foundation

/// Which modalities to index. Default: all four on (text, image, audio, video).
public struct IndexSettings: Sendable, Equatable {
    public var enabledKinds: Set<FileKind>
    /// Cap the largest image/PDF-page side decoded for embedding. The vision model
    /// resizes to <= ~1.3MP anyway, so decoding larger wastes time and memory.
    public var maxImageDimension: Int = 1568
    /// Frames sampled uniformly per video (per 240 s segment for videos longer than one).
    /// 32 matches the reference pipeline's evaluation policy, and the shared smart_resize
    /// pixel budget makes it cost the same GPU tokens as 16 for >= 720p sources (measured:
    /// 595 ms/6222 tok at 16 vs 590 ms/5830 tok at 32 on a 720p clip); perceptual dedup
    /// collapses the static low-res case where extra frames would actually cost.
    public var maxVideoFrames: Int = 32
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

    /// Open-vocabulary image tags (OmniTagger): images indexed while the tagger is ready get a
    /// content-tag snippet ("kitty, cosy, plush") instead of the bare filename. Rides the same
    /// embedding forward pass (one extra matmul per image); OFF only disables the extra scoring.
    /// Existing rows are untouched either way - tags appear as files (re)index.
    public var imageTags: Bool = true

    /// Bypass the content-dedup shortcut and always run the real embed. TRANSIENT - set only by
    /// the tag-backfill pass: dedup would otherwise hand a re-tagged file its OWN old (untagged)
    /// rows back, so the backfill could never converge. Normal indexing never sets this.
    public var forceFreshEmbed: Bool = false

    /// CWR multi-crop tag refinement for still images (5 extra crop forwards per image, the
    /// study's proven quality lever). TRANSIENT - set only by the search-driven retag pass,
    /// where the batch is 8 files the user is actually looking at; the bulk index pass never
    /// pays it. Video/scans are unaffected (their segments/pages are already localized views).
    public var hqMediaTags: Bool = false

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

    /// Fixed workload for the hidden paper benchmark. NOT `.profiling`: that one enables all four
    /// modalities, and the paper's index-pass case measures the text path over a synthetic text
    /// corpus - a media file appearing in the tree would change both the token count and the batch
    /// composition. Text only, no minimums, tags off (the tagger may or may not be resident, and a
    /// tagged image costs an extra matmul), noise dirs pruned so the crawl matches the app's.
    /// Frozen on purpose: changing any of these breaks comparability with every export already
    /// collected, exactly as for `.profiling`.
    public static let paper: IndexSettings = {
        var s = IndexSettings(enabledKinds: [.text])
        s.maxCharsPerChunk = 1800
        s.minImageDimension = 0; s.minAudioSeconds = 0; s.minVideoSeconds = 0; s.minTextChars = 0
        s.disabledExtensions = []
        s.imageTags = false
        s.skipDataless = true
        s.ignore = OmniIgnore(text: FileCrawler.skipDirNames.map { "\($0)/" }.joined(separator: "\n"))
        s.kindOrder = [.text]
        return s
    }()

    /// The media variant of `.paper`, used only by the optional tagging-overhead case. Images only;
    /// `imageTags` is the arm variable and is set by the case, not here.
    public static let paperMedia: IndexSettings = {
        var s = IndexSettings(enabledKinds: [.image])
        s.maxImageDimension = 1568
        s.minImageDimension = 0; s.minAudioSeconds = 0; s.minVideoSeconds = 0; s.minTextChars = 0
        s.disabledExtensions = []
        s.skipDataless = true
        s.ignore = OmniIgnore(text: FileCrawler.skipDirNames.map { "\($0)/" }.joined(separator: "\n"))
        s.kindOrder = [.image]
        return s
    }()
}
