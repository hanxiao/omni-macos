import Foundation

/// What indexing costs on THIS Mac, learned from the indexing itself, so a pass can say how long it
/// has left ("about 12 min") from the moment a folder's files are counted.
///
/// No numbers are assumed: a kind has no cost - and the pass no estimate - until this Mac has
/// measured enough of it (`trusted`). Time is wall time of the pass, the thing the user waits on,
/// attributed per kind phase (a wave indexes kind by kind, so the time between two checkpoints of
/// one phase belongs to that kind alone - nothing is split by guesswork):
/// - an image costs per file;
/// - audio costs per second of sound and video per 240 s segment (a segment is a fixed number of
///   frames however long it is), their lengths read from the file headers (MediaDuration) - not
///   per MB: the first real run priced 137 MB of WAV at the rate of short AAC clips and said 184 s
///   for a pass that took 15;
/// - text costs per chunk that reached the model (a reused vector costs nothing and is not
///   counted), and a scanned page per page, timed apart; how many chunks and scanned pages a MB
///   makes is learned separately for PDFs, office documents and plain text from the files stored.
/// Recent work counts most: a tally past `decayAfter` seconds is halved before more is added, and a
/// different model, image size, frame count or chunk size starts the table over (`regime`).
public final class IndexCosts: @unchecked Sendable {
    public enum TextSort: String, Codable, Sendable, CaseIterable { case pdf, office, plain }

    struct Tally: Codable, Equatable {
        var seconds = 0.0
        var units = 0.0
        var files = 0
    }
    struct ChunkTally: Codable, Equatable {
        var chunks = 0.0
        var scanPages = 0.0
        var mb = 0.0
        var files = 0
    }
    private struct Saved: Codable {
        var regime: String
        var time: [String: Tally]
        var chunks: [String: ChunkTally]
    }

    /// A kind's cost is used from its first measured file (the warm-up file is never measured);
    /// more files only refine it. Higher bars were tried on real runs and failed the same way: a
    /// kind the user has two of (2 videos in a folder of 480 files) never reached the bar, and when
    /// its phase came last the whole pass went without an estimate for a kind worth 5% of it. A
    /// per-sort chunk ratio does wait for `sortTrustFiles`, with the pooled ratio standing in.
    static let trustFiles = 1
    static let sortTrustFiles = 5
    static let trustScanPages = 10.0
    static let decayAfter = 900.0

    private let url: URL
    private let lock = NSLock()
    private var regime = ""
    private var time: [FileKind: Tally] = [:]
    private var chunks: [TextSort: ChunkTally] = [:]
    private var dirty = false

    public init(directory: URL) {
        url = directory.appendingPathComponent("costs.json")
        if let s = try? JSONDecoder().decode(Saved.self, from: Data(contentsOf: url)) {
            regime = s.regime
            for (k, v) in s.time { if let kind = FileKind(rawValue: k) { time[kind] = v } }
            for (k, v) in s.chunks { if let sort = TextSort(rawValue: k) { chunks[sort] = v } }
        }
    }

    /// What the costs depend on besides the Mac. A change starts the table over.
    public static func regime(dim: Int, settings: IndexSettings) -> String {
        // u2: media priced by length (u1 was by MB).
        "u3/\(dim)/\(settings.maxImageDimension)/\(settings.maxVideoFrames)/\(settings.maxCharsPerChunk)"
    }

    public func begin(regime r: String) {
        lock.withLock {
            guard r != regime else { return }
            regime = r; time = [:]; chunks = [:]; dirty = true
        }
    }

    public static func sort(ext: String) -> TextSort {
        switch ext {
        case "pdf": return .pdf
        case "doc", "docx", "pages", "key", "keynote", "ppt", "pptx", "xls", "xlsx", "numbers", "rtf", "rtfd",
             "odt", "ods", "odp", "epub": return .office
        default: return .plain
        }
    }

    /// `seconds` of one kind's phase, in which `units` were finished (files for images, seconds
    /// for audio, segments for video, chunks for text) across `files` files.
    func addTime(_ kind: FileKind, seconds: Double, units: Double, files: Int) {
        guard seconds > 0, units > 0 else { return }
        lock.withLock {
            var t = time[kind, default: Tally()]
            if t.seconds > Self.decayAfter { t.seconds /= 2; t.units /= 2; t.files /= 2 }
            t.seconds += seconds; t.units += units; t.files += files
            time[kind] = t; dirty = true
        }
    }

    /// A stored text file: its text chunks, or the pages it had to read as images, and its size.
    func addChunks(_ sort: TextSort, chunks n: Int, scanPages: Int, mb: Double) {
        guard n + scanPages > 0, mb > 0 else { return }
        lock.withLock {
            var c = chunks[sort, default: ChunkTally()]
            if c.files > 2000 { c.chunks /= 2; c.scanPages /= 2; c.mb /= 2; c.files /= 2 }
            c.chunks += Double(n); c.scanPages += Double(scanPages); c.mb += mb; c.files += 1
            chunks[sort] = c; dirty = true
        }
    }

    /// Seconds per unit, once trusted. Scans come a document at a time, so they are trusted by
    /// pages; until then a scanned page is priced as the image it is - one picture through the same
    /// vision tower, measured here too.
    public func rate(_ kind: FileKind) -> Double? {
        lock.withLock {
            func measured(_ k: FileKind) -> Double? {
                guard let t = time[k], t.units > 0, t.seconds > 0,
                      k == .scan ? t.units >= Self.trustScanPages : t.files >= Self.trustFiles else { return nil }
                return t.seconds / t.units
            }
            return measured(kind) ?? (kind == .scan ? measured(.image) : nil)
        }
    }

    /// Text chunks and scanned pages a MB of this sort makes, once trusted; the pooled ratio of
    /// all text stands in for a sort not seen enough yet.
    public func perMB(_ sort: TextSort) -> (chunks: Double, scanPages: Double)? {
        lock.withLock {
            if let c = chunks[sort], c.files >= Self.sortTrustFiles, c.mb > 0 { return (c.chunks / c.mb, c.scanPages / c.mb) }
            let all = chunks.values.reduce(into: ChunkTally()) {
                $0.chunks += $1.chunks; $0.scanPages += $1.scanPages; $0.mb += $1.mb; $0.files += $1.files
            }
            guard all.files >= Self.trustFiles, all.mb > 0 else { return nil }
            return (all.chunks / all.mb, all.scanPages / all.mb)
        }
    }

    public func save() {
        lock.withLock {
            guard dirty else { return }
            dirty = false
            let s = Saved(regime: regime,
                          time: Dictionary(uniqueKeysWithValues: time.map { ($0.key.rawValue, $0.value) }),
                          chunks: Dictionary(uniqueKeysWithValues: chunks.map { ($0.key.rawValue, $0.value) }))
            if let d = try? JSONEncoder().encode(s) { try? d.write(to: url, options: .atomic) }
        }
    }
}

/// The work a pass has left, by root, in the units IndexCosts prices.
struct WorkLeft: Sendable {
    var imageFiles = 0.0
    var audioSeconds = 0.0
    var videoSegments = 0.0
    var textMB: [IndexCosts.TextSort: Double] = [:]

    mutating func subtract(_ o: WorkLeft) {
        imageFiles -= o.imageFiles; audioSeconds -= o.audioSeconds; videoSegments -= o.videoSegments
        for (k, v) in o.textMB { textMB[k, default: 0] -= v }
    }

    mutating func add(_ u: WorkUnit, sign: Double = 1) {
        switch u.kind {
        case .image: imageFiles += sign * u.units
        case .audio: audioSeconds += sign * u.units
        case .video: videoSegments += sign * u.units
        case .text, .scan: textMB[u.sort, default: 0] += sign * u.units
        }
    }

    /// Seconds, or nil while a kind with work left has no measured cost.
    func seconds(_ costs: IndexCosts) -> Double? { seconds(rate: costs.rate, perMB: costs.perMB) }

    func seconds(rate: (FileKind) -> Double?,
                 perMB: (IndexCosts.TextSort) -> (chunks: Double, scanPages: Double)?) -> Double? {
        var s = 0.0
        func price(_ kind: FileKind, _ units: Double) -> Bool {
            guard units > 1e-9 else { return true }
            guard let r = rate(kind) else { return false }
            s += r * units; return true
        }
        guard price(.image, imageFiles), price(.audio, audioSeconds), price(.video, videoSegments) else { return nil }
        for (sort, mb) in textMB where mb > 1e-9 {
            guard let per = perMB(sort), price(.text, mb * per.chunks), price(.scan, mb * per.scanPages) else { return nil }
        }
        return s
    }
}

/// One file's work, for the estimate: its kind and its amount - one image, seconds of audio,
/// segments of video, MB of text (priced through chunks per MB).
struct WorkUnit: Sendable {
    let kind: FileKind
    let sort: IndexCosts.TextSort
    let units: Double

    init(kind: FileKind, sort: IndexCosts.TextSort, units: Double) { self.kind = kind; self.sort = sort; self.units = units }

    /// A long text file that is not read to its end (a log or data file, unless the setting says
    /// so) is read to the 2 MB cut only. Audio and video read their header for their length.
    init?(_ f: CrawledFile, settings: IndexSettings) {
        let ext = f.ext.lowercased()
        guard let k = FileExtractor.kind(forExtension: ext) else { return nil }
        kind = k
        sort = k == .text ? IndexCosts.sort(ext: ext) : .plain
        switch k {
        case .image:
            units = 1
        case .audio:
            units = f.isPhoto ? 0 : (MediaDuration.seconds(f.path, ext: ext) ?? 0)
        case .video:
            let d = f.isPhoto ? nil : MediaDuration.seconds(f.path, ext: ext)
            units = max(1, ((d ?? 0) / Indexer.mediaSegmentSeconds).rounded(.up))
        case .text, .scan:
            var bytes = Double(max(0, f.size))
            if !Indexer.readsToEnd(f, settings: settings) { bytes = Double(FileExtractor.maxTextBytes) }
            units = bytes / 1_000_000
        }
    }

    /// Many at once: the header reads of a big batch of media run in parallel.
    static func of(_ files: [CrawledFile], settings: IndexSettings) -> [WorkUnit?] {
        let media = files.reduce(0) { n, f in
            let e = f.ext.lowercased()
            return n + (FileExtractor.audioExtensions.contains(e) || FileExtractor.videoExtensions.contains(e) ? 1 : 0)
        }
        guard media >= 64 else { return files.map { WorkUnit($0, settings: settings) } }
        final class Out: @unchecked Sendable { var v: [WorkUnit?] = [] }
        let out = Out(); out.v = Array(repeating: nil, count: files.count)
        let stride = 256
        DispatchQueue.concurrentPerform(iterations: (files.count + stride - 1) / stride) { b in
            for i in b * stride ..< min(files.count, (b + 1) * stride) {
                let u = WorkUnit(files[i], settings: settings)
                out.v.withUnsafeMutableBufferPointer { $0[i] = u }
            }
        }
        return out.v
    }
}
