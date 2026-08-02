import Foundation

// The synthetic corpus the indexing cases run over.
//
// Every byte of it is a pure function of (corpusSeed, file index, position). Nothing is drawn from
// a shared consuming stream, nothing depends on generation order, nothing reads the clock, the
// hostname, the locale or the filesystem. That is not tidiness - it is the whole premise of the
// exercise: two machines whose exports are merged must have indexed THE SAME BYTES, and the only
// way to know that without shipping a 3 MB tarball is to derive the bytes from a seed and hash the
// result. `corpus.fnv1a64` is the merge key, and it is computed over the tree that was actually
// written, not over the parameters that were meant to produce it.
//
// Four decisions here are load-bearing and are deliberately NOT the obvious ones:
//
//  1. Sizes come from a fixed 100-entry table indexed by `i % 100`, not by `hash(i) % 100` as the
//     plan first sketched. With a hash the bucket counts are only probabilistic, so neither the
//     total byte count nor the plan's "at least 150 files with 5+ chunks" (which p05 depends on)
//     is guaranteed - and a merge key that is only probably right is not a merge key. The table is
//     a fixed permutation, so any prefix of it is still a mixed sample for a `--scale` run.
//  2. Text is English words sampled with a head bias, not uniformly over the vocabulary and
//     emphatically not a repeated sentence (run_bench.sh:55-65) or a repeated character. Token
//     counts feed the reported tok/s, and real prose is Zipfian: a uniform draw over 512 words
//     produces an unnaturally high rate of rare words, which splits into more subword tokens per
//     word than any real document does.
//  3. The PNGs are painted and encoded HERE, with integer arithmetic and stored (uncompressed)
//     deflate blocks, rather than through CoreGraphics and ImageIO. ImageIO's PNG encoder is not a
//     pinned dependency; if its output bytes change between two macOS releases the corpus hash
//     changes with them, and two machines that generated the identical picture would refuse to
//     merge - or worse, quietly differ in decoded pixels.
//  4. The manifest stamp lives OUTSIDE the crawled tree (`<root>/manifest.json`, tree at
//     `<root>/corpus/`), so the tree contains exactly `textFiles + wideFiles + images` files and
//     the crawl case's per-file cost is divided by a number that means something.
//
// Non-destructive guarantee: this file writes and deletes only under a directory it has verified is
// inside NSTemporaryDirectory() AND named `omni-paper-corpus-*`. PaperFS owns the same guarantee
// for the run directory; the corpus deliberately lives outside it so a second run on the same
// machine does not regenerate 4,616 files.

/// Sizes and identity of the corpus. Every field is part of the merge key: two runs that disagree
/// on any of them did not index the same bytes, whatever else they have in common.
public struct PaperCorpusSpec: Sendable, Equatable, Codable {
    /// Bumped deliberately, exactly like `ProfilingService.datasetVersion`. A bump produces a new
    /// cache directory rather than silently reusing old bytes under a new meaning.
    public static let version = "paper-corpus-1"
    /// The one seed. ASCII "OMNI_P1L". Every stream in this file derives from it by index.
    public static let seed: UInt64 = 0x4F4D_4E49_5F50_314C

    public var textFiles: Int
    public var wideFiles: Int
    public var images: Int
    /// Side of the synthetic PNGs. Not scaled: 512 is what the vision tower's patch packing is
    /// measured at, and a smaller image would measure a different code path, not less of the same.
    public var imagePixels: Int
    /// Share of paragraphs replaced by a paragraph copied from an earlier file, in permille. The
    /// dedup path must be non-degenerate; this is the INJECTION rate, and it is not claimed to
    /// reproduce the real corpus's measured 3.15% cross-file chunk duplication.
    public var duplicateParagraphPermille: Int

    public init(textFiles: Int = 600, wideFiles: Int = 4000, images: Int = 16,
                imagePixels: Int = 512, duplicateParagraphPermille: Int = 120) {
        self.textFiles = textFiles; self.wideFiles = wideFiles; self.images = images
        self.imagePixels = imagePixels; self.duplicateParagraphPermille = duplicateParagraphPermille
    }

    /// The suite's spec at a given `--scale`. File COUNTS shrink; per-file sizes, the image side and
    /// the duplication rate do not, because those change what is measured rather than how much.
    /// Minimums match the catalog's (`text_files` >= 24, `wide_files` >= 100, `images` >= 4).
    public init(scale: Double) {
        func shrink(_ v: Int, _ floor: Int) -> Int {
            scale == 1.0 ? v : max(floor, Int((Double(v) * scale).rounded()))
        }
        self.init(textFiles: shrink(600, 24), wideFiles: shrink(4000, 100), images: shrink(16, 4))
    }

    /// Directory tag for `PaperFS(corpusVersion:)`. A scaled corpus gets its own directory: it is a
    /// different tree with a different hash, and reusing the full corpus's directory for it would
    /// mean one of the two is silently wrong.
    public var directoryTag: String {
        self == PaperCorpusSpec() ? Self.version
            : "\(Self.version)-t\(textFiles)-w\(wideFiles)-i\(images)"
    }

    public var totalFiles: Int { textFiles + wideFiles + images }

    /// Exact, not estimated: the size table is indexed by `i % 100`, so the total is arithmetic.
    public var expectedTextBytes: Int {
        (0 ..< textFiles).reduce(0) { $0 + PaperCorpus.sizeTable[$1 % PaperCorpus.sizeTable.count] }
    }
    /// Every wide file is exactly `wideFileBytes` long by construction.
    public var expectedWideBytes: Int { wideFiles * PaperCorpus.wideFileBytes }
}

/// Which edit p05 applies. Raw values match the catalog's `edits` parameter, so the arm name in the
/// export and the edit actually performed cannot drift apart.
public enum PaperTextEdit: String, Sendable, Codable, CaseIterable {
    /// One line appended. Every earlier chunk boundary stays byte-identical - the best case for
    /// chunk reuse, and the most common real edit (notes, logs, appended sections).
    case append
    /// One line inserted at the byte midpoint. Every boundary after the edit shifts, so only the
    /// chunks before it can be reused. The honest average case.
    case mid
}

/// A generated (or cached) corpus tree, with everything the export and the case bodies need.
public struct PaperCorpus: Sendable {
    /// Cache directory. Holds `corpus/` (the tree) and `manifest.json` (the stamp).
    public let root: URL
    public let spec: PaperCorpusSpec
    /// What was actually written, measured after writing it.
    public let textBytes: Int
    public let wideBytes: Int
    public let imageBytes: Int
    public let fnv1a64: String
    /// False when a matching tree was already on disk and was reused unchanged.
    public let regenerated: Bool

    /// The crawl root: contains exactly `spec.totalFiles` files and nothing else.
    public var treeRoot: URL { root.appendingPathComponent("corpus", isDirectory: true) }
    /// p03's index pass and p05's edit sources. Text only, so a media file cannot change the
    /// token count or the batch composition.
    public var textRoot: URL { treeRoot.appendingPathComponent("text", isDirectory: true) }
    /// Thousands of tiny files: the crawl case's cost is per-file, so it needs file count, not bytes.
    public var wideRoot: URL { treeRoot.appendingPathComponent("wide", isDirectory: true) }
    /// p12 only.
    public var imagesRoot: URL { treeRoot.appendingPathComponent("images", isDirectory: true) }

    public var totalBytes: Int { textBytes + wideBytes + imageBytes }

    /// What the export stamps. `PaperCorpusStamp` has no field for the duplication rate; a case
    /// body that cares carries it in `extraParameters` instead of it being implied here.
    public var stamp: PaperCorpusStamp {
        PaperCorpusStamp(version: PaperCorpusSpec.version, seed: PaperCorpusSpec.seed,
                         fnv1a64: fnv1a64, textFiles: spec.textFiles, textBytes: textBytes,
                         wideFiles: spec.wideFiles, images: spec.images)
    }

    // MARK: - Generation

    /// Generate the corpus, or reuse an identical one already on disk.
    ///
    /// The cache is validated by RE-HASHING the tree, not by trusting the stamp: a half-written
    /// tree from a crashed run has a stamp too, and indexing it would produce numbers under a hash
    /// that describes a different corpus. Hashing 4,616 files costs a fraction of a second and
    /// removes that failure mode entirely.
    @discardableResult
    public static func ensure(at root: URL, spec: PaperCorpusSpec = PaperCorpusSpec(),
                              progress: (String) -> Void = { _ in },
                              cancelled: () -> Bool = { false }) throws -> PaperCorpus {
        try assertSafeCorpusRoot(root)
        let fm = FileManager.default
        let tree = root.appendingPathComponent("corpus", isDirectory: true)
        let stampURL = root.appendingPathComponent("manifest.json")

        if let data = try? Data(contentsOf: stampURL),
           let stored = try? JSONDecoder().decode(StoredManifest.self, from: data),
           stored.spec == spec,
           let digest = try? manifestDigest(of: tree),
           digest.hex == stored.fnv1a64, digest.files == spec.totalFiles {
            return PaperCorpus(root: root, spec: spec, textBytes: stored.textBytes,
                               wideBytes: stored.wideBytes, imageBytes: stored.imageBytes,
                               fnv1a64: stored.fnv1a64, regenerated: false)
        }

        // Anything already there is either a different spec or damaged. Removing the whole root is
        // safe because `assertSafeCorpusRoot` has already proved it is a $TMPDIR/omni-paper-corpus-*
        // directory and nothing else.
        try? fm.removeItem(at: root)
        try fm.createDirectory(at: tree, withIntermediateDirectories: true)

        let textBytes = try writeTextFiles(spec: spec, into: tree, progress: progress, cancelled: cancelled)
        let wideBytes = try writeWideFiles(spec: spec, into: tree, progress: progress, cancelled: cancelled)
        let imageBytes = try writeImages(spec: spec, into: tree, progress: progress, cancelled: cancelled)

        progress("Hashing corpus\u{2026}")
        let digest = try manifestDigest(of: tree)
        // Loud rather than silent: a tree with the wrong file count would still hash to something,
        // and that something would look like a legitimate merge key.
        guard digest.files == spec.totalFiles else {
            throw OmniError.store("paper corpus wrote \(digest.files) files, expected \(spec.totalFiles)")
        }
        let stamp = StoredManifest(spec: spec, fnv1a64: digest.hex, textBytes: textBytes,
                                   wideBytes: wideBytes, imageBytes: imageBytes)
        try JSONEncoder().encode(stamp).write(to: stampURL)
        return PaperCorpus(root: root, spec: spec, textBytes: textBytes, wideBytes: wideBytes,
                           imageBytes: imageBytes, fnv1a64: digest.hex, regenerated: true)
    }

    /// Convenience over the run's filesystem. `PaperFS` names the corpus directory from the version
    /// tag it was constructed with, so the caller must pass `spec.directoryTag` there.
    @discardableResult
    public static func ensure(in fs: PaperFS, spec: PaperCorpusSpec = PaperCorpusSpec(),
                              progress: (String) -> Void = { _ in },
                              cancelled: () -> Bool = { false }) throws -> PaperCorpus {
        try ensure(at: fs.corpusRoot, spec: spec, progress: progress, cancelled: cancelled)
    }

    // MARK: - Addressing

    /// The corpus is content-addressed by index, so a case body can name a file without walking the
    /// tree and without depending on enumeration order.
    public func textFileURL(_ i: Int) -> URL { root.appendingPathComponent(Self.textRelativePath(i)) }
    public func wideFileURL(_ i: Int) -> URL { root.appendingPathComponent(Self.wideRelativePath(i)) }
    public func imageURL(_ i: Int) -> URL { root.appendingPathComponent(Self.imageRelativePath(i)) }

    /// Byte size of text file `i`, exactly. Chosen from the fixed table, so this is arithmetic.
    public static func textFileBytes(_ i: Int) -> Int { sizeTable[i % sizeTable.count] }

    /// Chunks the indexer will make of text file `i`, under the paper's pinned chunking.
    ///
    /// Exact, not estimated, for three reasons that all hold here and would not hold for a real
    /// corpus: the content is pure ASCII (so bytes == Characters), `FileExtractor.extractText`
    /// trims exactly the one trailing newline this generator writes, and `Indexer.chunk` walks
    /// fixed `limit`/`step` character offsets (Indexer.swift:1482-1516).
    public static func predictedChunks(_ i: Int) -> Int {
        predictedChunks(forCharacters: textFileBytes(i) - 1)
    }

    public static func predictedChunks(forCharacters n: Int,
                                       limit: Int = IndexSettings.paper.maxCharsPerChunk,
                                       overlap: Int = defaultChunkOverlap) -> Int {
        guard n > limit else { return 1 }
        let step = max(1, limit - overlap)
        return 1 + (n - limit + step - 1) / step
    }

    /// `Indexer.chunkOverlap`'s shipped default (Indexer.swift:264). Mirrored rather than read
    /// because it is an instance property and constructing an Indexer to ask would need a store.
    /// If it ever moves, `predictedChunks` is wrong and p05's file selection must be rechecked.
    public static let defaultChunkOverlap = 200

    /// Text files with at least `minChunks` chunks, in index order. p05 needs multi-chunk files: an
    /// append edit to a single-chunk file rewrites the only chunk there is, and would measure
    /// nothing. With the shipped table this returns the 25% of files at 8 KiB and above - 150 of
    /// the 600, exactly as the sizing justification claims.
    public func multiChunkFileIndices(minChunks: Int, limit: Int = .max) -> [Int] {
        var out: [Int] = []
        for i in 0 ..< spec.textFiles where Self.predictedChunks(i) >= minChunks {
            out.append(i)
            if out.count >= limit { break }
        }
        return out
    }

    // MARK: - The edit tree (p05)

    /// Copy `files` multi-chunk text files into `dir` as a flat, self-contained tree.
    ///
    /// Flat and copied rather than edited in place, for two reasons. The corpus is shared between
    /// cases and between the two arms, so editing it would make arm 2 index arm 1's leftovers; and
    /// both arms must start from byte-identical trees or the cross-arm vector diff proves nothing.
    /// The destination is created by `PaperFS.scratch(named:)`, which is what keeps this inside the
    /// run directory.
    @discardableResult
    public func stageEditTree(files: Int, minChunks: Int, into dir: URL) throws -> [URL] {
        let fm = FileManager.default
        let indices = multiChunkFileIndices(minChunks: minChunks, limit: files)
        guard indices.count == files else {
            throw OmniError.store("paper corpus has \(indices.count) files with \(minChunks)+ chunks, needs \(files)")
        }
        var out: [URL] = []
        out.reserveCapacity(indices.count)
        for i in indices {
            let src = textFileURL(i)
            // Flat names keep the copied tree independent of the corpus's directory layout, and
            // the index prefix keeps them unique and ordered.
            let dst = dir.appendingPathComponent(src.lastPathComponent)
            try? fm.removeItem(at: dst)
            try fm.copyItem(at: src, to: dst)
            out.append(dst)
        }
        return out
    }

    /// The exact bytes an edit adds. Fixed strings, so the two arms edit identically and the
    /// re-indexed vectors are comparable bit for bit.
    public static func editLine(_ edit: PaperTextEdit) -> String {
        switch edit {
        case .append: "\nAppended line for the paper benchmark.\n"
        case .mid: "\nInserted line for the paper benchmark.\n"
        }
    }

    /// Apply one edit in place. Append seeks to the end (the FSEvents save path's cheapest case);
    /// mid splices at the byte midpoint, which shifts every later chunk boundary.
    public static func applyEdit(_ edit: PaperTextEdit, to url: URL) throws {
        let line = Data(editLine(edit).utf8)
        switch edit {
        case .append:
            let fh = try FileHandle(forWritingTo: url)
            defer { try? fh.close() }
            try fh.seekToEnd()
            try fh.write(contentsOf: line)
        case .mid:
            let data = try Data(contentsOf: url)
            let cut = data.count / 2
            var out = Data(data.prefix(cut))
            out.append(line)
            out.append(data.suffix(from: cut))
            try out.write(to: url)
        }
    }

    // MARK: - Content (pure functions of the index)

    /// 100 file sizes, 30x1 KiB / 25x2 / 20x4 / 15x8 / 7x16 / 3x32, in a fixed shuffled order.
    /// Mean 4.88 KiB, so 600 files are 2,998,272 B. The shuffle is computed from the corpus seed
    /// rather than typed out, and it is what makes a `--scale` prefix a mixed sample instead of 180
    /// consecutive 1 KiB files.
    public static let sizeTable: [Int] = {
        var pool: [Int] = []
        for (kib, n) in [(1, 30), (2, 25), (4, 20), (8, 15), (16, 7), (32, 3)] {
            pool.append(contentsOf: Array(repeating: kib * 1024, count: n))
        }
        var rng = PaperCorpusRNG(0x53495A45)                    // "SIZE"
        var i = pool.count - 1
        while i > 0 { pool.swapAt(i, rng.int(i + 1)); i -= 1 }   // Fisher-Yates, seeded
        return pool
    }()

    /// Every wide-tree file is this long, so `expectedWideBytes` is arithmetic.
    public static let wideFileBytes = 64

    /// Complete contents of text file `i`, exactly `textFileBytes(i)` bytes of ASCII.
    public static func textContent(_ i: Int) -> String {
        let target = textFileBytes(i)
        let markdown = fileExtension(i) != "txt"
        var out = title(i, markdown: markdown)
        var length = out.count
        var p = 0
        while length < target {
            let para = fileParagraph(file: i, index: p)
            out += para
            length += para.count
            p += 1
        }
        // Cut to the exact target. A document truncated mid-word is realistic (extractors cap
        // files all the time) and exactness is what makes the byte count arithmetic.
        var chars = Array(out.prefix(target))
        chars[target - 1] = "\n"
        // FileExtractor trims trailing whitespace, so a file ending in "  \n" would extract fewer
        // characters than its size implies and `predictedChunks` would be wrong by a chunk.
        if chars[target - 2].isWhitespace { chars[target - 2] = "." }
        return String(chars)
    }

    /// Complete contents of wide file `i`: 64 bytes, one line, real words. Big enough to be a
    /// legitimate text file, small enough that the crawl case measures per-file cost and not I/O.
    public static func wideContent(_ i: Int) -> String {
        var rng = PaperCorpusRNG(stream: UInt64(i), tag: 0x57494445)   // "WIDE"
        var s = ""
        while s.count < wideFileBytes - 1 { s += (s.isEmpty ? "" : " ") + word(&rng) }
        var chars = Array(s.prefix(wideFileBytes - 1))
        if chars[wideFileBytes - 2].isWhitespace { chars[wideFileBytes - 2] = "." }
        return String(chars) + "\n"
    }

    /// Filler prose of exactly `chars` characters, for snippets on synthetic store rows. Shares the
    /// vocabulary with the file corpus so the snippet column holds the same kind of text a real row
    /// would, at the same bytes per row.
    public static func filler(characters: Int, stream: UInt64) -> String {
        guard characters > 0 else { return "" }
        var rng = PaperCorpusRNG(stream: stream, tag: 0x46494C4C)       // "FILL"
        var s = ""
        s.reserveCapacity(characters + 16)
        // Length tracked rather than re-counted: this runs once per row, and p10 builds 120,000 of
        // them. ASCII throughout, so the character count is the byte count.
        var length = 0
        while length < characters {
            if length > 0 { s.append(" "); length += 1 }
            let w = word(&rng)
            s.append(w)
            length += w.count
        }
        return String(s.prefix(characters))
    }

    public static func fileExtension(_ i: Int) -> String { i % 4 == 3 ? "txt" : "md" }

    public static func textRelativePath(_ i: Int) -> String {
        var rng = PaperCorpusRNG(stream: UInt64(i), tag: 0x534C5547)    // "SLUG"
        let slug = "\(contentWord(&rng))-\(contentWord(&rng))"
        return String(format: "corpus/text/d%02d/%04d-%@.%@", i / 50, i, slug, fileExtension(i))
    }

    public static func wideRelativePath(_ i: Int) -> String {
        String(format: "corpus/wide/w%02d/n%04d.md", i / 100, i)
    }

    public static func imageRelativePath(_ i: Int) -> String {
        String(format: "corpus/images/img%02d.png", i)
    }

    // MARK: - Prose

    /// 512 English words, most frequent first. The first `headWords` are the function-word head
    /// that carries most of real text's token mass; the tail is document, office, technical and
    /// everyday content vocabulary, so a chunk reads like a document rather than like a word list.
    public static let vocabulary: [String] = vocabularySource
        .split(separator: " ", omittingEmptySubsequences: true).map(String.init)

    /// Size of the high-frequency head. Drawn from 60% of the time; see the Zipf note at the top.
    public static let headWords = 96

    private static func word(_ rng: inout PaperCorpusRNG) -> String {
        let r = rng.next()
        let head = r % 100 < 60
        let span = head ? headWords : vocabulary.count
        return vocabulary[Int((r >> 16) % UInt64(span))]
    }

    /// A content word only. Titles and filenames are drawn from the tail: a real document is not
    /// called "If So Has", and a heading full of function words would also give the first chunk of
    /// every file an unrepresentative token mix.
    private static func contentWord(_ rng: inout PaperCorpusRNG) -> String {
        vocabulary[headWords + rng.int(vocabulary.count - headWords)]
    }

    private static func title(_ i: Int, markdown: Bool) -> String {
        var rng = PaperCorpusRNG(stream: UInt64(i), tag: 0x5449544C)    // "TITL"
        let n = rng.range(3, 6)
        var words: [String] = []
        for _ in 0 ..< n { words.append(contentWord(&rng)) }
        let text = words.map { $0.prefix(1).uppercased() + $0.dropFirst() }.joined(separator: " ")
        return (markdown ? "# " : "") + text + "\n\n"
    }

    private static func sentence(_ rng: inout PaperCorpusRNG) -> String {
        let n = rng.range(8, 20)
        var words: [String] = []
        words.reserveCapacity(n)
        for _ in 0 ..< n { words.append(word(&rng)) }
        var text = words.joined(separator: " ")
        // Internal punctuation on the longer sentences: real prose has it, and the tokenizer emits
        // a separate token for it, so leaving it out would bias the token count low.
        if n >= 12 {
            let at = words[0 ..< (n / 3 + 1)].joined(separator: " ").count
            text = String(text.prefix(at)) + "," + String(text.dropFirst(at))
        }
        return text.prefix(1).uppercased() + String(text.dropFirst()) + (rng.int(20) == 0 ? "?" : ".")
    }

    /// Paragraph `p` of file `i`, as a pure function of both. Purity is what makes the duplicate
    /// injection below possible without generating the donor file: a donor paragraph can simply be
    /// asked for.
    private static func paragraph(file i: Int, index p: Int) -> String {
        var rng = PaperCorpusRNG(stream: UInt64(i), UInt64(p), tag: 0x50415241)   // "PARA"
        let sentences = rng.range(4, 7)
        var out = ""
        for s in 0 ..< sentences {
            out += sentence(&rng)
            if s < sentences - 1 { out += " " }
        }
        return out + "\n\n"
    }

    /// Paragraph `p` as it appears IN file `i`: usually its own, sometimes an earlier file's.
    ///
    /// The donor is always an earlier file, and always its paragraph 0 or 1 - the only two indices
    /// every file is long enough to actually contain (the smallest file is 1 KiB and a paragraph
    /// averages ~430 characters). Copying paragraph 7 of a 1 KiB file would copy text that exists
    /// nowhere on disk, and the "duplication rate" would then be a rate of nothing.
    private static func fileParagraph(file i: Int, index p: Int) -> String {
        guard i > 0, p > 0 else { return paragraph(file: i, index: p) }
        var rng = PaperCorpusRNG(stream: UInt64(i), UInt64(p), tag: 0x44555020)   // "DUP "
        guard rng.int(1000) < duplicatePermille else { return paragraph(file: i, index: p) }
        return paragraph(file: rng.int(i), index: rng.int(2))
    }

    /// Read once from the shipped spec: the injection rate is part of the corpus identity, so it is
    /// a constant of the generator rather than a per-call argument that could differ between the
    /// file that was written and the file that was expected.
    private static let duplicatePermille = PaperCorpusSpec().duplicateParagraphPermille

    // MARK: - Writing

    private static func writeTextFiles(spec: PaperCorpusSpec, into tree: URL,
                                       progress: (String) -> Void, cancelled: () -> Bool) throws -> Int {
        let fm = FileManager.default
        for d in 0 ... max(0, (spec.textFiles - 1) / 50) {
            try fm.createDirectory(at: tree.appendingPathComponent(String(format: "text/d%02d", d), isDirectory: true),
                                   withIntermediateDirectories: true)
        }
        var bytes = 0
        for i in 0 ..< spec.textFiles {
            if i % 32 == 0 {
                if cancelled() { throw CancellationError() }
                progress("Generating corpus text \(i)/\(spec.textFiles)")
            }
            let data = Data(textContent(i).utf8)
            try data.write(to: tree.appendingPathComponent(String(textRelativePath(i).dropFirst("corpus/".count))))
            bytes += data.count
        }
        return bytes
    }

    private static func writeWideFiles(spec: PaperCorpusSpec, into tree: URL,
                                       progress: (String) -> Void, cancelled: () -> Bool) throws -> Int {
        let fm = FileManager.default
        for d in 0 ... max(0, (spec.wideFiles - 1) / 100) {
            try fm.createDirectory(at: tree.appendingPathComponent(String(format: "wide/w%02d", d), isDirectory: true),
                                   withIntermediateDirectories: true)
        }
        var bytes = 0
        for i in 0 ..< spec.wideFiles {
            if i % 256 == 0 {
                if cancelled() { throw CancellationError() }
                progress("Generating corpus tree \(i)/\(spec.wideFiles)")
            }
            let data = Data(wideContent(i).utf8)
            try data.write(to: tree.appendingPathComponent(String(wideRelativePath(i).dropFirst("corpus/".count))))
            bytes += data.count
        }
        return bytes
    }

    private static func writeImages(spec: PaperCorpusSpec, into tree: URL,
                                    progress: (String) -> Void, cancelled: () -> Bool) throws -> Int {
        let fm = FileManager.default
        try fm.createDirectory(at: tree.appendingPathComponent("images", isDirectory: true),
                               withIntermediateDirectories: true)
        var bytes = 0
        for i in 0 ..< spec.images {
            if cancelled() { throw CancellationError() }
            progress("Generating corpus images \(i)/\(spec.images)")
            let png = PaperPNG.encodeRGB8(paintImage(i, side: spec.imagePixels),
                                          width: spec.imagePixels, height: spec.imagePixels)
            try png.write(to: tree.appendingPathComponent(String(imageRelativePath(i).dropFirst("corpus/".count))))
            bytes += png.count
        }
        return bytes
    }

    /// Row-major RGB8 pixels for image `i`: a horizontal hue ramp shaded vertically, plus
    /// `2 + i % 5` opaque squares at index-derived positions. Same visual recipe as the frame
    /// painter in omni-verify's `writeMP4`, but written with integer arithmetic only, so the bytes
    /// are identical on every machine and in every optimisation mode.
    public static func paintImage(_ i: Int, side: Int) -> [UInt8] {
        var px = [UInt8](repeating: 0, count: side * side * 3)
        let span = max(1, side - 1)
        for y in 0 ..< side {
            let shade = 128 + (127 * y) / span
            for x in 0 ..< side {
                let (r, g, b) = hue(((x * 1536) / side + i * 96) % 1536)
                let o = (y * side + x) * 3
                px[o] = UInt8((Int(r) * shade) / 255)
                px[o + 1] = UInt8((Int(g) * shade) / 255)
                px[o + 2] = UInt8((Int(b) * shade) / 255)
            }
        }
        let box = min(80, side)
        for k in 0 ..< (2 + i % 5) {
            let x0 = (i * 37 + k * 61) % max(1, side - box)
            let y0 = (i * 23 + k * 47) % max(1, side - box)
            let c: (UInt8, UInt8, UInt8) = k % 2 == 0 ? (255, 255, 255) : (24, 24, 32)
            for y in y0 ..< min(side, y0 + box) {
                for x in x0 ..< min(side, x0 + box) {
                    let o = (y * side + x) * 3
                    px[o] = c.0; px[o + 1] = c.1; px[o + 2] = c.2
                }
            }
        }
        return px
    }

    /// Six-segment integer hue ramp over 0..<1536. No floating point, no colour management.
    private static func hue(_ t: Int) -> (UInt8, UInt8, UInt8) {
        let k = UInt8(truncatingIfNeeded: t % 256)
        switch t / 256 {
        case 0: return (255, k, 0)
        case 1: return (255 &- k, 255, 0)
        case 2: return (0, 255, k)
        case 3: return (0, 255 &- k, 255)
        case 4: return (k, 0, 255)
        default: return (255, 0, 255 &- k)
        }
    }

    // MARK: - Manifest

    private struct StoredManifest: Codable {
        let spec: PaperCorpusSpec
        let fnv1a64: String
        let textBytes: Int
        let wideBytes: Int
        let imageBytes: Int
    }

    /// FNV-1a-64 over `<relative path>|<byte length>|<digest of every byte>` for every file in the
    /// tree, sorted by path.
    ///
    /// EVERY byte, not a prefix. A prefix digest plus the length is cheaper and was the first
    /// design, and it is wrong: this generator writes files of exact, fixed sizes whose first 64
    /// bytes are the title line, so a bug that changed the body of every paragraph in the corpus
    /// left the hash bit-identical - which is precisely the drift the hash exists to catch. The
    /// full read costs about 16 MB and a fraction of a second, once per run.
    public static func manifestDigest(of tree: URL) throws -> (hex: String, files: Int, bytes: Int) {
        let fm = FileManager.default
        let base = tree.standardizedFileURL.path
        guard let walker = fm.enumerator(at: tree, includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey]) else {
            throw OmniError.store("paper corpus tree is not readable: \(base)")
        }
        var lines: [String] = []
        var total = 0
        for case let url as URL in walker {
            let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
            guard values?.isRegularFile == true else { continue }
            let path = url.standardizedFileURL.path
            let relative = path.hasPrefix(base + "/") ? String(path.dropFirst(base.count + 1)) : path
            let size = values?.fileSize ?? 0
            total += size
            let body = (try? Data(contentsOf: url, options: .mappedIfSafe)) ?? Data()
            lines.append(String(format: "%@|%d|%016llx", relative, size, fnv1a64(body)))
        }
        lines.sort()
        return (PaperReport.fnv1a64Hex(lines.joined(separator: "\n")), lines.count, total)
    }

    /// FNV-1a-64 over raw bytes. `PaperReport.fnv1a64Hex` is the same function over a String's
    /// UTF-8; PNG bytes are not valid UTF-8, so they need this one.
    static func fnv1a64(_ data: Data) -> UInt64 {
        var h: UInt64 = 0xcbf2_9ce4_8422_2325
        data.withUnsafeBytes { raw in
            for b in raw { h = (h ^ UInt64(b)) &* 1_099_511_628_211 }
        }
        return h
    }

    // MARK: - The non-destructive guarantee

    /// The corpus lives outside the run directory, so `PaperFS`'s preconditions do not cover it and
    /// it needs its own. `ensure` deletes this directory wholesale on a spec change, which is only
    /// acceptable because of the two checks below.
    private static func assertSafeCorpusRoot(_ url: URL) throws {
        let path = url.standardizedFileURL.path
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true).standardizedFileURL.path
        precondition(path.hasPrefix(tmp + "/"),
                     "paper corpus root is not under the temporary directory: \(path)")
        precondition(url.lastPathComponent.hasPrefix("omni-paper-corpus-"),
                     "paper corpus root is not an omni-paper-corpus directory: \(path)")
    }
}

// MARK: - RNG

/// Seeded, index-addressed randomness: splitmix64 to derive a state from the stream's coordinates,
/// then xorshift64 to draw from it. The same recipe as omni-verify's benches (main.swift:3443-3447),
/// with the seeding made explicit so a stream depends on WHICH stream it is and never on how many
/// values were drawn before it.
public struct PaperCorpusRNG: Sendable {
    private var state: UInt64

    /// A stream identified by its coordinates plus a four-character tag, so two different uses of
    /// the same (file, paragraph) pair - the text and the duplicate decision, say - are independent.
    public init(stream a: UInt64, _ b: UInt64 = 0, tag: UInt64 = 0) {
        var mixed = PaperCorpusSpec.seed
        mixed ^= a &* 0x9E37_79B9_7F4A_7C15
        mixed ^= b &* 0xD1B5_4A32_D192_ED03
        mixed ^= tag &* 0xC2B2_AE3D_27D4_EB4F
        state = Self.splitmix64(mixed) | 1        // xorshift64 is stuck at zero
    }

    public init(_ tag: UInt64) { self.init(stream: 0, tag: tag) }

    public mutating func next() -> UInt64 {
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state
    }

    /// Uniform in 0..<n.
    public mutating func int(_ n: Int) -> Int { n <= 1 ? 0 : Int(next() % UInt64(n)) }
    /// Uniform in lo...hi.
    public mutating func range(_ lo: Int, _ hi: Int) -> Int { lo + int(hi - lo + 1) }

    static func splitmix64(_ x: UInt64) -> UInt64 {
        var z = x &+ 0x9E37_79B9_7F4A_7C15
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

// MARK: - PNG

/// A deterministic PNG writer: 8-bit truecolour, filter 0 on every row, and stored (uncompressed)
/// deflate blocks.
///
/// Stored blocks are the point. A compressed stream would be smaller, but every compressor's output
/// depends on its version and its heuristics, and the corpus hash must not. The cost is exact and
/// arithmetic: `height * (1 + 3 * width)` bytes of image data plus 5 bytes per 65,535-byte block,
/// so a 512x512 image is 787,065 bytes. That is a fair trade for a hash that means the same thing
/// on every machine.
enum PaperPNG {
    static func encodeRGB8(_ pixels: [UInt8], width: Int, height: Int) -> Data {
        precondition(pixels.count == width * height * 3, "PNG pixel buffer is not width*height*3")
        var raw = [UInt8]()
        raw.reserveCapacity(height * (1 + width * 3))
        for y in 0 ..< height {
            raw.append(0)                                  // filter type 0 (None)
            raw.append(contentsOf: pixels[(y * width * 3) ..< ((y + 1) * width * 3)])
        }

        var out = Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
        var ihdr = be32(UInt32(width)) + be32(UInt32(height))
        ihdr += [8, 2, 0, 0, 0]                            // depth 8, truecolour, deflate, no filter, no interlace
        out.append(chunk("IHDR", ihdr))
        out.append(chunk("IDAT", zlibStored(raw)))
        out.append(chunk("IEND", []))
        return out
    }

    private static func zlibStored(_ raw: [UInt8]) -> [UInt8] {
        var out: [UInt8] = [0x78, 0x01]                    // CM=deflate/32K window, FCHECK valid
        var offset = 0
        repeat {
            let n = min(65535, raw.count - offset)
            let final: UInt8 = offset + n >= raw.count ? 1 : 0
            out.append(final)                              // BTYPE 00 = stored
            out.append(contentsOf: [UInt8(n & 0xFF), UInt8(n >> 8)])
            out.append(contentsOf: [UInt8(~n & 0xFF), UInt8((~n >> 8) & 0xFF)])
            out.append(contentsOf: raw[offset ..< offset + n])
            offset += n
        } while offset < raw.count
        out.append(contentsOf: be32(adler32(raw)))
        return out
    }

    private static func chunk(_ type: String, _ payload: [UInt8]) -> Data {
        let body = Array(type.utf8) + payload
        return Data(be32(UInt32(payload.count)) + body + be32(crc32(body)))
    }

    private static func be32(_ v: UInt32) -> [UInt8] {
        [UInt8(v >> 24 & 0xFF), UInt8(v >> 16 & 0xFF), UInt8(v >> 8 & 0xFF), UInt8(v & 0xFF)]
    }

    private static let crcTable: [UInt32] = (0 ..< 256).map { i in
        var c = UInt32(i)
        for _ in 0 ..< 8 { c = (c & 1) != 0 ? 0xEDB8_8320 ^ (c >> 1) : c >> 1 }
        return c
    }

    private static func crc32(_ bytes: [UInt8]) -> UInt32 {
        var c: UInt32 = 0xFFFF_FFFF
        for b in bytes { c = crcTable[Int((c ^ UInt32(b)) & 0xFF)] ^ (c >> 8) }
        return c ^ 0xFFFF_FFFF
    }

    private static func adler32(_ bytes: [UInt8]) -> UInt32 {
        var a: UInt32 = 1, b: UInt32 = 0
        for byte in bytes {
            a = (a + UInt32(byte)) % 65521
            b = (b + a) % 65521
        }
        return (b << 16) | a
    }
}

// MARK: - Vocabulary

/// 512 distinct English words, ordered most-frequent-first for the first 96 and grouped by domain
/// after that. Frozen: changing a single word changes every generated file and therefore
/// `corpus.fnv1a64`, which is exactly the point of Risk 8 in the plan.
private let vocabularySource = """
the of and to in a is that for it as was with be by on \
not this are or from at which but have an they you one had all were \
we when can there use each she do how their if will up other about out \
many then them these so some her would make like him into time has look two \
more write see number no way could people my than first been call who its now \
find long down day did get come made may part over new take only little work \
report document quarterly revenue cloud growth machine learning infrastructure budget planning engineering organization fiscal detail project \
schedule milestone deliverable stakeholder requirement analysis design implementation testing deployment maintenance architecture component module interface protocol \
service database query index cache memory storage network latency throughput bandwidth capacity scaling cluster node container \
pipeline workflow automation monitoring alert dashboard metric baseline threshold regression benchmark experiment hypothesis evidence measurement observation \
sample dataset feature label model training evaluation validation accuracy precision recall error variance distribution correlation parameter \
gradient optimization convergence iteration batch epoch token sentence paragraph chapter summary abstract introduction conclusion reference citation \
appendix figure table chart diagram notebook script function variable constant pointer buffer thread process kernel driver \
firmware hardware software platform framework library package version release branch commit merge review approval ticket backlog \
sprint retrospective meeting agenda minutes owner deadline priority severity incident outage recovery backup restore migration rollout \
rollback config setting default override policy permission credential session request response payload header status timeout retry \
queue worker scheduler dispatch handler listener event signal trigger condition loop guard filter mapping reduce sort \
search ranking relevance embedding vector similarity neighbor candidate retrieval corpus passage snippet keyword phrase term weight \
revision draft proposal contract invoice payment vendor supplier customer account balance forecast margin expense allocation reserve \
quarter annual monthly weekly daily hourly calendar timeline duration interval phase instant lifetime horizon cadence rhythm \
team department division office building campus region country market segment channel partner alliance program portfolio initiative \
strategy objective outcome result impact benefit risk assumption constraint dependency tradeoff decision rationale context scope boundary \
interview survey questionnaire reply feedback comment suggestion complaint praise sentiment satisfaction retention churn adoption engagement usage \
article journal magazine newspaper column editorial headline byline caption footnote glossary bibliography manuscript publisher edition volume \
kitchen garden window doorway hallway staircase balcony rooftop courtyard fountain bridge harbour railway station tunnel terminal \
morning evening afternoon midnight sunrise sunset weather climate season autumn winter summer spring rainfall temperature humidity \
coffee bread cheese pepper garlic onion tomato potato carrot lemon orange apple banana berry walnut almond \
forest meadow valley mountain river ocean island desert canyon glacier prairie swamp jungle savanna tundra reef \
teacher student lecture seminar textbook classroom homework examination degree diploma scholarship faculty dormitory tuition semester curriculum \
doctor patient hospital clinic nurse surgery diagnosis treatment medicine therapy remedy symptom vaccine allergy fitness nutrition \
engine turbine battery circuit sensor antenna satellite telescope microscope laser magnet crystal polymer alloy concrete timber \
musician painter sculptor novelist poet dancer actor director producer gallery museum theatre concert festival exhibit rehearsal
"""
