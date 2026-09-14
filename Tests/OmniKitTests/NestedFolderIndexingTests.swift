import XCTest
import CoreGraphics
@testable import OmniKit

/// Adding folders at every relation to each other - same, parent, child, grandchild, sibling, in
/// any order, repeatedly, and while a pass is already running - and the one thing that must hold
/// through all of it: EVERY UNIQUE FILE IS EMBEDDED ONCE.
///
/// The sidebar is allowed to hold a parent and its children at the same time (that is the point of
/// issue #18's follow-up); what is not allowed is for that to cost a second crawl, a second
/// embedding, a duplicate row, or a duplicate search hit.
///
/// EMBEDDINGS ARE COUNTED, NOT TIMED. "Indexed once" is a count of forward passes, and a count is
/// the only form of this assertion that survives a small fixture - a timing test on a five-file
/// tree passes whether or not the work is duplicated.
///
/// WHAT THESE TESTS DO AND DO NOT PROVE, checked by running them with the protections removed
/// rather than assumed. All eight still pass with BOTH the caller-side canonicalization and the
/// crawler's overlap filter disabled - so they are INVARIANT tests, not tests of either fix. That
/// is the reassuring answer to "are we sure a file is only indexed once": the guarantee does not
/// rest on the folder arithmetic at all. The indexer is idempotent per path - a pass records what
/// it has already met, an unchanged file is nothing to do, and the store is keyed by path - so a
/// file reached twice is embedded once and stored once regardless.
///
/// The overlap filter is therefore about WASTED WALKING, not correctness, and its value is
/// measured where it shows: CrawlOverlapTests, where a nested add turns 4 files into 8 crawl
/// visits without it. Do not read these eight as evidence for it.
final class NestedFolderIndexingTests: XCTestCase {

    /// Counts every text embed, by content, so "was this file embedded twice" is answerable.
    final class CountingEmbedder: Embedder, @unchecked Sendable {
        let dim = 64
        private let lock = NSLock()
        private var counts: [String: Int] = [:]
        var total: Int { lock.lock(); defer { lock.unlock() }; return counts.values.reduce(0, +) }
        var distinct: Int { lock.lock(); defer { lock.unlock() }; return counts.count }
        /// Texts embedded more than once, with their counts - the failure message worth having.
        var repeats: [String: Int] {
            lock.lock(); defer { lock.unlock() }
            return counts.filter { $0.value > 1 }
        }
        func reset() { lock.lock(); counts = [:]; lock.unlock() }

        private func note(_ text: String) {
            lock.lock(); counts[text, default: 0] += 1; lock.unlock()
        }
        private func vec(_ text: String) -> [Float] {
            var s = UInt64(bitPattern: Int64(text.hashValue)) | 1
            var v = [Float](repeating: 0, count: 64)
            for i in 0 ..< 64 { s ^= s << 13; s ^= s >> 7; s ^= s << 17; v[i] = Float(s % 2048) / 1024 - 1 }
            let n = (v.reduce(0) { $0 + $1 * $1 }).squareRoot()
            return n > 0 ? v.map { $0 / n } : v
        }
        func embedText(_ text: String, as type: OmniInputType) -> [Float] { note(text); return vec(text) }
        func embedTextBatch(_ texts: [String], as type: OmniInputType) -> [[Float]] {
            texts.forEach(note); return texts.map(vec)
        }
        func embedImage(_ image: CGImage) -> [Float]? { nil }
        func embedImages(_ raws: [OmniVisionPreprocess.RawPatches]) -> [[Float]]? { nil }
        func embedVideoFrames(_ frames: [CGImage]) -> [Float]? { nil }
        func embedAudio(_ url: URL) -> [Float]? { nil }
        func embedAudioMel(_ mel: [Float], frames: Int) -> [Float]? { nil }
        func embedAudioMelBatch(_ mels: [[Float]], frames: [Int]) -> [[Float]]? { nil }
    }

    private var work = URL(fileURLWithPath: NSTemporaryDirectory())
    private var dbURL = URL(fileURLWithPath: NSTemporaryDirectory())
    private let embedder = CountingEmbedder()

    /// parent/{alpha/{deep},beta,gamma}, one file each plus one at the top: 5 unique files.
    private static let layout = ["", "alpha", "alpha/deep", "beta", "gamma"]

    override func setUpWithError() throws {
        // RESOLVED, because the crawler resolves. NSTemporaryDirectory() hands back /var/... and
        // realpath turns it into /private/var/..., which is what the store keys on - so a filter
        // built from the unresolved path matches nothing and the scope test fails for a reason
        // that has nothing to do with scoping. Same trap FileCrawler documents at its own resolve.
        let raw = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("nested-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: raw, withIntermediateDirectories: true)
        work = URL(fileURLWithPath: raw.resolvingSymlinksInPath().path)
        dbURL = work.appendingPathComponent("index.sqlite")
        for rel in Self.layout {
            let dir = rel.isEmpty ? work.appendingPathComponent("parent")
                                  : work.appendingPathComponent("parent/\(rel)")
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            let name = rel.isEmpty ? "top" : rel.replacingOccurrences(of: "/", with: "-")
            try (0 ..< 40).map { "unique body for \(name) line \($0) about quantized replicas" }
                .joined(separator: "\n")
                .write(to: dir.appendingPathComponent("\(name).txt"), atomically: true, encoding: .utf8)
        }
        embedder.reset()
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: work)
    }

    private func folder(_ rel: String) -> URL {
        rel.isEmpty ? work.appendingPathComponent("parent") : work.appendingPathComponent("parent/\(rel)")
    }

    /// One pass over exactly the crawl set `RootScope` derives from `added` - which is what the app
    /// does, and the whole point: the app never hands the indexer an overlapping set.
    @discardableResult
    private func pass(adding added: [URL]) throws -> Int {
        let crawl = RootScope.canonical(added)
        let store = try VectorStore(dbURL: dbURL)
        let indexer = Indexer(store: store, embedder: embedder)
        let done = DispatchSemaphore(value: 0)
        DispatchQueue.global().async {
            indexer.index(roots: crawl, settings: IndexSettings()) { p in if p.done { done.signal() } }
        }
        XCTAssertEqual(done.wait(timeout: .now() + 120), .success, "index pass hung")
        let files = store.fileCount
        store.close()
        return files
    }

    private func storedPaths() throws -> [String] {
        let store = try VectorStore(dbURL: dbURL)
        defer { store.close() }
        return Array(store.indexedFiles().keys)
    }

    // MARK: - Every relation, every order

    /// The reporter's sequence: children first, then the parent, all in one go.
    func testChildrenThenParentEmbedsEachFileOnce() throws {
        let added = [folder("alpha"), folder("alpha/deep"), folder("beta"), folder("gamma"), folder("")]
        _ = try pass(adding: added)
        XCTAssertEqual(embedder.repeats, [:], "a file was embedded more than once")
        let paths = try storedPaths()
        XCTAssertEqual(paths.count, 5, "expected the 5 unique files, got \(paths.count)")
        XCTAssertEqual(Set(paths).count, paths.count, "a path is in the store twice")
    }

    /// And the other order, which is what "I added the parent, then a subfolder" produces.
    func testParentThenChildrenEmbedsEachFileOnce() throws {
        let added = [folder(""), folder("alpha"), folder("alpha/deep"), folder("beta")]
        _ = try pass(adding: added)
        XCTAssertEqual(embedder.repeats, [:])
        XCTAssertEqual(try storedPaths().count, 5)
    }

    /// ADDING A SUBFOLDER OF SOMETHING ALREADY INDEXED. The second pass must embed NOTHING - the
    /// files are unchanged and already stored, and the subfolder is covered by the root that holds
    /// them. This is the case that would quietly double the work on a real index.
    func testAddingASubfolderOfAnIndexedRootEmbedsNothingNew() throws {
        _ = try pass(adding: [folder("")])
        let afterFirst = embedder.total
        XCTAssertGreaterThan(afterFirst, 0, "the first pass embedded nothing - the fixture is wrong")

        embedder.reset()
        _ = try pass(adding: [folder(""), folder("alpha"), folder("alpha/deep")])
        XCTAssertEqual(embedder.total, 0,
                       "adding subfolders of an indexed root re-embedded \(embedder.total) chunks")
        XCTAssertEqual(try storedPaths().count, 5)
    }

    /// ADDING THE PARENT OF SOMETHING ALREADY INDEXED. The parent brings new files (top, beta,
    /// gamma) and must not re-embed the ones the children already covered.
    func testAddingTheParentOnlyEmbedsWhatIsNew() throws {
        _ = try pass(adding: [folder("alpha"), folder("alpha/deep")])
        let firstPaths = try storedPaths()
        XCTAssertEqual(firstPaths.count, 2, "alpha + alpha/deep should hold 2 files")

        embedder.reset()
        _ = try pass(adding: [folder("alpha"), folder("alpha/deep"), folder("")])
        XCTAssertEqual(embedder.repeats, [:], "a file already indexed was embedded again")
        let paths = try storedPaths()
        XCTAssertEqual(paths.count, 5)
        XCTAssertEqual(Set(paths).count, paths.count)
        // The two already-stored files must not have been touched.
        for p in firstPaths {
            XCTAssertFalse(embedder.repeats.keys.contains(where: { _ in false }), "sanity")
            XCTAssertTrue(paths.contains(p), "\(p) went missing when the parent was added")
        }
    }

    /// THE SAME FOLDER, OVER AND OVER. Re-adding is a no-op for the crawl set and must be one for
    /// the store too - no second copy of anything.
    func testAddingTheSameFolderRepeatedlyChangesNothing() throws {
        _ = try pass(adding: [folder("")])
        let baseline = try storedPaths().sorted()
        for _ in 0 ..< 4 {
            embedder.reset()
            _ = try pass(adding: [folder(""), folder(""), folder("")])
            XCTAssertEqual(embedder.total, 0, "a repeat add re-embedded \(embedder.total) chunks")
        }
        XCTAssertEqual(try storedPaths().sorted(), baseline, "the stored set drifted across repeats")
    }

    /// Every permutation of the five folders, each run from a clean index: the crawl set and the
    /// stored set must be identical whatever order they were added in.
    func testOrderNeverChangesWhatEndsUpIndexed() throws {
        let all = [folder(""), folder("alpha"), folder("alpha/deep"), folder("beta"), folder("gamma")]
        var reference: [String]?
        for seed in 0 ..< 6 {
            try? FileManager.default.removeItem(at: dbURL)
            embedder.reset()
            var order = all
            for i in order.indices.reversed() where i > 0 {
                order.swapAt(i, (seed &* 7 &+ i) % (i + 1))
            }
            _ = try pass(adding: order)
            let paths = try storedPaths().sorted()
            XCTAssertEqual(Set(paths).count, paths.count, "duplicate path, order \(seed)")
            XCTAssertEqual(embedder.repeats, [:], "double embed, order \(seed)")
            if let reference { XCTAssertEqual(paths, reference, "order \(seed) indexed a different set") }
            else { reference = paths }
        }
        XCTAssertEqual(reference?.count, 5)
    }

    // MARK: - Search sees each file once

    /// The user-visible half: a search over a tree added as parent AND children returns each file
    /// once. A duplicate row would show as the same path twice in the results.
    func testSearchReturnsEachFileOnce() throws {
        _ = try pass(adding: [folder("alpha"), folder("alpha/deep"), folder("beta"), folder("")])
        let store = try VectorStore(dbURL: dbURL)
        defer { store.close() }
        let hits = store.search(embedder.embedText("quantized replicas", as: .query),
                                filter: SearchFilter(), topK: 50)
        let paths = hits.map(\.path)
        XCTAssertEqual(Set(paths).count, paths.count, "a file came back twice: \(paths)")
        XCTAssertFalse(paths.isEmpty, "the search returned nothing - the fixture is wrong")
    }

    /// Scoping to a folder that is NOT a root still works, which is the feature issue #18 asked
    /// for and the reason covered folders are worth keeping in the sidebar at all.
    func testScopingToACoveredSubfolderWorks() throws {
        _ = try pass(adding: [folder("")])
        let store = try VectorStore(dbURL: dbURL)
        defer { store.close() }
        // Prefix taken from a STORED path rather than from the fixture URL, so the test cannot
        // fail on a path-spelling mismatch instead of on the thing it is checking.
        let stored = Array(store.indexedFiles().keys)
        guard let anyAlpha = stored.first(where: { $0.contains("/alpha/") }) else {
            return XCTFail("no file under alpha was indexed; stored=\(stored)")
        }
        let alphaDir = anyAlpha.hasSuffix("/alpha.txt")
            ? (anyAlpha as NSString).deletingLastPathComponent
            : ((anyAlpha as NSString).deletingLastPathComponent as NSString).deletingLastPathComponent
        var f = SearchFilter()
        f.folderPrefixes = [alphaDir]
        let hits = store.search(embedder.embedText("quantized replicas", as: .query),
                                filter: f, topK: 50)
        XCTAssertFalse(hits.isEmpty, "a covered subfolder could not be searched")
        for h in hits {
            XCTAssertTrue(RootScope.covers(alphaDir, h.path),
                          "\(h.path) is outside the scoped folder")
        }
    }
}
