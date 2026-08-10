import XCTest
import CoreGraphics
@testable import OmniKit

/// Every mutation a user can perform, driven through the REAL Indexer against a real filesystem,
/// against the lean-DB (coverage) and 1-bit-first-stage regime.
///
/// The store-level suite proves each store API keeps its bookkeeping. This proves the thing above
/// it: that the operations a user actually performs - add, edit, rename, move, delete, at file AND
/// folder level - reach those APIs in a shape that keeps it. Renames are the reason this exists:
/// nothing in the store is called "rename", it arrives as a delete of one path plus an add of
/// another, and whether those two halves stay consistent is a property of the Indexer, not the
/// store. A folder rename is the same thing multiplied by every file beneath it.
final class MutationLifecycleTests: XCTestCase {
    /// Distinct per path so a vector that ends up under the wrong path is visible, not merely
    /// "some unit vector".
    final class PathEmbedder: Embedder, @unchecked Sendable {
        let dim = 64
        private func vec(_ text: String) -> [Float] {
            var s = UInt64(bitPattern: Int64(text.hashValue)) | 1
            var v = [Float](repeating: 0, count: 64)
            for i in 0 ..< 64 {
                s ^= s << 13; s ^= s >> 7; s ^= s << 17
                v[i] = Float(s % 2048) / 1024 - 1
            }
            let n = (v.reduce(0) { $0 + $1 * $1 }).squareRoot()
            return n > 0 ? v.map { $0 / n } : v
        }
        func embedText(_ text: String, as type: OmniInputType) -> [Float] { vec(text) }
        func embedTextBatch(_ texts: [String], as type: OmniInputType) -> [[Float]] { texts.map { vec($0) } }
        func embedImage(_ image: CGImage) -> [Float]? { nil }
        func embedImages(_ raws: [OmniVisionPreprocess.RawPatches]) -> [[Float]]? { nil }
        func embedVideoFrames(_ frames: [CGImage]) -> [Float]? { nil }
        func embedAudio(_ url: URL) -> [Float]? { nil }
        func embedAudioMel(_ mel: [Float], frames: Int) -> [Float]? { nil }
        func embedAudioMelBatch(_ mels: [[Float]], frames: [Int]) -> [[Float]]? { nil }
    }

    private var savedQuant: Int?

    override func setUp() {
        super.setUp()
        savedQuant = VectorStore.quantBaseOverride
        VectorStore.quantBaseOverride = VectorStore.scanBits   // coverage + the 1-bit tier both live
    }

    override func tearDown() {
        VectorStore.quantBaseOverride = savedQuant
        super.tearDown()
    }

    private func makeRoot() throws -> URL {
        var dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-mutlife-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        if let rp = realpath(dir.path, nil) { dir = URL(fileURLWithPath: String(cString: rp), isDirectory: true); free(rp) }
        return dir
    }

    private func write(_ url: URL, _ body: String) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try body.write(to: url, atomically: true, encoding: .utf8)
    }

    /// One full pass, then settle so coverage advances and the blobs it covers are actually dropped.
    private func indexAndSettle(_ dbURL: URL, _ root: URL, rounds: Int = 3) throws {
        let store = try VectorStore(dbURL: dbURL)
        let indexer = Indexer(store: store, embedder: PathEmbedder())
        let done = DispatchSemaphore(value: 0)
        DispatchQueue.global().async { indexer.index(roots: [root], settings: IndexSettings()) { p in if p.done { done.signal() } } }
        XCTAssertEqual(done.wait(timeout: .now() + 120), .success, "initial index hung")
        store.close()
        for _ in 0 ..< rounds { let s = try VectorStore(dbURL: dbURL); s.close() }
    }

    private func update(_ dbURL: URL, _ paths: [String]) throws {
        let store = try VectorStore(dbURL: dbURL)
        let indexer = Indexer(store: store, embedder: PathEmbedder())
        indexer.update(paths: paths, settings: IndexSettings())
        store.close()
    }

    /// MUST HOLD AFTER EVERY MUTATION, with no exceptions and no settling period: the coverage
    /// bookkeeping is self-consistent, and every path the index still claims retrieves its OWN
    /// content. The second half is what a shifted slot breaks - the index would happily return a
    /// real path carrying its neighbour's vector.
    @discardableResult
    private func assertInvariant(_ dbURL: URL, _ label: String,
                                 file: StaticString = #filePath, line: UInt = #line) throws -> Set<String> {
        let store = try VectorStore(dbURL: dbURL)
        defer { store.close() }
        if let bad = store.coverageAudit() {
            XCTFail("\(label): coverage invariant broken - \(bad)", file: file, line: line)
        }
        let indexed = Set(store.indexedFiles().keys)
        for p in indexed.sorted().prefix(25) {
            guard let text = try? String(contentsOfFile: p, encoding: .utf8) else { continue }
            let q = PathEmbedder().embedText(text, as: .passage)
            let hits = store.search(q, filter: SearchFilter(), topK: 5)
            // SCORE, not path. Between a folder event and the reconcile that follows it, a moved
            // file is legitimately indexed under both its old and its new path with byte-identical
            // content - they tie at 1.0 and which one surfaces is arbitrary. Asserting the path
            // would be asserting the tie-break. What must hold either way is that the vector the
            // index returns IS this content: a slot that shifted returns some other file's vector,
            // and that shows up as a score nowhere near 1.
            XCTAssertEqual(hits.first?.score ?? 0, 1.0, accuracy: 0.02,
                           "\(label): query for \(p) retrieved a vector that is not its content", file: file, line: line)
        }
        return indexed
    }

    /// The index matches the filesystem exactly. Asserted only where the system actually promises
    /// it. A FILE event names the file, so update() resolves it immediately; a DIRECTORY event
    /// names only the directory, and a vanished directory has no stored rows of its own - the files
    /// beneath it are reconciled by the next full pass instead. That is pre-existing behaviour (the
    /// app calls deleteUnderFolder for folders it removes from the sidebar, and runs a catch-up
    /// pass at launch), so the tests hold folder events to eventual agreement and file events to
    /// immediate agreement, rather than pretending both are the same.
    private func assertMatchesDisk(_ dbURL: URL, _ root: URL, _ label: String,
                                   file: StaticString = #filePath, line: UInt = #line) throws {
        let indexed = try assertInvariant(dbURL, label, file: file, line: line)
        var onDisk = Set<String>()
        if let en = FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil) {
            for case let u as URL in en where u.pathExtension == "txt" { onDisk.insert(u.path) }
        }
        XCTAssertEqual(indexed.subtracting(onDisk), [], "\(label): indexed files that no longer exist", file: file, line: line)
        XCTAssertEqual(onDisk.subtracting(indexed), [], "\(label): files on disk that are not indexed", file: file, line: line)
    }

    /// A folder-level change reaches the store either as deleteUnderFolder (the sidebar path) or as
    /// a reconcile pass (the watcher path). This is the latter.
    private func reconcile(_ dbURL: URL, _ root: URL) throws {
        try indexAndSettle(dbURL, root, rounds: 2)
    }

    func testAddEditRenameMoveDeleteAtFileAndFolderLevel() throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let dbURL = root.appendingPathComponent("index.sqlite")

        // ADD: a nested tree, enough rows that coverage has something to cover.
        for d in 0 ..< 4 {
            for f in 0 ..< 12 {
                try write(root.appendingPathComponent("dir\(d)/file\(f).txt"), "document \(d)-\(f) about distributed search")
            }
        }
        try indexAndSettle(dbURL, root)
        try assertMatchesDisk(dbURL, root, "after add")

        // EDIT CONTENT: same path, new bytes - the re-embed path, and the one that leaves a hole
        // behind for every covered row it replaces.
        var edited: [String] = []
        for f in 0 ..< 6 {
            let u = root.appendingPathComponent("dir0/file\(f).txt")
            try write(u, "rewritten \(f) concerning vector quantization and recall")
            edited.append(u.path)
        }
        try update(dbURL, edited)
        try assertMatchesDisk(dbURL, root, "after edit")

        // RENAME A FILE: nothing in the store is called rename - it arrives as a delete of the old
        // path and an add of the new one, and the index must end up with exactly one of them.
        let oldFile = root.appendingPathComponent("dir1/file0.txt")
        let newFile = root.appendingPathComponent("dir1/renamed0.txt")
        try FileManager.default.moveItem(at: oldFile, to: newFile)
        try update(dbURL, [oldFile.path, newFile.path])
        try assertMatchesDisk(dbURL, root, "after file rename")

        // MOVE A FILE ACROSS FOLDERS: same shape, different parent.
        let moveFrom = root.appendingPathComponent("dir1/file1.txt")
        let moveTo = root.appendingPathComponent("dir2/moved1.txt")
        try FileManager.default.moveItem(at: moveFrom, to: moveTo)
        try update(dbURL, [moveFrom.path, moveTo.path])
        try assertMatchesDisk(dbURL, root, "after file move")

        // RENAME A FOLDER: every file beneath it changes path at once. The watcher sees the
        // directory, so the update is driven by the two directory paths, not the files.
        let oldDir = root.appendingPathComponent("dir3")
        let newDir = root.appendingPathComponent("dir3-renamed")
        try FileManager.default.moveItem(at: oldDir, to: newDir)
        try update(dbURL, [oldDir.path, newDir.path])
        try assertInvariant(dbURL, "after folder rename (pre-reconcile)")
        try reconcile(dbURL, root)
        try assertMatchesDisk(dbURL, root, "after folder rename")

        // MOVE A FOLDER INSIDE ANOTHER.
        let nestFrom = root.appendingPathComponent("dir2")
        let nestTo = root.appendingPathComponent("dir0/nested2")
        try FileManager.default.moveItem(at: nestFrom, to: nestTo)
        try update(dbURL, [nestFrom.path, nestTo.path])
        try assertInvariant(dbURL, "after folder move (pre-reconcile)")
        try reconcile(dbURL, root)
        try assertMatchesDisk(dbURL, root, "after folder move")

        // DELETE A FILE.
        let gone = root.appendingPathComponent("dir0/file7.txt")
        try FileManager.default.removeItem(at: gone)
        try update(dbURL, [gone.path])
        try assertMatchesDisk(dbURL, root, "after file delete")

        // DELETE A FOLDER with everything under it.
        let goneDir = root.appendingPathComponent("dir3-renamed")
        try FileManager.default.removeItem(at: goneDir)
        try update(dbURL, [goneDir.path])
        try assertInvariant(dbURL, "after folder delete (pre-reconcile)")
        try reconcile(dbURL, root)
        try assertMatchesDisk(dbURL, root, "after folder delete")

        // ADD BACK at a path that was deleted: its old slot is a hole and the new rows append past
        // the covered prefix, so one file has rows of both kinds.
        try write(gone, "resurrected content about metal kernels")
        try update(dbURL, [gone.path])
        try assertMatchesDisk(dbURL, root, "after re-add")

        // A FULL PASS after all of it: reconcile drops anything stale and must not disturb the rest.
        try indexAndSettle(dbURL, root, rounds: 2)
        try assertMatchesDisk(dbURL, root, "after full reconcile")
    }

    /// The same lifecycle with filenames macOS actually produces: NFD combining marks, spaces, and
    /// non-ASCII. Folder removal compares BYTES in SQL and used to compare grapheme clusters in
    /// memory, which disagreed for exactly these names.
    func testLifecycleWithNFDAndNonASCIINames() throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let dbURL = root.appendingPathComponent("index.sqlite")

        let names = ["\u{0301}leading-mark.txt", "cafe\u{0301}.txt", "café.txt",
                     "with space.txt", "日本語.txt", "emoji-\u{1F600}.txt"]
        for (i, n) in names.enumerated() {
            try write(root.appendingPathComponent("docs/\(n)"), "content number \(i) about embeddings")
        }
        for f in 0 ..< 20 {
            try write(root.appendingPathComponent("bulk/b\(f).txt"), "bulk filler \(f)")
        }
        try indexAndSettle(dbURL, root)
        try assertMatchesDisk(dbURL, root, "nfd: after add")

        // Rename one of the awkward ones.
        let from = root.appendingPathComponent("docs/\(names[0])")
        let to = root.appendingPathComponent("docs/renamed-\u{0301}mark.txt")
        try FileManager.default.moveItem(at: from, to: to)
        try update(dbURL, [from.path, to.path])
        try assertMatchesDisk(dbURL, root, "nfd: after rename")

        // And delete the whole folder that holds them.
        let docs = root.appendingPathComponent("docs")
        try FileManager.default.removeItem(at: docs)
        try update(dbURL, [docs.path])
        try assertInvariant(dbURL, "nfd: after folder delete (pre-reconcile)")
        try reconcile(dbURL, root)
        try assertMatchesDisk(dbURL, root, "nfd: after folder delete")
    }
}
