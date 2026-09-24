import XCTest
import CoreGraphics
import ImageIO
@testable import OmniKit

/// Regression tests for the reconcile-deletion scope: a pass given a SUBSET of the user's roots
/// (the add-folder catch-up pass, or a full pass with some roots paused) must never delete
/// indexed files belonging to roots it was not asked to crawl. The bug: reconcile compared
/// `seen` (this pass's crawl) against `store.indexedFiles()` (the WHOLE store), so adding a new
/// folder from the sidebar wiped every other folder's index.
final class IndexerReconcileTests: XCTestCase {
    final class UnitTextEmbedder: Embedder, @unchecked Sendable {
        let dim = 8
        private func unit() -> [Float] { var v = [Float](repeating: 0, count: 8); v[0] = 1; return v }
        func embedText(_ text: String, as type: OmniInputType) -> [Float] { unit() }
        func embedTextBatch(_ texts: [String], as type: OmniInputType) -> [[Float]] { texts.map { _ in unit() } }
        func embedImage(_ image: CGImage) -> [Float]? { nil }
        func embedImages(_ raws: [OmniVisionPreprocess.RawPatches]) -> [[Float]]? { nil }
        func embedVideoFrames(_ frames: [CGImage]) -> [Float]? { nil }
        func embedAudio(_ url: URL) -> [Float]? { nil }
        func embedAudioMel(_ mel: [Float], frames: Int) -> [Float]? { nil }
        func embedAudioMelBatch(_ mels: [[Float]], frames: [Int]) -> [[Float]]? { nil }
    }

    private func makeRoot(_ name: String, files: Int) throws -> URL {
        var dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-reconcile-\(name)-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        // The crawler stores the enumerator's paths (/private/var/...), so the root the assertions
        // use must match that form. URL.resolvingSymlinksInPath() strips /private (the opposite),
        // so resolve via realpath.
        if let rp = realpath(dir.path, nil) {
            dir = URL(fileURLWithPath: String(cString: rp), isDirectory: true)
            free(rp)
        }
        for i in 0 ..< files {
            try "document \(name) \(i) about search indexes and folders"
                .write(to: dir.appendingPathComponent("\(name)\(i).txt"), atomically: true, encoding: .utf8)
        }
        return dir
    }

    private func runPass(_ indexer: Indexer, roots: [URL]) {
        let done = expectation(description: "pass \(roots.map(\.lastPathComponent))")
        indexer.index(roots: roots, settings: IndexSettings()) { p in if p.done { done.fulfill() } }
        wait(for: [done], timeout: 60)
    }

    /// The add-folder flow: index A, then run a catch-up pass over ONLY the new root B.
    /// A's files must survive, B's must be added.
    func testCatchUpPassOverNewRootKeepsOtherRoots() throws {
        let a = try makeRoot("a", files: 5)
        let b = try makeRoot("b", files: 3)
        defer { try? FileManager.default.removeItem(at: a); try? FileManager.default.removeItem(at: b) }
        let dbURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-reconcile-db-\(UUID().uuidString)", isDirectory: true)
            .appendingPathComponent("index.sqlite")
        let store = try VectorStore(dbURL: dbURL)
        let indexer = Indexer(store: store, embedder: UnitTextEmbedder())

        runPass(indexer, roots: [a])
        XCTAssertEqual(store.fileCount(underFolder: a.path), 5,
                       "a=\(a.path) stored=\(store.indexedFiles().keys.sorted().prefix(2))")

        // Simulates AppModel.catchUpPendingRoots after "add folder B" in the sidebar.
        runPass(indexer, roots: [b])
        XCTAssertEqual(store.fileCount(underFolder: b.path), 3, "new root indexed")
        XCTAssertEqual(store.fileCount(underFolder: a.path), 5,
                       "adding root B must not reconcile-delete root A's index")
    }

    final class CountingEmbedder: Embedder, @unchecked Sendable {
        let dim = 8
        private let lock = NSLock()
        private var n = 0, m = 0
        var texts: Int { lock.withLock { n } }
        var images: Int { lock.withLock { m } }
        private func unit() -> [Float] { var v = [Float](repeating: 0, count: 8); v[0] = 1; return v }
        func embedText(_ text: String, as type: OmniInputType) -> [Float] { lock.withLock { n += 1 }; return unit() }
        func embedTextBatch(_ t: [String], as type: OmniInputType) -> [[Float]] { lock.withLock { n += t.count }; return t.map { _ in unit() } }
        func embedImage(_ image: CGImage) -> [Float]? { nil }
        func embedImages(_ raws: [OmniVisionPreprocess.RawPatches]) -> [[Float]]? {
            lock.withLock { m += raws.count }; return raws.map { _ in unit() }
        }
        func embedVideoFrames(_ frames: [CGImage]) -> [Float]? { nil }
        func embedAudio(_ url: URL) -> [Float]? { nil }
        func embedAudioMel(_ mel: [Float], frames: Int) -> [Float]? { nil }
        func embedAudioMelBatch(_ mels: [[Float]], frames: [Int]) -> [[Float]]? { nil }
    }

    private func writePNG(_ url: URL, seed: Int) throws {
        let ctx = CGContext(data: nil, width: 64, height: 64, bitsPerComponent: 8, bytesPerRow: 0,
                            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.setFillColor(CGColor(red: CGFloat(seed % 7) / 7, green: CGFloat(seed % 5) / 5, blue: 0.5, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: 64, height: 32))
        let dest = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil)!
        CGImageDestinationAddImage(dest, ctx.makeImage()!, nil)
        guard CGImageDestinationFinalize(dest) else { throw CocoaError(.fileWriteUnknown) }
    }

    /// A copied-in tree arrives as one event per folder and file. Each file is embedded once.
    func testNestedEventsEmbedEachFileOnce() throws {
        let root = try makeRoot("nested", files: 0)
        let deep = root.appendingPathComponent("a/b", isDirectory: true)
        try FileManager.default.createDirectory(at: deep, withIntermediateDirectories: true)
        for i in 0 ..< 4 { try writePNG(deep.appendingPathComponent("d\(i).png"), seed: i) }
        defer { try? FileManager.default.removeItem(at: root) }
        let dbURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-reconcile-db-\(UUID().uuidString)", isDirectory: true)
            .appendingPathComponent("index.sqlite")
        let store = try VectorStore(dbURL: dbURL)
        let embedder = CountingEmbedder()
        let indexer = Indexer(store: store, embedder: embedder)
        let events = [root.path, root.appendingPathComponent("a").path, deep.path]
            + (0 ..< 4).map { deep.appendingPathComponent("d\($0).png").path }
        indexer.update(paths: events, settings: IndexSettings(), roots: [root.path])
        XCTAssertEqual(store.fileCount(underFolder: root.path), 4)
        XCTAssertEqual(embedder.images, 4, "each image embedded once, not once per covering event")
    }

    /// A watcher event naming a root's PARENT as gone (the parent was renamed, or the volume
    /// unmounted) must not delete the root's rows; a vanished subfolder inside the root still must.
    func testVanishedAncestorOfRootKeepsRows() throws {
        let parent = try makeRoot("parent", files: 0)
        let root = parent.appendingPathComponent("docs", isDirectory: true)
        try FileManager.default.createDirectory(at: root.appendingPathComponent("sub"), withIntermediateDirectories: true)
        for i in 0 ..< 3 {
            try "notes \(i) on design reviews".write(to: root.appendingPathComponent("n\(i).txt"), atomically: true, encoding: .utf8)
        }
        try "a file about budgets".write(to: root.appendingPathComponent("sub/s.txt"), atomically: true, encoding: .utf8)
        let moved = parent.deletingLastPathComponent().appendingPathComponent(parent.lastPathComponent + "-moved")
        defer { try? FileManager.default.removeItem(at: parent); try? FileManager.default.removeItem(at: moved) }
        let dbURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-reconcile-db-\(UUID().uuidString)", isDirectory: true)
            .appendingPathComponent("index.sqlite")
        let store = try VectorStore(dbURL: dbURL)
        let indexer = Indexer(store: store, embedder: UnitTextEmbedder())
        runPass(indexer, roots: [root])
        XCTAssertEqual(store.fileCount(underFolder: root.path), 4)

        try FileManager.default.moveItem(at: parent, to: moved)
        indexer.update(paths: [parent.path], settings: IndexSettings(), roots: [root.path])
        XCTAssertEqual(store.fileCount(underFolder: root.path), 4, "a moved parent must not wipe the root")
        indexer.update(paths: [root.path], settings: IndexSettings(), roots: [root.path])
        XCTAssertEqual(store.fileCount(underFolder: root.path), 4, "a missing root must not be wiped")

        try FileManager.default.moveItem(at: moved, to: parent)
        try FileManager.default.removeItem(at: root.appendingPathComponent("sub"))
        indexer.update(paths: [root.appendingPathComponent("sub").path], settings: IndexSettings(), roots: [root.path])
        XCTAssertEqual(store.fileCount(underFolder: root.path), 3, "a deleted subfolder is still pruned")
    }

    /// A watcher event admits exactly what the crawl admits: nothing under a hidden folder or
    /// inside a package, so a save there is not embedded only for the next full pass to delete it.
    func testEventPathsFollowCrawlAdmission() throws {
        let root = try makeRoot("admit", files: 1)   // admit0.txt
        let fm = FileManager.default
        try fm.createDirectory(at: root.appendingPathComponent(".vscode"), withIntermediateDirectories: true)
        try fm.createDirectory(at: root.appendingPathComponent("Tool.app/Contents"), withIntermediateDirectories: true)
        try "settings for the editor".write(to: root.appendingPathComponent(".vscode/notes.txt"), atomically: true, encoding: .utf8)
        try "bundle resource text".write(to: root.appendingPathComponent("Tool.app/Contents/readme.txt"), atomically: true, encoding: .utf8)
        defer { try? fm.removeItem(at: root) }
        func freshStore() throws -> VectorStore {
            try VectorStore(dbURL: FileManager.default.temporaryDirectory
                .appendingPathComponent("omni-reconcile-db-\(UUID().uuidString)", isDirectory: true)
                .appendingPathComponent("index.sqlite"))
        }
        let pass = try freshStore()
        runPass(Indexer(store: pass, embedder: UnitTextEmbedder()), roots: [root])
        let events = [root.appendingPathComponent("admit0.txt").path,
                      root.appendingPathComponent(".vscode/notes.txt").path,
                      root.appendingPathComponent("Tool.app").path,
                      root.appendingPathComponent("Tool.app/Contents/readme.txt").path]
        let live = try freshStore()
        Indexer(store: live, embedder: UnitTextEmbedder())
            .update(paths: events, settings: IndexSettings(), roots: [root.path])
        XCTAssertEqual(Set(live.indexedFiles().keys), Set(pass.indexedFiles().keys))
        XCTAssertEqual(live.fileCount(underFolder: root.path), 1)
    }

    /// The paused-root flow: a full pass excludes paused roots; their files must survive.
    func testPassExcludingPausedRootKeepsItsFiles() throws {
        let a = try makeRoot("a", files: 4)
        let b = try makeRoot("b", files: 2)
        defer { try? FileManager.default.removeItem(at: a); try? FileManager.default.removeItem(at: b) }
        let dbURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-reconcile-db-\(UUID().uuidString)", isDirectory: true)
            .appendingPathComponent("index.sqlite")
        let store = try VectorStore(dbURL: dbURL)
        let indexer = Indexer(store: store, embedder: UnitTextEmbedder())

        runPass(indexer, roots: [a, b])
        XCTAssertEqual(store.fileCount, 6)

        // User pauses B, then a full pass runs over the remaining roots.
        runPass(indexer, roots: [a])
        XCTAssertEqual(store.fileCount(underFolder: b.path), 2,
                       "a pass that wasn't asked to crawl B must not delete B's index")
    }

    /// Deletion still works where it should: a file deleted from disk inside a crawled root is
    /// reconciled away by the next pass over that root.
    func testReconcileStillRemovesDeletedFilesInCrawledRoot() throws {
        let a = try makeRoot("a", files: 3)
        defer { try? FileManager.default.removeItem(at: a) }
        let dbURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-reconcile-db-\(UUID().uuidString)", isDirectory: true)
            .appendingPathComponent("index.sqlite")
        let store = try VectorStore(dbURL: dbURL)
        let indexer = Indexer(store: store, embedder: UnitTextEmbedder())

        runPass(indexer, roots: [a])
        XCTAssertEqual(store.fileCount, 3)

        try FileManager.default.removeItem(at: a.appendingPathComponent("a0.txt"))
        runPass(indexer, roots: [a])
        XCTAssertEqual(store.fileCount, 2, "deleted file reconciled away")
        XCTAssertNil(store.fileVector(a.appendingPathComponent("a0.txt").path))
    }
}
