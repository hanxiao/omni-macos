import XCTest
import CoreGraphics
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
