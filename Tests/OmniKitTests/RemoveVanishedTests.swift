import XCTest
@testable import OmniKit

/// The watcher's delete lane: rows of what is gone leave at once, without the pipeline, and what
/// update() protects (a root, a path that came back) stays.
final class RemoveVanishedTests: XCTestCase {
    func testGoneFilesAndFoldersLeaveRootsAndPresentFilesStay() throws {
        var base = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-vanish-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        if let rp = realpath(base.path, nil) { base = URL(fileURLWithPath: String(cString: rp), isDirectory: true); free(rp) }
        defer { try? FileManager.default.removeItem(at: base) }
        let root = base.appendingPathComponent("root", isDirectory: true)
        let other = base.appendingPathComponent("other", isDirectory: true)
        for d in ["root/big", "root/keep", "other"] {
            try FileManager.default.createDirectory(at: base.appendingPathComponent(d), withIntermediateDirectories: true)
        }
        for i in 0 ..< 5 {
            try "big note \(i) about search".write(to: root.appendingPathComponent("big/b\(i).txt"), atomically: true, encoding: .utf8)
            try "kept note \(i) about search".write(to: root.appendingPathComponent("keep/k\(i).txt"), atomically: true, encoding: .utf8)
            try "other note \(i) about search".write(to: other.appendingPathComponent("o\(i).txt"), atomically: true, encoding: .utf8)
        }
        let dbURL = base.appendingPathComponent("db/index.sqlite")
        try FileManager.default.createDirectory(at: dbURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        let store = try VectorStore(dbURL: dbURL)
        let indexer = Indexer(store: store, embedder: IndexerReconcileTests.UnitTextEmbedder())
        let done = expectation(description: "pass")
        indexer.index(roots: [root, other], settings: IndexSettings(enabledKinds: [.text])) { p in if p.done { done.fulfill() } }
        wait(for: [done], timeout: 60)
        func names() -> Set<String> { Set(store.knownFiles().compactMap { p, _ in (p as NSString).lastPathComponent }) }
        XCTAssertEqual(names().count, 15)

        // A folder and a single file gone, a file still present, and a whole root gone.
        try FileManager.default.removeItem(at: root.appendingPathComponent("big"))
        try FileManager.default.removeItem(at: root.appendingPathComponent("keep/k0.txt"))
        try FileManager.default.removeItem(at: other)
        let removed = indexer.removeVanished([root.appendingPathComponent("big").path,
                                              root.appendingPathComponent("keep/k0.txt").path,
                                              root.appendingPathComponent("keep/k1.txt").path,
                                              other.path],
                                             roots: [root.path, other.path])
        XCTAssertEqual(removed, 2, "the folder and the file; the present file and the root are not removed")
        XCTAssertEqual(names(), ["k1.txt", "k2.txt", "k3.txt", "k4.txt", "o0.txt", "o1.txt", "o2.txt", "o3.txt", "o4.txt"],
                       "a missing root keeps its rows, as update() and the full pass do")

        // Idempotent with the reconcile that drains the same paths afterwards.
        XCTAssertEqual(indexer.removeVanished([root.appendingPathComponent("big").path], roots: [root.path, other.path]), 0)
        indexer.update(paths: [root.appendingPathComponent("big").path, root.appendingPathComponent("keep/k0.txt").path],
                       settings: IndexSettings(enabledKinds: [.text]), roots: [root.path, other.path])
        XCTAssertEqual(names().count, 9)
    }
}
