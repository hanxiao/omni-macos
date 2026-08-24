import XCTest
import CoreGraphics
@testable import OmniKit

/// How a Photos source behaves as a ROOT. These run in a process with no Photos authorization,
/// which is exactly the case that must not destroy anything: the library reads as empty, and an
/// empty root is indistinguishable from a revoked one, so the pass has to keep its rows.
final class PhotoSourceIndexTests: XCTestCase {
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

    private func makeRoot(files: Int) throws -> URL {
        var dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-photos-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        if let rp = realpath(dir.path, nil) {
            dir = URL(fileURLWithPath: String(cString: rp), isDirectory: true)
            free(rp)
        }
        for i in 0 ..< files {
            try "document \(i) about search indexes"
                .write(to: dir.appendingPathComponent("f\(i).txt"), atomically: true, encoding: .utf8)
        }
        return dir
    }

    private func makeStore() throws -> VectorStore {
        try VectorStore(dbURL: FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-photos-db-\(UUID().uuidString)", isDirectory: true)
            .appendingPathComponent("index.sqlite"))
    }

    private func seedPhotoRow(_ store: VectorStore, path: String) throws {
        var v = [Float](repeating: 0, count: 8); v[0] = 1
        try store.replace(path: path, chunks: [
            IndexedChunk(path: path, modified: 1_700_000_000, size: 12_000_000, kind: FileKind.image.rawValue,
                         chunkIndex: 0, snippet: "kitten, sofa", embedding: v, width: 4000, height: 3000)
        ])
    }

    private func runPass(_ indexer: Indexer, roots: [URL], photos: [PhotoLibrary.Source]) {
        let done = expectation(description: "pass")
        indexer.index(roots: roots, photos: photos, settings: IndexSettings()) { p in if p.done { done.fulfill() } }
        wait(for: [done], timeout: 60)
    }

    /// The regression that would have hurt most: a pass that includes an unreadable Photos source
    /// (access revoked, or - as here - never granted) must leave its rows alone. It reads as a
    /// blind root, exactly like a folder whose permission was withdrawn.
    func testUnreadablePhotoSourceDoesNotSweepItsRows() throws {
        let store = try makeStore()
        let indexer = Indexer(store: store, embedder: UnitTextEmbedder())
        let root = try makeRoot(files: 3)
        defer { try? FileManager.default.removeItem(at: root) }

        let source = PhotoLibrary.Source.all
        let photoPath = source.key + "/" + PhotoLibrary.esc("AAAA-BBBB/L0/001") + "/IMG_1.HEIC"
        try seedPhotoRow(store, path: photoPath)
        XCTAssertEqual(store.fileCount(underFolder: source.key), 1)

        runPass(indexer, roots: [root], photos: [source])

        XCTAssertEqual(store.fileCount(underFolder: root.path), 3, "the folder root still indexes normally")
        XCTAssertEqual(store.fileCount(underFolder: source.key), 1,
                       "an unreadable Photos source must not have its rows swept")
    }

    /// Removing a source drops exactly its rows - the prefix delete the sidebar's Remove relies on.
    func testDeletingASourceDropsOnlyItsRows() throws {
        let store = try makeStore()
        let all = PhotoLibrary.Source.all
        let album = PhotoLibrary.Source(id: "5E2F5C3A-0000-4000-8000-000000000001/L0/040", title: "Iceland")

        try seedPhotoRow(store, path: all.key + "/" + PhotoLibrary.esc("A/L0/001") + "/IMG_1.HEIC")
        try seedPhotoRow(store, path: album.key + "/" + PhotoLibrary.esc("B/L0/001") + "/IMG_2.HEIC")
        try seedPhotoRow(store, path: album.key + "/" + PhotoLibrary.esc("C/L0/001") + "/IMG_3.HEIC")

        store.deleteUnderFolder(album.key)

        XCTAssertEqual(store.fileCount(underFolder: album.key), 0)
        XCTAssertEqual(store.fileCount(underFolder: all.key), 1, "the other source is untouched")
    }

    /// A pass with no folder roots at all - the user who indexes only their Photos library.
    func testPhotosOnlyPassCompletes() throws {
        let store = try makeStore()
        let indexer = Indexer(store: store, embedder: UnitTextEmbedder())
        let done = expectation(description: "photos-only pass")
        indexer.index(roots: [], photos: [.all], settings: IndexSettings()) { p in
            if p.done { XCTAssertFalse(p.cancelled); done.fulfill() }
        }
        wait(for: [done], timeout: 60)
    }
}
