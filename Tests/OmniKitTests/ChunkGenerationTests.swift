import XCTest
import CoreGraphics
@testable import OmniKit

/// ONE INDEX, TWO CUTTERS, which is the state every existing index enters the moment the content
/// cutter is turned on.
///
/// Turning it on re-indexes nothing - "unchanged" is mtime and size - so a real index holds
/// generation-1 chunks for every file nobody has touched and generation-2 chunks for the ones that
/// have changed since. The properties that have to survive that are the ones here: the untouched
/// files are not re-embedded, the edited file is re-cut rather than partially reused, and a chunk
/// of one generation is never served for a chunk of the other.
final class ChunkGenerationTests: XCTestCase {

    final class CountingEmbedder: Embedder, @unchecked Sendable {
        let dim = 64
        private let lock = NSLock()
        private var _count = 0
        var embedded: Int { lock.lock(); defer { lock.unlock() }; return _count }
        func reset() { lock.lock(); _count = 0; lock.unlock() }
        private func vec(_ text: String) -> [Float] {
            var h: UInt64 = 1_469_598_103_934_665_603
            for b in text.utf8 { h = (h ^ UInt64(b)) &* 1_099_511_628_211 }
            var v = [Float](repeating: 0, count: 64)
            var n: Float = 0
            for i in 0 ..< 64 {
                h ^= h << 13; h ^= h >> 7; h ^= h << 17
                let x = Float(Int(h % 512) - 256) / 256.0
                v[i] = x; n += x * x
            }
            let inv = n > 0 ? 1 / n.squareRoot() : 0
            for i in 0 ..< 64 { v[i] *= inv }
            return v
        }
        func embedText(_ text: String, as type: OmniInputType) -> [Float] {
            lock.lock(); _count += 1; lock.unlock(); return vec(text)
        }
        func embedTextBatch(_ texts: [String], as type: OmniInputType) -> [[Float]] {
            lock.lock(); _count += texts.count; lock.unlock(); return texts.map(vec)
        }
        func embedImage(_ image: CGImage) -> [Float]? { nil }
        func embedImages(_ raws: [OmniVisionPreprocess.RawPatches]) -> [[Float]]? { nil }
        func embedVideoFrames(_ frames: [CGImage]) -> [Float]? { nil }
        func embedAudio(_ url: URL) -> [Float]? { nil }
        func embedAudioMel(_ mel: [Float], frames: Int) -> [Float]? { nil }
        func embedAudioMelBatch(_ mels: [[Float]], frames: [Int]) -> [[Float]]? { nil }
    }

    private var savedCDC = false
    override func setUp() { super.setUp(); savedCDC = Indexer.contentDefinedChunking }
    override func tearDown() { Indexer.contentDefinedChunking = savedCDC; super.tearDown() }

    /// Distinct, multi-chunk files. Numbered so no two chunks are the same content, which would
    /// let dedup answer a question this test is asking of the cutter.
    private func corpus(_ n: Int) throws -> URL {
        var dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-gen-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        if let rp = realpath(dir.path, nil) {
            dir = URL(fileURLWithPath: String(cString: rp), isDirectory: true); free(rp)
        }
        for i in 0 ..< n {
            var text = ""
            for p in 0 ..< 8 {
                text += "file \(i) section \(p): "
                for w in 0 ..< 40 {
                    text += "word\(i)_\(p)_\(w) about indexes chunks vectors and retrieval. "
                }
                text += "\n\n"
            }
            try text.write(to: dir.appendingPathComponent("f\(i).txt"), atomically: true, encoding: .utf8)
        }
        return dir
    }

    private func pass(_ store: VectorStore, _ embedder: CountingEmbedder, _ root: URL) {
        let idx = Indexer(store: store, embedder: embedder)
        let done = expectation(description: "pass")
        idx.index(roots: [root], settings: IndexSettings()) { p in if p.done { done.fulfill() } }
        wait(for: [done], timeout: 300)
    }

    func testTurningTheCutterOnReIndexesNothingUntilAFileChanges() throws {
        let n = 6
        let root = try corpus(n); defer { try? FileManager.default.removeItem(at: root) }
        let dbDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-gen-db-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dbDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dbDir) }
        let store = try VectorStore(dbURL: dbDir.appendingPathComponent("index.sqlite"))
        defer { store.close() }
        let embedder = CountingEmbedder()

        // An index built the old way.
        Indexer.contentDefinedChunking = false
        pass(store, embedder, root)
        let gridChunks = store.count
        XCTAssertGreaterThan(gridChunks, n * 3, "the fixture files are not multi-chunk")

        // THE UPGRADE. Nothing has changed on disk, so nothing may reach the encoder - the cutter
        // is not part of the unchanged test, and making it one would re-embed every corpus on
        // every machine the day this shipped.
        Indexer.contentDefinedChunking = true
        embedder.reset()
        pass(store, embedder, root)
        XCTAssertEqual(embedder.embedded, 0, "turning the cutter on re-embedded an untouched corpus")
        XCTAssertEqual(store.count, gridChunks, "the chunk table changed without a file changing")

        // Edit ONE file. It is re-cut under generation 2, and because the two key spaces are
        // disjoint none of its old chunks can be reused - which is correct, they are different
        // chunks. The other five must not move.
        let edited = root.appendingPathComponent("f0.txt")
        let old = try String(contentsOf: edited, encoding: .utf8)
        try ("an inserted first line that shifts every grid boundary below it.\n" + old)
            .write(to: edited, atomically: true, encoding: .utf8)
        embedder.reset()
        pass(store, embedder, root)
        XCTAssertGreaterThan(embedder.embedded, 0, "the edit never reached the indexer")
        // Only one file's worth: the other five were not touched.
        XCTAssertLessThan(embedder.embedded, gridChunks / n * 2,
                          "\(embedder.embedded) chunks embedded for a one-file edit")

        // Everything is still findable, under both generations.
        for i in 0 ..< n {
            let hits = store.search(embedder.embedText("file \(i) section 3", as: .query), topK: 6)
            XCTAssertFalse(hits.isEmpty, "file \(i) returned nothing after the mixed-generation pass")
        }
        XCTAssertEqual(store.allIndexedPaths().count, n, "files lost across the generation change")

        // AND NOW THE POINT OF THE WHOLE CHANGE. f0 is generation 2 already, so a SECOND insertion
        // at the top is the case the cutter exists for: under the grid every boundary below the
        // edit moves and the file re-embeds whole, and under this cutter only the chunks around
        // the edit do. Measured here through the real indexer rather than the cutter's own unit
        // test, because what matters is how many forward passes a user's edit actually costs.
        // With the flag flipped back to the grid for this phase it re-embeds 11 of 11; with the
        // content cutter, 3.
        let perFile = gridChunks / n
        let again = try String(contentsOf: edited, encoding: .utf8)
        try ("a second inserted line, same shape as the first.\n" + again)
            .write(to: edited, atomically: true, encoding: .utf8)
        embedder.reset()
        pass(store, embedder, root)
        XCTAssertGreaterThan(embedder.embedded, 0, "the second edit never reached the indexer")
        XCTAssertLessThanOrEqual(embedder.embedded, 3,
            "\(embedder.embedded) chunks re-embedded for a one-line insertion into a \(perFile)-chunk file")
    }

    /// THE KEY SPACES ARE DISJOINT, stated directly rather than inferred from the behaviour above.
    /// A generation-1 chunk served for a generation-2 chunk of the same bytes would be a stale
    /// vector for text that was cut differently, and nothing downstream could notice.
    func testTheSameBytesUnderTwoCuttersAreDifferentChunks() {
        let text = "a passage that exists in an index under both cutters, long enough to be real."
        let g = ChunkKey.grid(text, maxChars: 1800, overlap: 200, dim: 768)
        let c = ChunkKey.text(text, cutter: ContentChunker.Params.forMaxChars(1800).fingerprint, dim: 768)
        XCTAssertNotEqual(g, c)
        XCTAssertEqual(g.count, c.count)
    }
}
