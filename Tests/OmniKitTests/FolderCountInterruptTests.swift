import XCTest
@testable import OmniKit

/// An older request's interrupt landing on the walk that is current: the counts it gives back are the
/// whole answer, not the part read before the interrupt.
final class FolderCountInterruptTests: XCTestCase {
    func testAnInterruptMeantForAnOlderWalkDoesNotCutTheCurrentOneShort() throws {
        let d = FileManager.default.temporaryDirectory.appendingPathComponent("aggint-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: d) }
        let s = try VectorStore(dbURL: d.appendingPathComponent("index.sqlite"))
        defer { s.close() }
        var batch: [(path: String, chunks: [IndexedChunk])] = []
        for folder in 0 ..< 40 {
            for f in 0 ..< 5 {
                let p = "/r/f\(folder)/n\(f).txt"
                var v = [Float](repeating: 0, count: 16); v[(folder + f) % 16] = 1
                batch.append((p, [IndexedChunk(path: p, modified: 1, size: 1, kind: "text", chunkIndex: 0, snippet: "s", embedding: v)]))
            }
        }
        try s.replaceMany(batch)
        let whole = s.folderCounts(under: "/r")
        XCTAssertEqual(whole.count, 40, "the probe")
        VectorStore.aggregateInterruptForTest = { s.interruptAggregatesForTest() }
        defer { VectorStore.aggregateInterruptForTest = nil }
        let cut = s.folderCounts(under: "/r")
        XCTAssertNil(VectorStore.aggregateInterruptForTest, "the probe: the interrupt never landed")
        XCTAssertEqual(cut.count, whole.count, "partial counts were handed back as the answer")
        XCTAssertEqual(cut.values.map(\.count).reduce(0, +), 200)
    }
}
