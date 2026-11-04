import XCTest
@testable import OmniKit

/// A folder's removal is noted in the index until its rows are gone, so a quit part-way does not
/// leave the rest searchable: the note survives a reopen, settles per key, and a removal noted
/// while another settles is kept.
final class OwedRemovalsTests: XCTestCase {
    func testOwedRemovalsSurviveReopenAndSettlePerKey() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("owed-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("t.sqlite")

        var store: VectorStore? = try VectorStore(dbURL: url)
        XCTAssertEqual(store!.owedRemovals, [])
        store!.oweRemoval(["/r/Big", "photos:album/x"])
        store!.oweRemoval(["/r/Big"])                       // noted twice, owed once
        XCTAssertEqual(store!.owedRemovals, ["/r/Big", "photos:album/x"])
        store!.close(); store = nil

        let reopened = try VectorStore(dbURL: url)
        defer { reopened.close() }
        XCTAssertEqual(reopened.owedRemovals, ["/r/Big", "photos:album/x"], "a quit before the delete keeps it owed")
        reopened.settleRemoval(["/r/Big"])
        XCTAssertEqual(reopened.owedRemovals, ["photos:album/x"])
        reopened.settleRemoval(["/never/owed"])
        XCTAssertEqual(reopened.owedRemovals, ["photos:album/x"])

        // Noting and settling from many threads at once: every key noted and not settled stays.
        DispatchQueue.concurrentPerform(iterations: 64) { i in
            reopened.oweRemoval(["/k/\(i)"])
            if i % 2 == 0 { reopened.settleRemoval(["/k/\(i)"]) }
        }
        let odd = (0 ..< 64).filter { $0 % 2 == 1 }.map { "/k/\($0)" }
        XCTAssertEqual(Set(reopened.owedRemovals), Set(odd + ["photos:album/x"]))
        reopened.settleRemoval(odd + ["photos:album/x"])
        XCTAssertEqual(reopened.owedRemovals, [])
        XCTAssertNil(reopened.metaGet(VectorStore.owedRemovalsKey), "nothing owed leaves no key")
    }
}
