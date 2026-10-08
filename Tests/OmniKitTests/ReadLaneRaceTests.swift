import XCTest
@testable import OmniKit

/// The first browses on a fresh store arrive together (sidebar counts, the folder listing). The
/// read lanes were `lazy var`s, which are not thread-safe: two threads each built a lane and one
/// went on with a freed queue. Crashed every run of this test (SIGABRT/SIGSEGV) before the lanes
/// were built in init.
final class ReadLaneRaceTests: XCTestCase {
    func testTheFirstBrowsesOnAFreshStoreMayRace() throws {
        let base = FileManager.default.temporaryDirectory.appendingPathComponent("lanerace-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: base) }
        for i in 0 ..< 300 {
            let dir = base.appendingPathComponent("\(i)")
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            let store = try VectorStore(dbURL: dir.appendingPathComponent("index.sqlite"))
            DispatchQueue.concurrentPerform(iterations: 8) { _ in _ = store.indexedChildren(ofFolder: "/") }
            store.close()
        }
    }
}
