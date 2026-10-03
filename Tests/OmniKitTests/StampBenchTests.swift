import XCTest
@testable import OmniKit

/// What one coverage stamp costs on a real index once coverage has caught up - the stamp that runs
/// two seconds after every write. OMNI_STAMP_BENCH=<index.sqlite> (use a clone).
final class StampBenchTests: XCTestCase {
    func testCaughtUpStampCost() throws {
        guard let path = ProcessInfo.processInfo.environment["OMNI_STAMP_BENCH"] else { throw XCTSkip("set OMNI_STAMP_BENCH") }
        let store = try VectorStore(dbURL: URL(fileURLWithPath: path))
        defer { store.close() }
        if ProcessInfo.processInfo.environment["OMNI_STAMP_MIGRATE"] == "1" {
            let t = Date()
            let n = store.runMigrationStampsForTest()
            print(String(format: "STAMP migrated in %d stamps, %.0f s", n, -t.timeIntervalSinceNow))
        }
        store.advanceCoverageForTest()
        print("STAMP holes=\(store.holesForTest().count) covered=\(store.coveredRowsForTest)")
        var times: [Double] = []
        for _ in 0 ..< (Int(ProcessInfo.processInfo.environment["OMNI_STAMP_REPS"] ?? "") ?? 5) {
            let t = Date()
            store.stampCoverageForTest()
            times.append(-t.timeIntervalSinceNow * 1000)
        }
        print("STAMP ms per caught-up stamp: " + times.map { String(format: "%.1f", $0) }.joined(separator: " "))
    }
}
