import XCTest
@testable import OmniKit

/// Folder-switch cost against a REAL index, with the store's serial queue contended the way the
/// indexer contends it. Skipped unless OMNI_REAL_INDEX names a database, because there is no
/// synthetic index whose numbers mean anything here - the whole question is what a 2.6M-file
/// `dirs` table costs.
///
///   OMNI_REAL_INDEX=~/Library/.../index.sqlite \
///   OMNI_BROWSE_FOLDERS=~/Desktop:~/Documents:~/Downloads \
///   swift test --filter BrowseReaderBench
///
/// Run it twice, once with OMNI_BROWSE_READER=0, for the A/B that says what the second connection
/// actually buys.
final class BrowseReaderBenchTests: XCTestCase {

    func testFolderSwitchUnderContention() throws {
        guard let raw = ProcessInfo.processInfo.environment["OMNI_REAL_INDEX"] else {
            throw XCTSkip("set OMNI_REAL_INDEX to a real index to run this")
        }
        let db = URL(fileURLWithPath: (raw as NSString).expandingTildeInPath)
        let folders = (ProcessInfo.processInfo.environment["OMNI_BROWSE_FOLDERS"] ?? NSHomeDirectory())
            .split(separator: ":").map { ($0 as NSString).expandingTildeInPath }
        let readerOn = ProcessInfo.processInfo.environment["OMNI_BROWSE_READER"] != "0"

        let store = try VectorStore(dbURL: db)
        defer { store.close() }
        print("reader=\(readerOn ? "ON" : "OFF")  files=\(store.fileCount)  db=\(db.path)")

        for f in folders { _ = store.indexedChildrenDetailed(ofFolder: f, aggregates: false) }

        // A contending holder. Not a real write - that would change the user's index - but the same
        // effect on a browse: it occupies the serial queue in slices the size of a write batch.
        let stop = Flag()
        let worker = Thread {
            while !stop.isSet {
                store.holdSerialQueueForTesting(milliseconds: 120)
                usleep(30_000)
            }
        }
        worker.start()
        defer { stop.set(); usleep(300_000) }
        usleep(200_000)

        var list: [Double] = [], counts: [Double] = []
        for _ in 0 ..< 8 {
            for f in folders {
                var t = Date()
                let rows = store.indexedChildrenDetailed(ofFolder: f, aggregates: false)
                list.append(Date().timeIntervalSince(t) * 1000)
                t = Date()
                _ = store.folderCounts(under: f)
                counts.append(Date().timeIntervalSince(t) * 1000)
                XCTAssertFalse(rows.isEmpty, "\(f) listed nothing - wrong folder for this index?")
            }
        }
        report("list", list)
        report("counts", counts)
    }

    private func report(_ name: String, _ xs: [Double]) {
        let s = xs.sorted()
        print(String(format: "%-7@ n=%d  p50 %.1fms  p90 %.1fms  max %.1fms",
                     name, s.count, s[s.count / 2], s[Int(Double(s.count) * 0.9)], s.last ?? 0))
    }

    private final class Flag: @unchecked Sendable {
        private var v = false
        private let l = NSLock()
        var isSet: Bool { l.lock(); defer { l.unlock() }; return v }
        func set() { l.lock(); v = true; l.unlock() }
    }
}
