import XCTest
import SQLite3
@testable import OmniKit

/// The 0.4.x -> 0.5.0 upgrade showed "Upgrading your index" behind a bar pinned at exactly 100% for
/// the whole rewrite, which reads as a hang at the worst possible moment.
///
/// The cause was not the migration. The upgrade runs AFTER loadIntoMemory - loading first is what
/// lets the rewrite drop the duplicate vector blobs as it goes - and the load ended by reporting 1
/// on the same channel. The app's consumer is monotonic so the bar cannot jump backwards
/// mid-launch:
///
///     storeLoadFrac   = max(storeLoadFrac, min(1, f))
///     loadingProgress = max(loadingProgress ?? 0, 0.5 * storeLoadFrac + 0.5 * engineLoadFrac)
///
/// so every value the upgrade reported was already <= the value held.
///
/// The fix keeps ONE bar and makes the store scale both phases into it, which is what these tests
/// hold it to: the displayed value must still be rising after the upgrade begins.
final class UpgradeProgressTests: XCTestCase {
    private func open(_ url: URL) -> OpaquePointer? {
        var db: OpaquePointer?
        _ = sqlite3_open(url.path, &db)
        return db
    }

    private func exec(_ db: OpaquePointer?, _ sql: String) { sqlite3_exec(db, sql, nil, nil, nil) }

    /// A 0.4.x index: path-per-row `chunks`, user_version 2.
    private func makeLegacyIndex(_ dbURL: URL, files: Int) {
        let db = open(dbURL)
        defer { sqlite3_close(db) }
        exec(db, """
            PRAGMA journal_mode=WAL;
            CREATE TABLE chunks(path TEXT NOT NULL, modified REAL NOT NULL, size INTEGER NOT NULL DEFAULT 0,
                kind TEXT NOT NULL, chunk_index INTEGER NOT NULL, snippet TEXT NOT NULL,
                dim INTEGER NOT NULL, vec BLOB NOT NULL, width INTEGER NOT NULL DEFAULT 0,
                height INTEGER NOT NULL DEFAULT 0, duration REAL NOT NULL DEFAULT 0,
                locator TEXT NOT NULL DEFAULT '', indexed_at REAL NOT NULL DEFAULT 0,
                chunk_key TEXT NOT NULL DEFAULT '',
                PRIMARY KEY(path, chunk_index));
            CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            PRAGMA user_version = 2;
            """)
        exec(db, "BEGIN;")
        for f in 0 ..< files {
            for i in 0 ..< 3 {
                exec(db, "INSERT INTO chunks(path,modified,size,kind,chunk_index,snippet,dim,vec) VALUES('/i/f\(f).txt',1,10,'text',\(i),'s\(f)-\(i)',64,x'00');")
            }
        }
        exec(db, "COMMIT;")
    }

    /// AppModel's launch bar, reproduced exactly - monotonic, 50/50 with the engine, capped just
    /// short of the end, and with the engine already finished, which is the situation a slow
    /// upgrade is in. Mirrors AppModel.refreshLoadingProgress; if that changes, this must, or the
    /// test is measuring a bar the app does not have.
    private final class LaunchBar: @unchecked Sendable {
        static let ceiling = 0.99         // AppModel.launchBarCeiling
        private let lock = NSLock()
        private var storeFrac = 0.0
        private var engineFrac = 1.0
        private(set) var shown: [Double] = []
        private(set) var phases: [StoreOpenPhase] = []
        private(set) var storeReports: [Double] = []
        /// Bar value at the moment each phase was announced.
        private(set) var atPhase: [Double] = []
        var displayed: Double { lock.withLock { shown.last ?? 0 } }
        func note(_ f: Double) {
            lock.withLock {
                storeReports.append(f)
                storeFrac = max(storeFrac, min(1, f))
                shown.append(max(shown.last ?? 0, min(Self.ceiling, 0.5 * storeFrac + 0.5 * engineFrac)))
            }
        }
        func notePhase(_ p: StoreOpenPhase) {
            lock.withLock { phases.append(p); atPhase.append(shown.last ?? 0) }
        }
    }

    private func runUpgrade(files: Int, slice: Int) throws -> LaunchBar {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-upgrade-progress-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbURL = dir.appendingPathComponent("index.sqlite")
        makeLegacyIndex(dbURL, files: files)

        let saved = VectorStore.upgradeSliceRows
        VectorStore.upgradeSliceRows = slice        // the shipped 250k would convert this in one step
        defer { VectorStore.upgradeSliceRows = saved }

        let bar = LaunchBar()
        let store = try VectorStore(dbURL: dbURL,
                                    onLoadProgress: { bar.note($0) },
                                    onPhase: { bar.notePhase($0) })
        store.close()
        return bar
    }

    func testTheBarKeepsMovingThroughTheUpgrade() throws {
        let bar = try runUpgrade(files: 400, slice: 200)

        XCTAssertEqual(bar.phases, [.loadingIndex, .upgradingIndex],
                       "the launch screen was never told what the store was doing")

        // THE DEFECT: the bar used to be full before the upgrade even started, so nothing it
        // reported could move it. It must now have headroom left at that moment.
        guard let upgradeIdx = bar.phases.firstIndex(of: .upgradingIndex) else { return XCTFail("no upgrade") }
        let atUpgradeStart = bar.atPhase[upgradeIdx]
        XCTAssertLessThan(atUpgradeStart, 0.95,
                          "the bar was at \(atUpgradeStart) when the upgrade began - no room left to show it")

        // ... and it must actually rise afterwards, on the SAME bar.
        let after = bar.shown.filter { $0 > atUpgradeStart }
        XCTAssertGreaterThanOrEqual(after.count, 2,
                                    "the bar moved \(after.count) times during the upgrade")
        XCTAssertEqual(bar.shown, bar.shown.sorted(), "the launch bar went backwards")
        // The ceiling, not 1.0: the screen is torn down when the work finishes, so a bar that
        // reads 100% while still working would be the original defect all over again.
        XCTAssertEqual(bar.displayed, LaunchBar.ceiling, accuracy: 0.001, "the bar stalled short of the end")
        XCTAssertEqual(bar.storeReports.last ?? 0, 1.0, accuracy: 0.001,
                       "the store never reported its own completion")
    }

    /// A normal launch - nothing to upgrade - must be unchanged: the load still spans the whole
    /// store share, so the bar is not left short of the end by headroom reserved for an upgrade
    /// that never happens.
    func testNoUpgradeStillFillsTheBar() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-upgrade-none-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let dbURL = dir.appendingPathComponent("index.sqlite")

        // Build a current-schema store, close it, reopen it: the reopen has nothing to upgrade.
        let seed = try VectorStore(dbURL: dbURL)
        for f in 0 ..< 20 {
            let p = "/n/f\(f).txt"
            try seed.replace(path: p, chunks: [IndexedChunk(path: p, modified: 1, size: 1, kind: "text",
                                                            chunkIndex: 0, snippet: "s",
                                                            embedding: [Float](repeating: 0.1, count: 64))])
        }
        seed.close()

        let bar = LaunchBar()
        let store = try VectorStore(dbURL: dbURL, onLoadProgress: { bar.note($0) },
                                    onPhase: { bar.notePhase($0) })
        defer { store.close() }
        XCTAssertEqual(bar.displayed, LaunchBar.ceiling, accuracy: 0.001,
                       "a launch with no upgrade must still drive the bar to the end")
        XCTAssertEqual(bar.storeReports.last ?? 0, 1.0, accuracy: 0.001,
                       "the store must still report a full load when there is nothing to upgrade")
        XCTAssertFalse(bar.phases.contains(.upgradingIndex), "announced an upgrade that was not needed")
        XCTAssertFalse(bar.phases.contains(.loadingIndex),
                       "named a load too short to be worth naming - that is a label flash, not information")
    }
}
