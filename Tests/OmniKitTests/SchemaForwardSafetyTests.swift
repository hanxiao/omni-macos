import XCTest
import SQLite3
@testable import OmniKit

/// An index stamped with a format this build does not know is refused and left untouched.
final class SchemaForwardSafetyTests: XCTestCase {
    func testNewerFormatIsRefusedAndNotRestamped() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-fwd-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("index.sqlite")
        var db: OpaquePointer?
        XCTAssertEqual(sqlite3_open(url.path, &db), SQLITE_OK)
        sqlite3_exec(db, "CREATE TABLE future(x); PRAGMA user_version = 6;", nil, nil, nil)
        sqlite3_close(db)

        XCTAssertThrowsError(try VectorStore(dbURL: url))

        db = nil
        XCTAssertEqual(sqlite3_open(url.path, &db), SQLITE_OK)
        var st: OpaquePointer?
        sqlite3_prepare_v2(db, "PRAGMA user_version;", -1, &st, nil)
        XCTAssertEqual(sqlite3_step(st), SQLITE_ROW)
        XCTAssertEqual(sqlite3_column_int(st, 0), 6, "a refused index keeps its version")
        sqlite3_finalize(st)
        sqlite3_prepare_v2(db, "SELECT count(*) FROM sqlite_master WHERE name='future';", -1, &st, nil)
        sqlite3_step(st)
        XCTAssertEqual(sqlite3_column_int(st, 0), 1, "and its tables")
        sqlite3_finalize(st)
        sqlite3_close(db)
    }
}
