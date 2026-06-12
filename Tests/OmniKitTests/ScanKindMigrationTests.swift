import XCTest
@testable import OmniKit

/// The one-time 'scan' kind migration: indexes built before the scan kind stored scanned-PDF
/// pages as kind='text' with an exact signature (snippet == file name on EVERY chunk). The
/// store re-labels those rows to kind='scan' on open - a metadata UPDATE, no vectors touched -
/// and must leave every other row alone.
final class ScanKindMigrationTests: XCTestCase {
    private func chunk(_ path: String, _ idx: Int, snippet: String, locator: String, vec: [Float]) -> IndexedChunk {
        IndexedChunk(path: path, modified: 1, size: 10, kind: "text", chunkIndex: idx,
                     snippet: snippet, embedding: vec, locator: locator)
    }

    func testLegacyScannedPDFRowsAreRelabeled() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("scan-mig-\(UUID().uuidString)")
        let dbURL = dir.appendingPathComponent("test.sqlite")
        let store = try VectorStore(dbURL: dbURL)
        // Legacy rows, all kind='text' the way a pre-scan-kind indexer wrote them:
        // a multi-page scan (every snippet == file name), a single-page scan, a text PDF
        // (real excerpts), a defensive half-signed PDF, and a .txt whose content happens to
        // BE its own name (signature match, but not a PDF - must stay text).
        try store.replace(path: "/tmp/scan.pdf", chunks: [
            chunk("/tmp/scan.pdf", 0, snippet: "scan.pdf", locator: "Page 1", vec: [1, 0, 0, 0]),
            chunk("/tmp/scan.pdf", 1, snippet: "scan.pdf", locator: "Page 2", vec: [0, 1, 0, 0]),
        ])
        try store.replace(path: "/tmp/single.pdf", chunks: [
            chunk("/tmp/single.pdf", 0, snippet: "single.pdf", locator: "", vec: [0, 0, 1, 0]),
        ])
        try store.replace(path: "/tmp/textdoc.pdf", chunks: [
            chunk("/tmp/textdoc.pdf", 0, snippet: "Quarterly revenue grew by", locator: "Page 1", vec: [0, 0, 0, 1]),
            chunk("/tmp/textdoc.pdf", 1, snippet: "and the outlook for next year", locator: "Page 2", vec: [1, 1, 0, 0]),
        ])
        try store.replace(path: "/tmp/half.pdf", chunks: [
            chunk("/tmp/half.pdf", 0, snippet: "half.pdf", locator: "Page 1", vec: [1, 0, 1, 0]),
            chunk("/tmp/half.pdf", 1, snippet: "an actual extracted excerpt", locator: "Page 2", vec: [0, 1, 1, 0]),
        ])
        try store.replace(path: "/tmp/notes.txt", chunks: [
            chunk("/tmp/notes.txt", 0, snippet: "notes.txt", locator: "", vec: [1, 0, 0, 1]),
        ])
        // Legacy v0.1.0-v0.1.45 snippet formats: "name.pdf - page N" (multi-page) and
        // "name.pdf - name.pdf" (single page) - the dominant real-world cohort.
        try store.replace(path: "/tmp/legacy-multi.pdf", chunks: [
            chunk("/tmp/legacy-multi.pdf", 0, snippet: "legacy-multi.pdf - page 1", locator: "", vec: [0, 1, 0, 1]),
            chunk("/tmp/legacy-multi.pdf", 1, snippet: "legacy-multi.pdf - page 2", locator: "", vec: [0, 0, 1, 1]),
        ])
        try store.replace(path: "/tmp/legacy-single.pdf", chunks: [
            chunk("/tmp/legacy-single.pdf", 0, snippet: "legacy-single.pdf - legacy-single.pdf", locator: "", vec: [1, 1, 1, 0]),
        ])
        // A text PDF whose excerpt merely STARTS like the legacy page form must stay text:
        // the suffix after "- page " is not an integer.
        try store.replace(path: "/tmp/pagey.pdf", chunks: [
            chunk("/tmp/pagey.pdf", 0, snippet: "pagey.pdf - page layout principles for designers", locator: "Page 1", vec: [0, 1, 1, 1]),
        ])
        // The store was created flag-done (a fresh DB has nothing to migrate); reset the flag so
        // the reopen below behaves exactly like a pre-scan-kind index meeting the new app.
        store.metaSet("scan_kind_migrated", "0")
        store.close()

        let migrated = try VectorStore(dbURL: dbURL)
        defer { migrated.close() }
        XCTAssertEqual(migrated.metaGet("scan_kind_migrated"), "1", "migration must mark itself done")
        XCTAssertEqual(migrated.fileCount(kinds: ["scan"]), 4, "current + both legacy formats re-labeled")
        XCTAssertEqual(migrated.fileCount(kinds: ["text"]), 4, "text PDF, half-signed, page-prefix excerpt, and .txt untouched")

        // Filter semantics at the STORE level are exact (no superset here - that lives in the
        // app/serving layer): 'scan' selects only the scans, 'text' only the text rows.
        var scanOnly = SearchFilter(); scanOnly.kinds = ["scan"]
        let scanHits = Set(migrated.search([1, 0, 0, 0], filter: scanOnly, topK: 10).map { $0.path })
        XCTAssertEqual(scanHits, ["/tmp/scan.pdf", "/tmp/single.pdf", "/tmp/legacy-multi.pdf", "/tmp/legacy-single.pdf"])
        var textOnly = SearchFilter(); textOnly.kinds = ["text"]
        let textHits = Set(migrated.search([1, 0, 0, 0], filter: textOnly, topK: 10).map { $0.path })
        XCTAssertEqual(textHits, ["/tmp/textdoc.pdf", "/tmp/half.pdf", "/tmp/pagey.pdf", "/tmp/notes.txt"])

        // Vectors and locators rode through untouched.
        XCTAssertEqual(migrated.search([0, 1, 0, 0], filter: scanOnly, topK: 1).first?.locator, "Page 2")
        XCTAssertEqual(migrated.rankChunks([1, 0, 0, 0], path: "/tmp/scan.pdf").first?.locator, "Page 1")
    }

    func testMigrationRunsOnce() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("scan-mig-once-\(UUID().uuidString)")
        let dbURL = dir.appendingPathComponent("test.sqlite")
        let store = try VectorStore(dbURL: dbURL)
        store.metaSet("scan_kind_migrated", "0")
        store.close()
        let second = try VectorStore(dbURL: dbURL)
        XCTAssertEqual(second.metaGet("scan_kind_migrated"), "1")
        // Rows written AFTER the flag is set (e.g. by an older app version) are NOT re-scanned -
        // the migration is strictly one-time; the new indexer writes kind='scan' directly.
        try second.replace(path: "/tmp/late.pdf", chunks: [
            chunk("/tmp/late.pdf", 0, snippet: "late.pdf", locator: "Page 1", vec: [1, 0, 0, 0]),
        ])
        second.close()
        let third = try VectorStore(dbURL: dbURL)
        defer { third.close() }
        XCTAssertEqual(third.fileCount(kind: "scan"), 0, "one-time: post-flag legacy writes stay text")
        XCTAssertEqual(third.fileCount(kind: "text"), 1)
    }
}
