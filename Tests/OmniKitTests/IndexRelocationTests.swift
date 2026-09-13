import XCTest
@testable import OmniKit

/// Moving the index is the most destructive thing Settings can do: it decides whether a library
/// that took hours to build survives. The button used to write the new path into the preferences
/// and re-open, so a folder with no room or no write access left an empty index at the new location
/// and the old one stranded - tens of gigabytes, with nothing in the app pointing back at it.
///
/// Every check below runs BEFORE anything is copied and before the setting is touched.
final class IndexRelocationTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-reloc-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    /// An index folder holding the real file set, with recognisable contents.
    private func makeIndex(_ name: String, bytesEach: Int = 64) throws -> URL {
        let dir = root.appendingPathComponent(name, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for f in IndexRelocation.fileNames {
            try Data(repeating: UInt8(f.count % 251), count: bytesEach)
                .write(to: dir.appendingPathComponent(f))
        }
        return dir
    }

    func testItFindsExactlyTheIndexFiles() throws {
        let dir = try makeIndex("src")
        // A stranger in the same folder must not be picked up.
        try "notes".write(to: dir.appendingPathComponent("my-notes.txt"), atomically: true, encoding: .utf8)
        try "temp".write(to: dir.appendingPathComponent("index.sqlite.vecs.new"), atomically: true, encoding: .utf8)
        let found = IndexRelocation.files(in: dir).map(\.lastPathComponent).sorted()
        XCTAssertEqual(found, IndexRelocation.fileNames.sorted())
        XCTAssertFalse(found.contains("my-notes.txt"))
        XCTAssertFalse(found.contains("index.sqlite.vecs.new"),
                       "a compaction temp was carried across; it means nothing outside its compaction")
    }

    func testTheSameFolderIsRefused() throws {
        let dir = try makeIndex("src")
        XCTAssertNotNil(IndexRelocation.refusal(from: dir, to: dir, payload: 0))
    }

    /// Nesting either way. Copying into a child duplicates forever; choosing a parent makes the old
    /// copy a child of the new home, where a later cleanup would take both.
    func testNestedFoldersAreRefusedBothWays() throws {
        let dir = try makeIndex("src")
        let child = dir.appendingPathComponent("inner", isDirectory: true)
        XCTAssertNotNil(IndexRelocation.refusal(from: dir, to: child, payload: 0))
        XCTAssertNotNil(IndexRelocation.refusal(from: dir, to: root, payload: 0),
                        "a parent of the current index folder was accepted")
    }

    /// The reason the check writes a probe instead of asking `isWritableFile`.
    func testAnUnwritableFolderIsRefused() throws {
        let dir = try makeIndex("src")
        let locked = root.appendingPathComponent("locked", isDirectory: true)
        try FileManager.default.createDirectory(at: locked, withIntermediateDirectories: true)
        try FileManager.default.setAttributes([.posixPermissions: 0o500], ofItemAtPath: locked.path)
        defer { try? FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: locked.path) }
        let refusal = IndexRelocation.refusal(from: dir, to: locked, payload: 0)
        XCTAssertNotNil(refusal, "a read-only folder was accepted as an index destination")
        XCTAssertTrue(refusal?.contains("cannot write") ?? false, "got: \(refusal ?? "nil")")
    }

    func testTooLittleSpaceIsRefused() throws {
        let dir = try makeIndex("src")
        let dst = root.appendingPathComponent("dst", isDirectory: true)
        let payload: Int64 = 21_000_000_000
        XCTAssertNotNil(IndexRelocation.refusal(from: dir, to: dst, payload: payload,
                                                freeBytes: { _ in payload }),
                        "an exactly-full volume was accepted - the 5% headroom is what makes this safe")
        XCTAssertNil(IndexRelocation.refusal(from: dir, to: dst, payload: payload,
                                             freeBytes: { _ in payload * 2 }))
    }

    func testACleanMoveCopiesEveryFileIntact() throws {
        let src = try makeIndex("src", bytesEach: 4096)
        let dst = root.appendingPathComponent("dst", isDirectory: true)
        XCTAssertNil(IndexRelocation.refusal(from: src, to: dst,
                                             payload: IndexRelocation.byteSize(of: IndexRelocation.files(in: src))))
        try IndexRelocation.copy(from: src, to: dst)
        for f in IndexRelocation.fileNames {
            let a = try Data(contentsOf: src.appendingPathComponent(f))
            let b = try Data(contentsOf: dst.appendingPathComponent(f))
            XCTAssertEqual(a, b, "\(f) differs after the move")
        }
        // The source is left in place: deleting tens of gigabytes on the user's behalf, right after
        // a move they may still be verifying, is not something to do silently.
        XCTAssertEqual(IndexRelocation.files(in: src).count, IndexRelocation.fileNames.count)
    }

    /// A failure partway through must leave NOTHING half-written at the destination, and the
    /// source untouched. Injected by making one SOURCE file unreadable: the files before it copy,
    /// then the set throws, which is exactly the shape of a disk filling up mid-move.
    func testAFailedCopyCleansUpAndKeepsTheSource() throws {
        let src = try makeIndex("src", bytesEach: 1024)
        let dst = root.appendingPathComponent("dst", isDirectory: true)
        let blocked = src.appendingPathComponent("index.sqlite.vecs")
        try FileManager.default.setAttributes([.posixPermissions: 0o000], ofItemAtPath: blocked.path)
        defer { try? FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: blocked.path) }

        XCTAssertThrowsError(try IndexRelocation.copy(from: src, to: dst),
                             "an unreadable source file did not fail the move")
        XCTAssertEqual(IndexRelocation.files(in: src).count, IndexRelocation.fileNames.count,
                       "the source lost files to a failed move")
        XCTAssertTrue(IndexRelocation.files(in: dst).isEmpty,
                      "a failed move left \(IndexRelocation.files(in: dst).map(\.lastPathComponent)) behind")
    }
}
