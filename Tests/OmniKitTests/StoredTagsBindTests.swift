import XCTest
import SQLite3
@testable import OmniKit

/// `storedTags(paths:)` returned an empty dictionary for every path.
///
/// `StoreSchema.fileIDByPath` is a TWO-parameter form - `d.path = ? AND f.name = ?` - and this one
/// call site bound the whole path to the first parameter and never bound the second. An unbound
/// parameter is NULL, `f.name = NULL` is never true, so the subquery matched no file and the
/// statement stepped zero rows. Every other call site uses `bindPath`, which binds the pair.
///
/// It fails silently in both directions: absent from the result means "not media" to the caller, so
/// the browser's Tags column drew blanks and the serving layer reported untagged files, neither of
/// them as an error. Confirmed against the real index before the fix: 0 rows for a file whose
/// `chunk_text` row holds five tags.
///
/// A SECOND defect sat in the same statement and was fatal on its own: `ORDER BY chunk_index`
/// against a table that has no such column, so the statement never prepared and every slice took
/// the `guard ... == SQLITE_OK else { return }` early exit. Both are covered here, because either
/// one alone produces exactly the same silent "no tags".
final class StoredTagsBindTests: XCTestCase {

    private func tempDB() -> URL {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("tagbind-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("index.sqlite")
    }

    private func mediaChunk(_ path: String, tags: String) -> IndexedChunk {
        var v = [Float](repeating: 0, count: 8)
        v[0] = 1
        // ", "-joined is the separator the `tag:` filter normalizes on - see storedTags.
        return IndexedChunk(path: path, modified: 1, size: 1, kind: "image",
                            chunkIndex: 0, snippet: tags, embedding: v,
                            width: 100, height: 100)
    }

    func testTagsComeBackForAStoredMediaFile() throws {
        let store = try VectorStore(dbURL: tempDB())
        defer { store.close() }
        try store.replace(path: "/pics/holiday/beach.png",
                          chunks: [mediaChunk("/pics/holiday/beach.png", tags: "sand, surf, sunset")])

        let tags = store.storedTags(paths: ["/pics/holiday/beach.png"])
        XCTAssertEqual(tags["/pics/holiday/beach.png"], ["sand", "surf", "sunset"],
                       "storedTags returned \(tags) - the path bind is wrong again")
    }

    /// Several files across several directories in one batch: the bug was in the per-path bind, so
    /// a single-path test could in principle pass on a fluke of the previous statement's bindings.
    func testABatchAcrossDirectories() throws {
        let store = try VectorStore(dbURL: tempDB())
        defer { store.close() }
        let paths = ["/pics/a/one.png", "/pics/b/two.png", "/pics/b/three.png"]
        for (i, p) in paths.enumerated() {
            try store.replace(path: p, chunks: [mediaChunk(p, tags: "tag\(i), shared")])
        }
        let tags = store.storedTags(paths: paths)
        XCTAssertEqual(tags.count, 3, "got \(tags.count) of 3 files back")
        for (i, p) in paths.enumerated() {
            XCTAssertEqual(tags[p], ["tag\(i)", "shared"], "wrong tags for \(p)")
        }
    }

    /// A file with the same basename in two directories must not collide - which is the case the
    /// two-parameter form exists for.
    func testSameNameInTwoFoldersStaysDistinct() throws {
        let store = try VectorStore(dbURL: tempDB())
        defer { store.close() }
        try store.replace(path: "/pics/a/img.png", chunks: [mediaChunk("/pics/a/img.png", tags: "alpha")])
        try store.replace(path: "/pics/b/img.png", chunks: [mediaChunk("/pics/b/img.png", tags: "beta")])
        let tags = store.storedTags(paths: ["/pics/a/img.png", "/pics/b/img.png"])
        XCTAssertEqual(tags["/pics/a/img.png"], ["alpha"])
        XCTAssertEqual(tags["/pics/b/img.png"], ["beta"])
    }

    /// A non-media file is ABSENT, not empty: the caller distinguishes "media with no tags yet"
    /// from "not a media file at all" by key presence, so the fix must not turn one into the other.
    func testATextFileIsAbsentRatherThanEmpty() throws {
        let store = try VectorStore(dbURL: tempDB())
        defer { store.close() }
        var v = [Float](repeating: 0, count: 8); v[0] = 1
        try store.replace(path: "/docs/notes.txt",
                          chunks: [IndexedChunk(path: "/docs/notes.txt", modified: 1, size: 1,
                                                kind: "text", chunkIndex: 0, snippet: "some prose",
                                                embedding: v)])
        let tags = store.storedTags(paths: ["/docs/notes.txt"])
        XCTAssertNil(tags["/docs/notes.txt"], "a text file was reported as untagged media")
    }

    /// The browser reads tags through `browseTags(inFolder:)` on the read connection, not through
    /// `storedTags(paths:)` on the writer's queue. Two code paths over one table means they can
    /// drift, so the parity is asserted rather than assumed.
    func testTheBrowsePathAgreesWithTheServingPath() throws {
        let store = try VectorStore(dbURL: tempDB())
        defer { store.close() }
        let paths = ["/pics/a/one.png", "/pics/a/two.png", "/pics/a/three.png"]
        for (i, p) in paths.enumerated() {
            try store.replace(path: p, chunks: [mediaChunk(p, tags: "tag\(i), shared")])
        }
        // A text file in the same folder: absent from both, not empty in one and absent in the other.
        var v = [Float](repeating: 0, count: 8); v[0] = 1
        try store.replace(path: "/pics/a/notes.txt",
                          chunks: [IndexedChunk(path: "/pics/a/notes.txt", modified: 1, size: 1,
                                                kind: "text", chunkIndex: 0, snippet: "prose",
                                                embedding: v)])
        let browse = store.browseTags(inFolder: "/pics/a")
        let serving = store.storedTags(paths: paths + ["/pics/a/notes.txt"])
        XCTAssertEqual(browse, serving, "the browse and serving tag paths disagree")
        XCTAssertEqual(browse["/pics/a/one.png"], ["tag0", "shared"])
        XCTAssertNil(browse["/pics/a/notes.txt"], "a text file came back from the browse path")
    }

    /// A file in a SUBFOLDER must not leak into the parent's listing - the browse query is scoped
    /// to one directory row, and `dirs` holds the whole subtree.
    func testASubfolderDoesNotLeakIntoTheParent() throws {
        let store = try VectorStore(dbURL: tempDB())
        defer { store.close() }
        try store.replace(path: "/pics/a/here.png", chunks: [mediaChunk("/pics/a/here.png", tags: "near")])
        try store.replace(path: "/pics/a/deep/there.png", chunks: [mediaChunk("/pics/a/deep/there.png", tags: "far")])
        let browse = store.browseTags(inFolder: "/pics/a")
        XCTAssertEqual(Array(browse.keys), ["/pics/a/here.png"], "the subtree leaked into the folder")
    }

    /// A filename-derived snippet is not a tag. Both paths drop it, and they have to drop it the
    /// same way or the Tags column shows the filename back to the user.
    ///
    /// The snippet is the basename WITH its extension - that is the form OmniTagger recognises, and
    /// the form a media row carries before anything has tagged it.
    func testFilenameDerivedSnippetsAreNotTags() throws {
        let store = try VectorStore(dbURL: tempDB())
        defer { store.close() }
        try store.replace(path: "/pics/a/sunset.png",
                          chunks: [mediaChunk("/pics/a/sunset.png", tags: "sunset.png")])
        let browse = store.browseTags(inFolder: "/pics/a")
        let serving = store.storedTags(paths: ["/pics/a/sunset.png"])
        XCTAssertEqual(browse, serving)
        XCTAssertEqual(browse["/pics/a/sunset.png"], [],
                       "a name-derived snippet was reported as a tag")
    }
}
