import XCTest
@testable import OmniKit

/// The clipboard history is a folder of ordinary files; these pin what goes into it and what does
/// not, and that a folder of that shape is crawled although it lives in Omni's own data.
final class ClipboardHistoryTests: XCTestCase {
    private var dir: URL!
    private var history: ClipboardHistory!

    override func setUpWithError() throws {
        dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("clip-\(UUID().uuidString)", isDirectory: true)
        history = ClipboardHistory(directory: dir.appendingPathComponent("Clipboard"))
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: dir)
    }

    private func names() -> [String] {
        ((try? FileManager.default.contentsOfDirectory(atPath: history.directory.path)) ?? [])
            .filter { !$0.hasPrefix(".") }.sorted()
    }

    private func date(_ s: String) -> Date {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyy-MM-dd HH:mm:ss"
        return f.date(from: s)!
    }

    func testTextIsStoredUnderItsTimeAndFirstWords() throws {
        let url = try XCTUnwrap(history.save(.text("  Quarterly numbers: see /tmp\nsecond line"),
                                             at: date("2026-09-29 19:33:02")))
        XCTAssertEqual(url.lastPathComponent, "2026-09-29 19.33.02 Quarterly numbers see tmp.txt")
        XCTAssertEqual(try String(contentsOf: url, encoding: .utf8), "  Quarterly numbers: see /tmp\nsecond line")
    }

    func testImageIsStoredAsPNG() throws {
        let png = Data([0x89, 0x50, 0x4E, 0x47, 1, 2, 3])
        let url = try XCTUnwrap(history.save(.image(png), at: date("2026-09-29 19:33:02")))
        XCTAssertEqual(url.lastPathComponent, "2026-09-29 19.33.02 Image.png")
        XCTAssertEqual(try Data(contentsOf: url), png)
    }

    func testWhitespaceAndEmptyImagesAreNotStored() throws {
        XCTAssertNil(try history.save(.text(" \n\t ")))
        XCTAssertNil(try history.save(.image(Data())))
        XCTAssertEqual(names(), [])
    }

    func testSeparatorsInTitlesBecomeSpaces() {
        XCTAssertEqual(ClipboardHistory.title(for: "https://example.com/quarterly-report.pdf"),
                       "https example.com quarterly-report.pdf")
        XCTAssertEqual(ClipboardHistory.title(for: "a\\b:c"), "a b c")
    }

    func testLongTitlesAreCutAtAWord() {
        let t = ClipboardHistory.title(for: "one two three four five six seven eight nine ten eleven")
        XCTAssertLessThanOrEqual(t.count, 40)
        XCTAssertFalse(t.hasSuffix(" "))
        XCTAssertTrue("one two three four five six seven eight nine ten eleven".hasPrefix(t))
    }

    func testTheSameContentAgainMovesTheOneFileToTheNewTime() throws {
        try history.save(.text("hello"), at: date("2026-09-29 10:00:00"))
        try history.save(.text("other"), at: date("2026-09-29 10:00:05"))
        let again = try XCTUnwrap(history.save(.text("hello"), at: date("2026-09-29 11:00:00")))
        XCTAssertEqual(names(), ["2026-09-29 10.00.05 other.txt", "2026-09-29 11.00.00 hello.txt"])
        XCTAssertEqual(again.lastPathComponent, "2026-09-29 11.00.00 hello.txt")
        let mtime = try again.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate
        XCTAssertEqual(mtime, date("2026-09-29 11:00:00"))
    }

    func testDuplicatesAreFoundByAFreshInstance() throws {
        try history.save(.text("hello"), at: date("2026-09-29 10:00:00"))
        let reopened = ClipboardHistory(directory: history.directory)
        try reopened.save(.text("hello"), at: date("2026-09-29 12:00:00"))
        XCTAssertEqual(names(), ["2026-09-29 12.00.00 hello.txt"])
    }

    func testTwoClipsInOneSecondKeepBoth() throws {
        let t = date("2026-09-29 10:00:00")
        try history.save(.text("same title a"), at: t)
        try history.save(.text("same title a\nbut different body"), at: t)
        XCTAssertEqual(names().count, 2)
    }

    func testPruneDeletesOnlyOldClips() throws {
        let now = date("2026-09-29 12:00:00")
        try history.save(.text("old"), at: now.addingTimeInterval(-31 * 86_400))
        try history.save(.text("new"), at: now.addingTimeInterval(-1 * 86_400))
        XCTAssertEqual(history.prune(olderThanDays: 30, now: now), 1)
        XCTAssertEqual(names().count, 1)
        XCTAssertTrue(names()[0].hasSuffix("new.txt"))
        XCTAssertEqual(history.prune(olderThanDays: 0, now: now), 0, "zero keeps everything")
    }

    func testClearRemovesTheFolder() throws {
        try history.save(.text("x"))
        try history.clear()
        XCTAssertFalse(FileManager.default.fileExists(atPath: history.directory.path))
        XCTAssertEqual(history.count, 0)
    }

    func testConcealedAndTransientItemsAreSkipped() {
        XCTAssertTrue(ClipboardHistory.shouldSkip(types: ["public.utf8-plain-text", "org.nspasteboard.ConcealedType"]))
        XCTAssertTrue(ClipboardHistory.shouldSkip(types: ["org.nspasteboard.TransientType"]))
        XCTAssertFalse(ClipboardHistory.shouldSkip(types: ["public.utf8-plain-text"]))
    }

    /// The clipboard folder sits inside Omni's Application Support folder, which the crawl skips
    /// as Omni's own data. It is user content, so it is crawled all the same.
    func testTheClipboardFolderIsCrawledInsideOwnData() throws {
        let support = dir.appendingPathComponent("Omni")
        let clip = ClipboardHistory(directory: support.appendingPathComponent("Clipboard"))
        try clip.save(.text("a clipped passage"))
        try FileManager.default.createDirectory(at: support.appendingPathComponent("Transcripts"),
                                                withIntermediateDirectories: true)
        try "internal".write(to: support.appendingPathComponent("Transcripts/t.md"), atomically: true, encoding: .utf8)

        func crawl(_ exceptions: [String]) -> [String] {
            var found: [String] = []
            let c = FileCrawler(roots: [clip.directory, support], ownDataPaths: [support.path],
                                ownDataExceptions: exceptions)
            c.walk(shouldContinue: { true }) { found.append(($0.path as NSString).lastPathComponent) }
            return found
        }
        XCTAssertEqual(crawl([]), [], "without the exception the whole folder is own data")
        let found = crawl([clip.directory.path])
        XCTAssertEqual(found.filter { $0.hasSuffix(".txt") }.count, 1)
        XCTAssertFalse(found.contains("t.md"), "the rest of Omni's data stays out")
    }
}

/// Ignore rules apply BELOW a folder the user added, never to the folders above it: the crawl does
/// not test a root or its ancestors, and the watcher and the prune must agree with it.
final class IgnoreRootBoundTests: XCTestCase {
    private let rules = OmniIgnore(text: "Library/\nnode_modules/\n")
    private let root = "/Users/me/Library/Mobile Documents/com~apple~CloudDocs"

    func testAWatcherEventUnderARootInsideLibraryIsKept() {
        let path = root + "/Notes/plan.txt"
        XCTAssertTrue(rules.isIgnoredIncludingAncestors(path, isDir: false),
                      "without the root, the Library ancestor excludes it")
        XCTAssertFalse(rules.isIgnoredIncludingAncestors(path, isDir: false, root: root))
    }

    func testRulesStillApplyBelowTheRoot() {
        XCTAssertTrue(rules.isIgnoredIncludingAncestors(root + "/app/node_modules/x.js", isDir: false, root: root))
        XCTAssertTrue(rules.isIgnoredIncludingAncestors(root + "/Library/x.txt", isDir: false, root: root))
    }

    func testThePruneStopsAtTheRoot() {
        let path = root + "/Notes/plan.txt"
        XCTAssertTrue(rules.excludesIndexedFile()(path))
        XCTAssertFalse(rules.excludesIndexedFile(roots: [root])(path))
        XCTAssertTrue(rules.excludesIndexedFile(roots: [root])(root + "/node_modules/a.js"))
    }
}

/// Served search passes the clipboard folder in `excludeFolders`: nothing under it may answer,
/// on any route, while the same query in the app still finds it.
final class ExcludeFoldersTests: XCTestCase {
    private func store() throws -> VectorStore {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("excl-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return try VectorStore(dbURL: dir.appendingPathComponent("index.sqlite"))
    }

    private func write(_ s: VectorStore, _ path: String, _ hot: Int) throws {
        var v = [Float](repeating: 0.01, count: 8)
        v[hot] = 1
        try s.replace(path: path, chunks: [IndexedChunk(path: path, modified: 1, size: 1, kind: "text",
                                                        chunkIndex: 0, snippet: "s", embedding: v)])
    }

    func testExcludedFolderNeverAnswers() throws {
        let s = try store()
        defer { s.close() }
        try write(s, "/u/Clipboard/2026-09-29 10.00.00 secret.txt", 0)
        try write(s, "/u/Clipboardish/near.txt", 0)          // a sibling that only starts the same
        try write(s, "/u/docs/a.txt", 1)
        var q = [Float](repeating: 0, count: 8); q[0] = 1
        var f = SearchFilter()
        f.minScore = 0
        XCTAssertTrue(s.search(q, filter: f, topK: 10).map(\.path).contains("/u/Clipboard/2026-09-29 10.00.00 secret.txt"))
        f.excludeFolders = ["/u/Clipboard"]
        let hits = s.search(q, filter: f, topK: 10).map(\.path)
        XCTAssertFalse(hits.contains { $0.hasPrefix("/u/Clipboard/") })
        XCTAssertTrue(hits.contains("/u/Clipboardish/near.txt"))
        XCTAssertTrue(hits.contains("/u/docs/a.txt"))
        // Scoping INTO the excluded folder still returns nothing from it.
        f.folderPrefix = "/u/Clipboard"
        XCTAssertEqual(s.search(q, filter: f, topK: 10).map(\.path), [])
    }
}

extension ClipboardHistoryTests {
    func testExistingFileFindsAStoredClipWithoutWriting() throws {
        XCTAssertNil(history.existingFile(for: .text("not yet")))
        let url = try XCTUnwrap(history.save(.text("stored once")))
        XCTAssertEqual(ClipboardHistory(directory: history.directory).existingFile(for: .text("stored once")), url)
        XCTAssertNil(history.existingFile(for: .text("stored twice")))
        XCTAssertEqual(names().count, 1)
    }
}
