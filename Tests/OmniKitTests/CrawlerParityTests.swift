import XCTest
@testable import OmniKit

/// THE FAST WALK MUST SEE EXACTLY WHAT THE OLD ONE SAW.
///
/// Replacing FileManager's enumerator with getattrlistbulk buys 8x, and it also takes over every
/// policy the enumerator used to apply for free: hidden files, package bundles, symlinks, volume
/// boundaries. Each of those is a way to silently index MORE than the user asked for (walking into
/// a .photoslibrary) or LESS (dropping a whole folder) - and neither shows up as an error, only as
/// a file count nobody checks.
///
/// So the test is not "does the new walk work". It is "do the two walks return the identical set",
/// asserted on a tree built to contain every case that differs between them.
final class CrawlerParityTests: XCTestCase {
    private var dir: URL!

    override func setUpWithError() throws {
        dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("crawl-parity-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: dir)
    }

    private func write(_ rel: String, _ text: String = "hello world") throws {
        let u = dir.appendingPathComponent(rel)
        try FileManager.default.createDirectory(at: u.deletingLastPathComponent(), withIntermediateDirectories: true)
        try text.write(to: u, atomically: true, encoding: .utf8)
    }

    private func crawl(legacy: Bool, ignore: OmniIgnore = OmniIgnore(text: ""),
                       kinds: Set<FileKind> = [.text, .image, .video, .audio],
                       caps: [FileKind: Int] = FileCrawler.defaultMaxFileSize) -> [String: CrawledFile] {
        setenv("OMNI_CRAWLER", legacy ? "legacy" : "bulk", 1)
        defer { unsetenv("OMNI_CRAWLER") }
        var out: [String: CrawledFile] = [:]
        FileCrawler(roots: [dir], ignore: ignore, enabledKinds: kinds, maxFileSize: caps)
            .walk { out[$0.path] = $0 }
        return out
    }

    private func assertParity(_ label: String, ignore: OmniIgnore = OmniIgnore(text: ""),
                              kinds: Set<FileKind> = [.text, .image, .video, .audio],
                              caps: [FileKind: Int] = FileCrawler.defaultMaxFileSize,
                              file: StaticString = #filePath, line: UInt = #line) {
        let old = crawl(legacy: true, ignore: ignore, kinds: kinds, caps: caps)
        let new = crawl(legacy: false, ignore: ignore, kinds: kinds, caps: caps)
        let onlyOld = Set(old.keys).subtracting(new.keys).sorted()
        let onlyNew = Set(new.keys).subtracting(old.keys).sorted()
        XCTAssertTrue(onlyOld.isEmpty, "\(label): the fast walk MISSED \(onlyOld.count): \(onlyOld.prefix(5))",
                      file: file, line: line)
        XCTAssertTrue(onlyNew.isEmpty, "\(label): the fast walk added \(onlyNew.count): \(onlyNew.prefix(5))",
                      file: file, line: line)
        // Same files is not enough - the incremental check compares mtime and size, so a walk that
        // returns the right paths with the wrong stamps re-embeds the world or skips a changed file.
        for (path, o) in old {
            guard let n = new[path] else { continue }
            XCTAssertEqual(o.size, n.size, "\(label): size differs for \(path)", file: file, line: line)
            XCTAssertEqual(o.modified, n.modified, accuracy: 0.000_01,
                           "\(label): mtime differs for \(path)", file: file, line: line)
        }
    }

    /// The ordinary tree, plus the two the walk has to actively exclude: hidden files and hidden
    /// directories. The enumerator gets those from .skipsHiddenFiles; the fast walk has to know.
    func testOrdinaryTreeAndHiddenEntries() throws {
        try write("a.txt")
        try write("nested/b.txt")
        try write("nested/deeper/c.txt")
        try write(".hidden.txt")
        try write(".hiddendir/d.txt")
        try write("nested/.also-hidden.txt")
        assertParity("ordinary + hidden")

        let seen = crawl(legacy: false)
        XCTAssertFalse(seen.keys.contains { $0.contains("/.hidden") || $0.contains("/.also-hidden") },
                       "a hidden file was indexed")
        XCTAssertEqual(seen.count, 3, "expected exactly the three visible files")
    }

    /// A PACKAGE is a directory the user thinks of as one file. Walking into it indexes hundreds of
    /// internal resources nobody searched for - and .app bundles are full of text.
    func testPackageDirectoryIsNotDescendedInto() throws {
        try write("normal.txt")
        try write("MyApp.app/Contents/Resources/inside.txt")
        try write("Photos.photoslibrary/originals/inside.txt")
        assertParity("packages")

        let seen = crawl(legacy: false)
        XCTAssertFalse(seen.keys.contains { $0.contains(".app/") }, "walked into an .app bundle")
        XCTAssertFalse(seen.keys.contains { $0.contains(".photoslibrary/") }, "walked into a photo library")
    }

    /// Symlinks: a loop is the failure that matters, because a walk that follows them never ends.
    func testSymlinksAreNotFollowed() throws {
        try write("real/x.txt")
        try FileManager.default.createSymbolicLink(at: dir.appendingPathComponent("link-to-real"),
                                                   withDestinationURL: dir.appendingPathComponent("real"))
        try FileManager.default.createSymbolicLink(at: dir.appendingPathComponent("loop"),
                                                   withDestinationURL: dir)
        try FileManager.default.createSymbolicLink(at: dir.appendingPathComponent("link-to-file.txt"),
                                                   withDestinationURL: dir.appendingPathComponent("real/x.txt"))
        assertParity("symlinks")
    }

    /// .omniignore applies to directories (never descended into - where the saving comes from) and
    /// to files. Both engines take the same OmniIgnore, so this is about WHERE they consult it.
    func testIgnoreRulesMatch() throws {
        try write("keep/one.txt")
        try write("node_modules/dep/index.txt")
        try write("build/output.txt")
        try write("keep/skipme.txt")
        let ignore = OmniIgnore(text: "node_modules/\nbuild/\nskipme.txt\n")
        assertParity("ignore rules", ignore: ignore)

        let seen = crawl(legacy: false, ignore: ignore)
        XCTAssertEqual(Set(seen.keys.map { ($0 as NSString).lastPathComponent }), ["one.txt"],
                       "ignore rules did not exclude what they name")
    }

    /// The per-kind size cap, and the kind filter itself - both read the size the walk returns, so
    /// a wrong size silently changes which files are indexed.
    func testKindFilterAndSizeCapMatch() throws {
        try write("small.txt", String(repeating: "x", count: 100))
        try write("big.txt", String(repeating: "x", count: 50_000))
        assertParity("text only", kinds: [.text])
        assertParity("capped", kinds: [.text], caps: [.text: 1_000])

        let capped = crawl(legacy: false, kinds: [.text], caps: [.text: 1_000])
        XCTAssertEqual(Set(capped.keys.map { ($0 as NSString).lastPathComponent }), ["small.txt"],
                       "the size cap was not applied")
    }

    /// A deep, wide tree: the pool's own bookkeeping (who is finished, when is the walk done) only
    /// shows up under real fan-out, and a lost directory is invisible in a small fixture.
    func testDeepWideTreeIsFullyWalked() throws {
        var expected = 0
        for a in 0 ..< 6 {
            for b in 0 ..< 6 {
                for c in 0 ..< 3 {
                    try write("d\(a)/d\(b)/f\(c).txt")
                    expected += 1
                }
            }
        }
        assertParity("deep tree")
        XCTAssertEqual(crawl(legacy: false).count, expected, "the pool lost files in a wide tree")
    }

    /// Cancellation has to stop the pool, not just the thread that noticed. Nothing may deadlock,
    /// and the walk must return promptly rather than draining every queued directory first.
    func testCancellationStopsPromptly() throws {
        for a in 0 ..< 8 {
            for b in 0 ..< 8 { try write("c\(a)/c\(b)/f.txt") }
        }
        var seen = 0
        let t = Date()
        FileCrawler(roots: [dir]).walk(shouldContinue: { seen < 2 }) { _ in seen += 1 }
        XCTAssertLessThan(-t.timeIntervalSinceNow, 5, "a cancelled walk did not return promptly")
        XCTAssertLessThan(seen, 64, "a cancelled walk kept going long past the cancel")
    }

    /// An unreadable directory must not take the walk down with it - the enumerator swallowed those
    /// through its errorHandler, and a crawl that stops at the first permission error would silently
    /// leave the rest of a root unindexed.
    func testUnreadableDirectoryIsSkippedNotFatal() throws {
        try write("readable/a.txt")
        try write("locked/b.txt")
        let locked = dir.appendingPathComponent("locked")
        try FileManager.default.setAttributes([.posixPermissions: 0], ofItemAtPath: locked.path)
        defer { try? FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: locked.path) }

        let seen = crawl(legacy: false)
        XCTAssertTrue(seen.keys.contains { $0.hasSuffix("readable/a.txt") },
                      "an unreadable sibling stopped the walk")
    }
}
