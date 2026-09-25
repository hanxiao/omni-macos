import XCTest
@testable import OmniKit

/// A `.omniignore` inside an indexed folder (issue #23). It is rewritten into central rules anchored
/// at its folder, so these tests pin the rewrite against git's rules for a nested `.gitignore`.
final class NestedIgnoreTests: XCTestCase {
    private func policy(central: String = "", _ nested: [(dir: String, text: String)]) -> OmniIgnore {
        OmniIgnore(text: ([central] + nested.map { OmniIgnore.scoped($0.text, to: $0.dir) }).joined(separator: "\n"))
    }

    func testASlashlessPatternMatchesAtAnyDepthBelowItsFolderOnly() {
        let g = policy([("/p/a", "*.json\n*.jsonl\n")])
        XCTAssertTrue(g.isIgnored("/p/a/x.json", isDir: false))
        XCTAssertTrue(g.isIgnored("/p/a/b/c/y.jsonl", isDir: false))
        XCTAssertFalse(g.isIgnored("/p/x.json", isDir: false))          // the folder's parent
        XCTAssertFalse(g.isIgnored("/p/ab/x.json", isDir: false))       // a sibling sharing a prefix
        XCTAssertFalse(g.isIgnored("/p/a/x.txt", isDir: false))
    }

    func testAPatternWithASlashIsRelativeToItsFolder() {
        let g = policy([("/p/a", "/build\nout/tmp/\n")])
        XCTAssertTrue(g.isIgnored("/p/a/build", isDir: true))
        XCTAssertTrue(g.isIgnored("/p/a/build/x.txt", isDir: false))
        XCTAssertFalse(g.isIgnored("/p/a/sub/build", isDir: true))
        XCTAssertTrue(g.isIgnored("/p/a/out/tmp", isDir: true))
        XCTAssertFalse(g.isIgnored("/p/a/out/tmp", isDir: false))        // trailing '/' = directories only
    }

    func testAFolderRuleOverridesTheCentralOne() {
        let g = policy(central: "*.json", [("/p/a", "!keep.json")])
        XCTAssertFalse(g.isIgnored("/p/a/keep.json", isDir: false))
        XCTAssertTrue(g.isIgnored("/p/a/other.json", isDir: false))
        XCTAssertTrue(g.isIgnored("/p/b/keep.json", isDir: false))      // only under that folder
    }

    func testGlobCharactersInTheFolderNameAreLiteral() {
        let g = policy([("/p/[draft] v?", "*.txt")])
        XCTAssertTrue(g.isIgnored("/p/[draft] v?/n.txt", isDir: false))
        XCTAssertFalse(g.isIgnored("/p/d/n.txt", isDir: false))
        XCTAssertFalse(g.isIgnored("/p/[draft] vx/n.txt", isDir: false))
    }

    func testCommentsAndBlanksProduceNothing() {
        XCTAssertEqual(OmniIgnore.scoped("# only a comment\n\n   \n", to: "/p/a"), "")
        XCTAssertEqual(OmniIgnore.scoped("*.log\r\n", to: "/p/a/"), "/p/a/**/*.log")
    }

    /// A file saved with Windows line endings. Swift reads "\r\n" as one Character, so a split on
    /// "\n" left the whole file as a single rule that matched nothing.
    func testCRLFFilesAreReadLineByLine() {
        let g = OmniIgnore(text: "*.json\r\n*.jsonl\r\n")
        XCTAssertTrue(g.isIgnored("/p/a/x.json", isDir: false))
        XCTAssertTrue(g.isIgnored("/p/a/x.jsonl", isDir: false))
        XCTAssertEqual(OmniIgnore.scoped("*.json\r\n*.jsonl\r\n", to: "/p/a"), "/p/a/**/*.json\n/p/a/**/*.jsonl")
    }

    func testAWatcherEventHonoursTheFolderRules() {
        let g = policy([("/p/a", "cache/\n")])
        XCTAssertTrue(g.isIgnoredIncludingAncestors("/p/a/x/cache/y.txt", isDir: false))
        XCTAssertFalse(g.isIgnoredIncludingAncestors("/p/b/cache/y.txt", isDir: false))
    }

    /// The crawl reports every `.omniignore` it walks past, which is how one that was already on
    /// disk before this version is found without a second walk.
    func testTheCrawlReportsFolderPolicyFiles() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("nested-ignore-\(UUID().uuidString)")
        let sub = root.appendingPathComponent("sub")
        try FileManager.default.createDirectory(at: sub, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try "*.json\n".write(to: sub.appendingPathComponent(".omniignore"), atomically: true, encoding: .utf8)
        try "hello".write(to: sub.appendingPathComponent("a.txt"), atomically: true, encoding: .utf8)

        let found = FoundDirs()
        var crawler = FileCrawler(roots: [root])
        crawler.onPolicyFile = { found.add($0) }
        var files: [String] = []
        crawler.walk { files.append($0.path) }
        let real = URL(fileURLWithPath: sub.path).resolvingSymlinksInPath().path
        XCTAssertEqual(found.all.map { URL(fileURLWithPath: $0).resolvingSymlinksInPath().path }, [real])
        XCTAssertFalse(files.contains { $0.hasSuffix("/.omniignore") })   // never indexed itself
    }
}

private final class FoundDirs: @unchecked Sendable {
    private let lock = NSLock()
    private var dirs: [String] = []
    func add(_ d: String) { lock.lock(); dirs.append(d); lock.unlock() }
    var all: [String] { lock.lock(); defer { lock.unlock() }; return dirs }
}
