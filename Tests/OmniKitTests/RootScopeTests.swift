import XCTest
@testable import OmniKit

/// Issue #18, second report: "I just added 6 different folders, then upon adding the parent
/// folder? It consumed all of them instead of literally indexing and being separate."
///
/// Confirmed, and the swallowing itself is correct: crawling a parent AND its children walks the
/// same files twice. What was wrong is that the six folders then vanished from the sidebar with no
/// word, which reads as "they are no longer indexed". They are - by the parent - and a search can
/// still be scoped to any of them, verified end to end over HTTP with the parent as the only
/// source: `folders: [alpha, gamma]` returned exactly alpha and gamma, `folder: epsilon` exactly
/// epsilon. So `covered` keeps them as names.
final class RootScopeTests: XCTestCase {

    private func urls(_ paths: [String]) -> [URL] { paths.map { URL(fileURLWithPath: $0) } }
    private func paths(_ urls: [URL]) -> [String] { urls.map(\.path) }

    /// The reporter's exact sequence: six folders, then their parent.
    func testTheParentBecomesTheOnlyRootAndTheSixAreKept() {
        let kids = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"].map { "/w/parent/\($0)" }
        let all = urls(kids + ["/w/parent"])
        XCTAssertEqual(paths(RootScope.canonical(all)), ["/w/parent"],
                       "the crawl set must be the parent alone - crawling both walks the same files twice")
        XCTAssertEqual(paths(RootScope.covered(all)).sorted(), kids.sorted(),
                       "the six folders were dropped on the floor instead of kept as names")
    }

    /// Order must not decide the answer. Adding the parent FIRST and the children after has to
    /// produce the same crawl set as the reporter's order.
    func testTheOrderFoldersWereAddedDoesNotMatter() {
        let kids = ["/w/p/a", "/w/p/b", "/w/p/c"]
        let parentFirst = RootScope.canonical(urls(["/w/p"] + kids))
        let parentLast = RootScope.canonical(urls(kids + ["/w/p"]))
        XCTAssertEqual(paths(parentFirst), ["/w/p"])
        XCTAssertEqual(paths(parentFirst), paths(parentLast))
    }

    /// SIBLINGS ARE NOT CHILDREN. A bare `hasPrefix` says "/w/Docs2" is inside "/w/Docs"; it is
    /// not, and that exact defect was found in the dense-hit filter while fixing this issue the
    /// first time. Both must stay roots.
    func testANamePrefixIsNotAParent() {
        let all = urls(["/w/Docs", "/w/Docs2", "/w/DocsOther"])
        XCTAssertEqual(paths(RootScope.canonical(all)).sorted(), ["/w/Docs", "/w/Docs2", "/w/DocsOther"])
        XCTAssertTrue(RootScope.covered(all).isEmpty, "a sibling was swallowed as if it were a child")
    }

    /// Unrelated trees stay separate - the common case, and the one that must not regress.
    func testUnrelatedFoldersAllStayRoots() {
        let all = urls(["/w/one", "/x/two", "/y/three"])
        XCTAssertEqual(paths(RootScope.canonical(all)).count, 3)
        XCTAssertTrue(RootScope.covered(all).isEmpty)
    }

    /// Several levels: a grandchild is covered by the grandparent, and only the grandparent crawls.
    func testAGrandchildIsCoveredByTheGrandparent() {
        let all = urls(["/w/a/b/c", "/w/a/b", "/w/a"])
        XCTAssertEqual(paths(RootScope.canonical(all)), ["/w/a"])
        XCTAssertEqual(paths(RootScope.covered(all)).sorted(), ["/w/a/b", "/w/a/b/c"])
    }

    func testRootCoveringNamesTheFolderThatIndexesIt() {
        let roots = urls(["/w/parent"])
        XCTAssertEqual(RootScope.rootCovering(URL(fileURLWithPath: "/w/parent/alpha"), in: roots)?.path,
                       "/w/parent")
        XCTAssertNil(RootScope.rootCovering(URL(fileURLWithPath: "/elsewhere/alpha"), in: roots))
        XCTAssertNil(RootScope.rootCovering(URL(fileURLWithPath: "/w/parent2/alpha"), in: roots),
                     "a sibling of the root was reported as covered by it")
    }

    /// The root itself is covered by itself, which is what keeps a duplicate add from producing
    /// two rows for one folder.
    func testAFolderIsCoveredByItself() {
        XCTAssertTrue(RootScope.covers("/w/a", "/w/a"))
        XCTAssertEqual(paths(RootScope.canonical(urls(["/w/a", "/w/a"]))), ["/w/a"])
    }

    /// A trailing slash is the same folder. Defaults and file panels disagree about it.
    func testATrailingSlashIsTheSameFolder() {
        XCTAssertTrue(RootScope.covers("/w/a/", "/w/a/b"))
        XCTAssertTrue(RootScope.covers("/w/a", "/w/a/b"))
    }
}

/// The invariant the whole nesting rule exists for: a file is crawled ONCE however many of the
/// folders covering it the user added.
///
/// This is the belt to canonicalization's braces. Until it existed, `AppModel.canonicalizeRoots`
/// was the only thing standing between "add a folder and its parent" and walking the same tree
/// twice - BulkDirWalker pushes every root onto one shared stack and has no notion of one
/// containing another, so a nested pair is walked from both ends: every file hashed twice, and
/// every file the index had not seen embedded twice.
final class CrawlOverlapTests: XCTestCase {

    private func tree() throws -> URL {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("overlap-\(UUID().uuidString)")
        for sub in ["alpha", "beta", "alpha/deep"] {
            try FileManager.default.createDirectory(at: root.appendingPathComponent(sub),
                                                    withIntermediateDirectories: true)
        }
        for rel in ["top.txt", "alpha/a.txt", "alpha/deep/d.txt", "beta/b.txt"] {
            try String(repeating: "content for \(rel)\n", count: 40)
                .write(to: root.appendingPathComponent(rel), atomically: true, encoding: .utf8)
        }
        return root
    }

    private func crawl(_ roots: [URL]) -> [String] {
        var seen: [String] = []
        let lock = NSLock()
        FileCrawler(roots: roots).walk(shouldContinue: { true }) { file in
            lock.lock(); seen.append(file.path); lock.unlock()
        }
        return seen
    }

    func testAFileUnderBothAParentAndAChildRootIsCrawledOnce() throws {
        let root = try tree()
        defer { try? FileManager.default.removeItem(at: root) }

        let parentOnly = crawl([root]).sorted()
        XCTAssertEqual(parentOnly.count, 4, "the corpus itself is wrong")

        // The reporter's shape: children added first, then the parent.
        let withOverlap = crawl([root.appendingPathComponent("alpha"),
                                 root.appendingPathComponent("alpha/deep"),
                                 root.appendingPathComponent("beta"),
                                 root]).sorted()
        XCTAssertEqual(Set(withOverlap).count, withOverlap.count,
                       "a file was crawled twice: \(withOverlap)")
        XCTAssertEqual(withOverlap, parentOnly,
                       "adding nested folders changed what gets crawled")
    }

    /// And the other order, because the walker sees whatever order the caller kept.
    func testTheParentFirstIsTheSameCrawl() throws {
        let root = try tree()
        defer { try? FileManager.default.removeItem(at: root) }
        let a = crawl([root, root.appendingPathComponent("alpha")]).sorted()
        let b = crawl([root.appendingPathComponent("alpha"), root]).sorted()
        XCTAssertEqual(a, b)
        XCTAssertEqual(Set(a).count, a.count, "a file was crawled twice")
    }

    /// Siblings are not nested and must BOTH be crawled - the failure mode of over-eager dropping.
    func testTwoSiblingRootsAreBothCrawled() throws {
        let root = try tree()
        defer { try? FileManager.default.removeItem(at: root) }
        let both = crawl([root.appendingPathComponent("alpha"),
                          root.appendingPathComponent("beta")]).sorted()
        XCTAssertEqual(both.count, 3, "expected alpha/a.txt, alpha/deep/d.txt, beta/b.txt - got \(both)")
        XCTAssertTrue(both.contains { $0.hasSuffix("/beta/b.txt") }, "the sibling root was dropped")
    }
}

/// The sidebar's folder tree. A flat list cannot say what the user's folders mean to each other -
/// after a parent absorbs six children it held seven rows with no sign that six live inside the
/// seventh - so they are nested by containment, the way Finder's sidebar nests.
final class FolderTreeTests: XCTestCase {

    private func urls(_ p: [String]) -> [URL] { p.map { URL(fileURLWithPath: $0) } }
    private func shape(_ nodes: [RootScope.Node]) -> [String] {
        nodes.flatMap { n -> [String] in
            [n.url.lastPathComponent] + (n.children ?? []).flatMap { c in
                ["  " + c.url.lastPathComponent] + (c.children ?? []).map { "    " + $0.url.lastPathComponent }
            }
        }
    }

    func testChildrenNestUnderTheParentTheyWereAddedWith() {
        let t = RootScope.tree(urls(["/w/parent/alpha", "/w/parent/beta", "/w/parent"]))
        XCTAssertEqual(t.count, 1, "the parent should be the only top-level row")
        XCTAssertEqual(shape(t), ["parent", "  alpha", "  beta"])
    }

    /// NEAREST ancestor, not the outermost: a/b/c hangs off a/b, not off a.
    func testEachFolderHangsOffItsNearestAddedAncestor() {
        let t = RootScope.tree(urls(["/w/a", "/w/a/b", "/w/a/b/c"]))
        XCTAssertEqual(shape(t), ["a", "  b", "    c"])
    }

    /// A gap in the chain is fine - with a/b not added, a/b/c hangs off a.
    func testAMissingLevelIsSkipped() {
        let t = RootScope.tree(urls(["/w/a", "/w/a/b/c"]))
        XCTAssertEqual(shape(t), ["a", "  c"])
    }

    /// Unrelated folders are all top level, in the order they were added.
    func testSiblingsStayTopLevelInAddOrder() {
        let t = RootScope.tree(urls(["/w/two", "/w/one", "/x/three"]))
        XCTAssertEqual(t.map { $0.url.lastPathComponent }, ["two", "one", "three"])
        XCTAssertTrue(t.allSatisfy { $0.children == nil })
    }

    /// A NAME PREFIX IS NOT A PARENT. "/w/Docs2" must not nest under "/w/Docs" - the same trap the
    /// crawl set has, and it would be visible here as a folder filed inside one it has nothing to
    /// do with.
    func testANamePrefixDoesNotNest() {
        let t = RootScope.tree(urls(["/w/Docs", "/w/Docs2"]))
        XCTAssertEqual(t.count, 2)
        XCTAssertTrue(t.allSatisfy { $0.children == nil })
    }

    /// nil, not [], for a leaf: SwiftUI draws a disclosure triangle for an empty array, and a
    /// folder with nothing added under it must not get one.
    func testALeafHasNilChildrenSoItDrawsNoTriangle() {
        let t = RootScope.tree(urls(["/w/only"]))
        XCTAssertNil(t.first?.children)
    }

    /// Adding the same folder twice must not produce two rows.
    func testARepeatedFolderAppearsOnce() {
        let t = RootScope.tree(urls(["/w/a", "/w/a", "/w/a/b"]))
        XCTAssertEqual(t.count, 1)
        XCTAssertEqual(shape(t), ["a", "  b"])
    }

    /// Order of addition must not change the shape, only the order within a level.
    func testTheShapeDoesNotDependOnAddOrder() {
        let a = RootScope.tree(urls(["/w/p", "/w/p/x", "/w/p/y"]))
        let b = RootScope.tree(urls(["/w/p/y", "/w/p/x", "/w/p"]))
        XCTAssertEqual(a.count, b.count)
        XCTAssertEqual(Set(shape(a)), Set(shape(b)))
    }

    /// Every added folder appears exactly once somewhere in the tree - nothing is lost to nesting.
    func testEveryAddedFolderAppearsExactlyOnce() {
        let added = urls(["/w/p", "/w/p/x", "/w/p/x/deep", "/w/q", "/w/p/y"])
        func flatten(_ ns: [RootScope.Node]) -> [String] {
            ns.flatMap { [$0.url.path] + flatten($0.children ?? []) }
        }
        let all = flatten(RootScope.tree(added))
        XCTAssertEqual(all.sorted(), added.map(\.path).sorted())
        XCTAssertEqual(Set(all).count, all.count)
    }
}
