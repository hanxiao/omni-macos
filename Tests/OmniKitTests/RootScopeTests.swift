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
