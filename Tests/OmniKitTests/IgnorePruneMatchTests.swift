import XCTest
@testable import OmniKit

/// Pruning already-indexed files after an ignore edit tests FILE paths. A directory rule must still
/// reach the files beneath it, or adding `site-packages/` prunes nothing that is indexed.
final class IgnorePruneMatchTests: XCTestCase {
    func testDirectoryRulesReachIndexedFiles() {
        let ig = OmniIgnore(text: "site-packages/\n/Users/u/Docs/icons/\ndrawable-*dpi*/\n*.min.js\n")
        let excluded = ig.excludesIndexedFile()
        for p in ["/Users/u/venv/lib/python3.13/site-packages/numpy/core.py",
                  "/Users/u/Docs/icons/a/b.png",
                  "/Users/u/app/res/drawable-xhdpi/ic.png",
                  "/Users/u/app/res/drawable-hdpi-v11/ic.png",
                  "/Users/u/web/app.min.js"] {
            XCTAssertTrue(excluded(p), p)
            XCTAssertEqual(excluded(p), ig.isIgnoredIncludingAncestors(p, isDir: false), p)
        }
        for p in ["/Users/u/Docs/notes.md", "/Users/u/Docs/icons.md", "/Users/u/app/res/drawable/ic.xml", "/Users/u/web/app.js"] {
            XCTAssertFalse(excluded(p), p)
        }
    }

    /// The one-time merge adds only what is missing, inside the noise block, and respects a rule
    /// the user negated.
    func testAddedDefaultsMergeOnce() {
        let old = "# header\n\n# Noise directories (x).\nnode_modules/\nbuild/\n/my/private/\n\n# Disabled file types.\n*.gif\n"
        let merged = OmniIgnore.withAddedDefaults(old)
        let lines = merged.components(separatedBy: "\n")
        for d in OmniIgnore.addedDefaults { XCTAssertEqual(lines.filter { $0 == d }.count, 1, d) }
        XCTAssertLessThan(lines.firstIndex(of: "site-packages/")!, lines.firstIndex(of: "# Disabled file types.")!)
        XCTAssertGreaterThan(lines.firstIndex(of: "site-packages/")!, lines.firstIndex(of: "/my/private/")!)
        XCTAssertEqual(OmniIgnore.withAddedDefaults(merged), merged, "idempotent")
        let negated = OmniIgnore.withAddedDefaults("node_modules/\n!site-packages/\n")
        XCTAssertFalse(negated.components(separatedBy: "\n").contains("site-packages/"))
        let fresh = OmniIgnore.synthesize(enabledKinds: [.text, .image, .audio, .video], disabledExtensions: [])
        XCTAssertEqual(OmniIgnore.withAddedDefaults(fresh), fresh, "a new install already has them")
    }
}
