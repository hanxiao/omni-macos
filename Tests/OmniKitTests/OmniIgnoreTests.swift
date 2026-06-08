import XCTest
@testable import OmniKit

final class OmniIgnoreTests: XCTestCase {
    private func ig(_ text: String) -> OmniIgnore { OmniIgnore(text: text) }

    func testEmptyIgnoresNothing() {
        let g = ig("")
        XCTAssertTrue(g.isEmpty)
        XCTAssertFalse(g.isIgnored("/a/b/x.png", isDir: false))
    }

    func testBasenameGlob() {
        let g = ig("*.png")
        XCTAssertTrue(g.isIgnored("/a/b/x.png", isDir: false))     // any depth
        XCTAssertTrue(g.isIgnored("/top.png", isDir: false))
        XCTAssertFalse(g.isIgnored("/a/b/x.jpg", isDir: false))
    }

    func testCommentsAndBlanksIgnored() {
        let g = ig("# a comment\n\n   \n*.gif\n")
        XCTAssertTrue(g.isIgnored("/x.gif", isDir: false))
        XCTAssertFalse(g.isIgnored("/x.png", isDir: false))
    }

    func testDirOnlyMatchesDirectoriesNotFiles() {
        let g = ig("build/")
        XCTAssertTrue(g.isIgnored("/proj/build", isDir: true))
        XCTAssertFalse(g.isIgnored("/proj/build", isDir: false))   // a FILE named build is not matched
        // a directory named build at any depth is pruned; a file under it isn't tested here (pruned in crawl)
        XCTAssertTrue(g.isIgnored("/a/b/build", isDir: true))
    }

    func testAnchoredDoesNotCrossSlashWithStar() {
        let g = ig("/Users/me/Downloads/*.zip")
        XCTAssertTrue(g.isIgnored("/Users/me/Downloads/a.zip", isDir: false))
        XCTAssertFalse(g.isIgnored("/Users/me/Downloads/sub/a.zip", isDir: false))   // * doesn't cross '/'
        XCTAssertFalse(g.isIgnored("/Users/other/Downloads/a.zip", isDir: false))
    }

    func testAnchoredPrefixIgnoresSubtree() {
        let g = ig("/Users/me/secret")
        XCTAssertTrue(g.isIgnored("/Users/me/secret", isDir: true))
        XCTAssertTrue(g.isIgnored("/Users/me/secret/deep/file.txt", isDir: false))   // everything under it
        XCTAssertFalse(g.isIgnored("/Users/me/public/file.txt", isDir: false))
    }

    func testDoubleStarAnyDepth() {
        let g = ig("**/node_modules/")
        XCTAssertTrue(g.isIgnored("/a/node_modules", isDir: true))
        XCTAssertTrue(g.isIgnored("/a/b/c/node_modules", isDir: true))
        XCTAssertFalse(g.isIgnored("/a/node_modules", isDir: false))   // dirOnly
    }

    func testDoubleStarMatchesZeroSegments() {
        let g = ig("/a/**/b")
        XCTAssertTrue(g.isIgnored("/a/b", isDir: true))         // ** matches zero dirs
        XCTAssertTrue(g.isIgnored("/a/x/b", isDir: true))
        XCTAssertTrue(g.isIgnored("/a/x/y/b", isDir: true))
        XCTAssertFalse(g.isIgnored("/a/x/c", isDir: true))
    }

    func testTrailingDoubleStarMatchesEverythingUnder() {
        let g = ig("/Users/me/Work/secret/**")
        XCTAssertTrue(g.isIgnored("/Users/me/Work/secret/a.txt", isDir: false))
        XCTAssertTrue(g.isIgnored("/Users/me/Work/secret/deep/a.txt", isDir: false))
        XCTAssertFalse(g.isIgnored("/Users/me/Work/other/a.txt", isDir: false))
    }

    func testNegationLastMatchWins() {
        let g = ig("*.log\n!important.log")
        XCTAssertTrue(g.isIgnored("/a/debug.log", isDir: false))
        XCTAssertFalse(g.isIgnored("/a/important.log", isDir: false))   // re-included
    }

    func testNegationOrderMatters() {
        // exclude then re-include then exclude again -> last wins (excluded)
        let g = ig("*.tmp\n!keep.tmp\nkeep.tmp")
        XCTAssertTrue(g.isIgnored("/x/keep.tmp", isDir: false))
    }

    func testCharacterClass() {
        let g = ig("*.[oa]")
        XCTAssertTrue(g.isIgnored("/x.o", isDir: false))
        XCTAssertTrue(g.isIgnored("/x.a", isDir: false))
        XCTAssertFalse(g.isIgnored("/x.b", isDir: false))
    }

    func testCaseInsensitive() {
        let g = ig("*.PNG")
        XCTAssertTrue(g.isIgnored("/a/photo.png", isDir: false))   // APFS default
    }

    func testKindDisableAsExtensionExcludes() {
        // disabling "images" = excluding its extensions
        let g = ig("# images off\n*.png\n*.jpg\n*.jpeg\n*.gif")
        XCTAssertTrue(g.isIgnored("/photos/a.jpg", isDir: false))
        XCTAssertFalse(g.isIgnored("/docs/a.md", isDir: false))
    }

    func testTextHashStableAndDistinct() {
        XCTAssertEqual(ig("*.png\n").textHash, ig("*.png\n").textHash)
        XCTAssertNotEqual(ig("*.png\n").textHash, ig("*.jpg\n").textHash)
    }

    // MARK: - Migration synthesizer (behavior-preservation)

    /// The synthesized policy must exclude EXACTLY what the pre-OmniIgnore crawl excluded:
    /// `isSupported(enabledKinds, disabledExtensions)` false  <=>  ignored OR not extractable.
    func testSynthesizeMatchesLegacyIsSupported() {
        let cases: [(Set<FileKind>, Set<String>)] = [
            ([.text, .image, .video, .audio], []),          // everything on
            ([.text], []),                                   // only text
            ([.text, .image, .video, .audio], ["gif", "log"]), // loose exclusions
            ([.text, .audio], ["mp3"]),                       // mixed
        ]
        let samples = ["a.png", "b.jpg", "c.gif", "d.mp4", "e.mp3", "f.md", "g.swift",
                       "h.pdf", "i.docx", "j.unknownext", "k.log", "l.wav"]
        for (kinds, disabled) in cases {
            let g = OmniIgnore(text: OmniIgnore.synthesize(enabledKinds: kinds, disabledExtensions: disabled))
            for name in samples {
                let url = URL(fileURLWithPath: "/root/sub/\(name)")
                let oldIndexed = FileExtractor.isSupported(url, enabledKinds: kinds, disabledExtensions: disabled)
                let newIndexed = FileExtractor.kind(for: url) != nil && !g.isIgnored(url.path, isDir: false)
                XCTAssertEqual(oldIndexed, newIndexed, "\(name) kinds=\(kinds) disabled=\(disabled)")
            }
        }
    }

    /// Seeded noise directories are pruned at any depth (the crawl skips their whole subtree).
    func testSynthesizeSeedsNoiseDirs() {
        let g = OmniIgnore(text: OmniIgnore.synthesize(enabledKinds: [.text, .image, .video, .audio], disabledExtensions: []))
        XCTAssertTrue(g.isIgnored("/proj/node_modules", isDir: true))
        XCTAssertTrue(g.isIgnored("/a/b/__pycache__", isDir: true))
        XCTAssertFalse(g.isIgnored("/proj/src", isDir: true))
    }
}
