import XCTest
@testable import OmniKit

/// Issue #24: a folder whose name starts with a dot could not be re-included by any `!` rule,
/// because the crawl and the watcher skipped hidden names before reading the policy. Hidden names are
/// now the `.*` line of the policy file itself, so the usual last-rule-wins order applies to them.
final class HiddenFolderPolicyTests: XCTestCase {
    func testTheDefaultFileExcludesHiddenNamesWithAVisibleLine() {
        let text = OmniIgnore.synthesize(enabledKinds: Set(FileKind.indexable), disabledExtensions: [])
        let firstRule = text.split(separator: "\n").map(String.init).first { OmniIgnore.parseLine($0) != nil }
        XCTAssertEqual(firstRule, ".*", "the hidden rule is the file's first rule, so every line after it wins")
        let p = OmniIgnore(text: text)
        XCTAssertTrue(p.isIgnored("/u/notes/.obsidian", isDir: true))
        XCTAssertFalse(p.isIgnored("/u/notes/v1.2", isDir: true), "a dot inside a name is not hidden")
        XCTAssertFalse(OmniIgnore(text: "").isIgnored("/u/notes/.obsidian", isDir: true),
                       "no rule outside the file: an empty policy excludes nothing")
    }

    func testANegationReIncludesAHiddenFolderAndItsOrdinaryContents() {
        let p = OmniIgnore(text: ".*\n!.obsidian/\n")
        XCTAssertFalse(p.isIgnored("/u/notes/.obsidian", isDir: true))
        XCTAssertFalse(p.isIgnoredIncludingAncestors("/u/notes/.obsidian/workspace.json", isDir: false, root: "/u/notes"))
        XCTAssertTrue(p.isIgnored("/u/notes/.obsidian/.trash", isDir: true), "a hidden name inside needs its own rule")
        XCTAssertTrue(p.isIgnored("/u/notes/.git", isDir: true))
    }

    func testTheMigrationPutsTheHiddenRuleBeforeTheUsersLines() {
        let old = "# my policy\n\nnode_modules/\n!.obsidian/\n"
        let new = OmniIgnore.withHiddenRule(old)
        let rules = new.split(separator: "\n").map(String.init).filter { OmniIgnore.parseLine($0) != nil }
        XCTAssertEqual(rules, [".*", "node_modules/", "!.obsidian/"])
        XCTAssertTrue(new.hasPrefix("# my policy\n"), "comments ahead of the first rule stay where they are")
        XCTAssertFalse(OmniIgnore(text: new).isIgnored("/u/n/.obsidian", isDir: true), "a negation already written now works")
        XCTAssertEqual(OmniIgnore.withHiddenRule(new), new, "run once")
        XCTAssertEqual(OmniIgnore.withHiddenRule("!.*\n"), "!.*\n", "a file that opted out stays out")
    }

    func testTheCrawlAndTheWatcherFollowTheFile() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("hidden-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let fm = FileManager.default
        for d in [".obsidian", ".obsidian/.trash", ".git", "visible"] {
            try fm.createDirectory(at: root.appendingPathComponent(d), withIntermediateDirectories: true)
        }
        for f in [".obsidian/workspace.md", ".obsidian/.trash/old.md", ".git/notes.md", "visible/a.md", ".hidden.md"] {
            try "# \(f) with enough words to count as a document".write(to: root.appendingPathComponent(f),
                                                                      atomically: true, encoding: .utf8)
        }
        // Relative to the test folder by its unique name: the crawl reports /private/var for /var.
        let mark = root.lastPathComponent + "/"
        func crawl(_ policy: String) -> Set<String> {
            var found = Set<String>()
            FileCrawler(roots: [root], ignore: OmniIgnore(text: policy))
                .walk(shouldContinue: { true }) { found.insert($0.path.components(separatedBy: mark).last ?? $0.path) }
            return found
        }
        XCTAssertEqual(crawl(".*\n"), ["visible/a.md"])
        XCTAssertEqual(crawl(".*\n!.obsidian/\n"), ["visible/a.md", ".obsidian/workspace.md"])
        XCTAssertEqual(crawl(".*\n!.obsidian/\n!.trash/\n!.hidden.md\n"),
                       ["visible/a.md", ".obsidian/workspace.md", ".obsidian/.trash/old.md", ".hidden.md"])
        XCTAssertEqual(crawl(""), ["visible/a.md", ".obsidian/workspace.md", ".obsidian/.trash/old.md",
                                   ".git/notes.md", ".hidden.md"], "without the line, hidden files are indexed")
        XCTAssertEqual(crawl("node_modules/\n").count, 5, "nothing else in the app excludes them either")

        let crawler = FileCrawler(roots: [root], ignore: OmniIgnore(text: ".*\n!.obsidian/\n"))
        let p = root.path + "/.obsidian/workspace.md"
        XCTAssertTrue(crawler.admitsEventPath(p, isDir: false, size: 10, root: root.path))
        XCTAssertFalse(OmniIgnore(text: ".*\n!.obsidian/\n").isIgnoredIncludingAncestors(p, isDir: false, root: root.path))
        XCTAssertTrue(OmniIgnore(text: ".*\n").isIgnoredIncludingAncestors(p, isDir: false, root: root.path))
    }
}

/// The policy grammar is gitignore(5), in full. Each case is the behaviour git documents.
final class GitignoreGrammarTests: XCTestCase {
    private func ig(_ text: String, _ path: String, dir: Bool = false, bases: [String] = []) -> Bool {
        OmniIgnore(text: text, bases: bases).isIgnored(path, isDir: dir)
    }

    func testCommentsBlanksAndEscapedLeaders() {
        XCTAssertFalse(ig("# a.md\n\n", "/r/a.md"))
        XCTAssertTrue(ig("\\#a.md", "/r/#a.md"), "\\# starts a pattern with #")
        XCTAssertTrue(ig("\\!a.md", "/r/!a.md"), "\\! starts a pattern with !")
        XCTAssertFalse(ig("\\!a.md", "/r/a.md"))
    }

    func testTrailingSpacesDropUnlessEscapedAndLeadingSpacesCount() {
        XCTAssertTrue(ig("a.md   ", "/r/a.md"))
        XCTAssertTrue(ig("a\\ ", "/r/a "), "an escaped trailing space is part of the name")
        XCTAssertFalse(ig("a\\ ", "/r/a"))
        XCTAssertFalse(ig(" a.md", "/r/a.md"), "a leading space is part of the pattern")
        XCTAssertTrue(ig(" a.md", "/r/ a.md"))
    }

    func testBackslashMakesAGlobCharacterLiteral() {
        XCTAssertTrue(ig("a\\*.md", "/r/a*.md"))
        XCTAssertFalse(ig("a\\*.md", "/r/abc.md"))
        XCTAssertTrue(ig("what\\?.md", "/r/what?.md"))
        XCTAssertFalse(ig("what\\?.md", "/r/whatx.md"))
        XCTAssertTrue(ig("\\[draft\\].md", "/r/[draft].md"))
    }

    func testCharacterClasses() {
        XCTAssertTrue(ig("v[0-9].md", "/r/v7.md"))
        XCTAssertFalse(ig("v[0-9].md", "/r/vx.md"))
        XCTAssertTrue(ig("v[!0-9].md", "/r/vx.md"))
        XCTAssertTrue(ig("v[^0-9].md", "/r/vx.md"))
        XCTAssertTrue(ig("v[[:digit:]].md", "/r/v3.md"))
        XCTAssertFalse(ig("v[[:digit:]].md", "/r/va.md"))
        XCTAssertTrue(ig("[]x].md", "/r/].md"), "a ] first in the set is literal")
        XCTAssertTrue(ig("a[b.md", "/r/a[b.md"), "an unterminated [ is literal")
    }

    func testDoubleStar() {
        XCTAssertTrue(ig("**/logs", "/r/a/b/logs", dir: true))
        XCTAssertTrue(ig("a/**/b", "/r/a/b", bases: ["/r"]), "zero folders between")
        XCTAssertTrue(ig("a/**/b", "/r/a/x/y/b", bases: ["/r"]))
        XCTAssertTrue(ig("a/**", "/r/a/x.md", bases: ["/r"]))
        XCTAssertFalse(ig("a/**", "/r/a", dir: true, bases: ["/r"]), "a/** is what is inside a, not a")
        // ...which is what makes git's classic exception work.
        let p = OmniIgnore(text: "a/**\n!a/keep.md\n", bases: ["/r"])
        XCTAssertFalse(p.isIgnoredIncludingAncestors("/r/a/keep.md", isDir: false, root: "/r"))
        XCTAssertTrue(p.isIgnoredIncludingAncestors("/r/a/other.md", isDir: false, root: "/r"))
        XCTAssertTrue(ig("a**b.md", "/r/axyb.md"), "** inside a name is an ordinary *")
    }

    func testASlashMakesAPatternRelativeToTheIndexedFolder() {
        let bases = ["/Users/me/Notes", "/Users/me/Code"]
        XCTAssertTrue(ig("docs/build/", "/Users/me/Code/docs/build", dir: true, bases: bases))
        XCTAssertFalse(ig("docs/build/", "/Users/me/Code/x/docs/build", dir: true, bases: bases),
                       "relative to the folder, not at any depth")
        XCTAssertTrue(ig("/drafts", "/Users/me/Notes/drafts", dir: true, bases: bases), "a leading / is the folder's top")
        XCTAssertFalse(ig("/drafts", "/Users/me/Notes/x/drafts", dir: true, bases: bases))
        XCTAssertTrue(ig("/Users/me/Notes/private/", "/Users/me/Notes/private", dir: true, bases: bases),
                      "an absolute path still works")
        XCTAssertTrue(ig("!.obsidian/workspace.json\n.obsidian/*\n!.obsidian/workspace.json", "/Users/me/Notes/.obsidian/app.json", bases: bases))
        XCTAssertFalse(ig(".obsidian/*\n!.obsidian/workspace.json", "/Users/me/Notes/.obsidian/workspace.json", bases: bases))
    }

    func testAFolderPolicyIsRelativeToItsFolderAndEscapesItsName() {
        let scoped = OmniIgnore.scoped("*.json\n/build/\n!keep.json\n", to: "/u/[draft] notes")
        let p = OmniIgnore(text: "", folderRules: scoped)
        XCTAssertTrue(p.isIgnored("/u/[draft] notes/sub/a.json", isDir: false))
        XCTAssertFalse(p.isIgnored("/u/[draft] notes/sub/keep.json", isDir: false))
        XCTAssertTrue(p.isIgnored("/u/[draft] notes/build", isDir: true))
        XCTAssertFalse(p.isIgnored("/u/[draft] notes/sub/build", isDir: true))
        XCTAssertFalse(p.isIgnored("/u/d notes/sub/a.json", isDir: false), "the folder name is literal, not a glob")
    }

    func testAFolderPolicyCanReIncludeAHiddenFolderBelowIt() {
        let p = OmniIgnore(text: ".*\n", folderRules: OmniIgnore.scoped("!.config/\n", to: "/u/proj"))
        XCTAssertFalse(p.isIgnored("/u/proj/.config", isDir: true))
        XCTAssertFalse(p.isIgnored("/u/proj/sub/.config", isDir: true))
        XCTAssertTrue(p.isIgnored("/u/other/.config", isDir: true))
    }
}

/// The policy is user text and runs on every crawled path: no input may hang or crash it.
final class IgnoreRobustnessTests: XCTestCase {
    private func seconds(_ body: () -> Void) -> Double { let t = Date(); body(); return -t.timeIntervalSinceNow }

    func testManyDoubleStarsStayPolynomial() {
        // Without memoisation this is exponential in the number of `**` (each tries every split).
        let p = OmniIgnore(text: "**/a/**/a/**/a/**/a/**/a/**/b\n")
        let deep = "/" + Array(repeating: "a", count: 60).joined(separator: "/") + "/c"
        var ignored = true
        XCTAssertLessThan(seconds { ignored = p.isIgnored(deep, isDir: false) }, 0.5)
        XCTAssertFalse(ignored)
        XCTAssertTrue(p.isIgnored("/a/x/a/a/y/a/a/b", isDir: false))
    }

    func testBacktrackingGlobIsLinearish() {
        let p = OmniIgnore(text: "*a*a*a*a*a*a*a*a*a*a*b\n")
        let name = "/r/" + String(repeating: "a", count: 500)
        XCTAssertLessThan(seconds { _ = p.isIgnored(name, isDir: false) }, 0.2)
        XCTAssertFalse(p.isIgnored(name, isDir: false))
    }

    func testPathologicalLinesParseQuicklyAndMatchNothingStrange() {
        let brackets = String(repeating: "[", count: 20_000)
        let posix = String(repeating: "[:", count: 2_000) + "]"
        let huge = String(repeating: "x", count: 100_000)
        var p = OmniIgnore(text: "")
        XCTAssertLessThan(seconds { p = OmniIgnore(text: [brackets, posix, huge, "\\", "!", "/", "**", "[]", "[!]", "\\\\\\"].joined(separator: "\n")) }, 1.0)
        _ = p.isIgnored("/r/a", isDir: false)
        XCTAssertFalse(OmniIgnore(text: huge).isIgnored("/r/" + huge, isDir: false), "an over-long line is no rule")
    }

    func testRandomPatternsNeverCrashOrHang() {
        var rng = SystemRandomNumberGenerator()
        let alphabet = Array("ab*?[]!^-:\\/ .#")
        let names = ["a", "b", ".a", "a b", "*", "[", "]", "!x", "aa-b", "Ä", "café", "x:y"]
        let t = Date()
        for _ in 0 ..< 3_000 {
            let len = Int.random(in: 0 ... 14, using: &rng)
            let pat = String((0 ..< len).map { _ in alphabet.randomElement(using: &rng)! })
            let p = OmniIgnore(text: pat + "\n!" + pat + "x\n", bases: ["/r"])
            let depth = Int.random(in: 1 ... 6, using: &rng)
            let path = "/r/" + (0 ..< depth).map { _ in names.randomElement(using: &rng)! }.joined(separator: "/")
            _ = p.isIgnored(path, isDir: Bool.random(using: &rng))
            _ = p.isIgnoredIncludingAncestors(path, isDir: false, root: "/r")
            _ = p.excludesIndexedFile(roots: ["/r"])(path)
        }
        XCTAssertLessThan(-t.timeIntervalSinceNow, 5.0)
    }

    func testDecomposedNamesMatchComposedPatterns() {
        let decomposed = "/r/Cafe\u{301}/menu.md"     // how a file name can arrive from disk
        XCTAssertTrue(OmniIgnore(text: "café/\n").isIgnoredIncludingAncestors(decomposed, isDir: false, root: "/r"))
        XCTAssertTrue(OmniIgnore(text: "CAFÉ/\n").isIgnoredIncludingAncestors(decomposed, isDir: false, root: "/r"))
    }
}

/// Migration 3 on the real policy file from the index the new defaults were measured on.
final class DefaultsV3Tests: XCTestCase {
    func testMigrationAddsOnlyWhatIsMissingAndKeepsTheUsersLinesLast() {
        let old = "# Omni ignore\n\n# Noise directories (build output, caches, dependencies). Delete a line to start indexing it.\nnode_modules/\n/Users/me/private/\nsite-packages/\n\n# mine\n!wandb/\n"
        var t = OmniIgnore.withAddedDefaults(old, OmniIgnore.addedDefaultsV3)
        t = OmniIgnore.withHiddenRule(t)
        let rules = t.split(separator: "\n").map(String.init).filter { OmniIgnore.parseLine($0) != nil }
        XCTAssertEqual(rules.first, ".*")
        XCTAssertFalse(rules.contains("wandb/"), "a default the user negated is not added")
        XCTAssertEqual(rules.last, "!wandb/")
        for d in OmniIgnore.addedDefaultsV3 where d != "wandb/" { XCTAssertTrue(rules.contains(d), d) }
        XCTAssertEqual(OmniIgnore.withHiddenRule(OmniIgnore.withAddedDefaults(t, OmniIgnore.addedDefaultsV3)), t, "run once")
    }

    func testTheHiddenBlockKeepsSectionsTogether() {
        let old = "# Omni ignore - header.\n\n# Noise directories.\nnode_modules/\n\n# mine\n!.obsidian/\n"
        XCTAssertEqual(OmniIgnore.withHiddenRule(old), """
            # Omni ignore - header.

            # Names starting with a dot. Re-include one below with !.name/
            .*

            # Noise directories.
            node_modules/

            # mine
            !.obsidian/

            """)
    }

    func testTheNewDefaultsMatchWhatTheyWereMeasuredOn() {
        let p = OmniIgnore(text: OmniIgnore.synthesize(enabledKinds: Set(FileKind.indexable), disabledExtensions: []))
        XCTAssertTrue(p.isIgnoredIncludingAncestors("/r/App/Assets.xcassets/AppIcon.appiconset/128.png", isDir: false, root: "/r"))
        XCTAssertTrue(p.isIgnoredIncludingAncestors("/r/docs/_build/html/index.html", isDir: false, root: "/r"))
        XCTAssertTrue(p.isIgnoredIncludingAncestors("/r/app/user-data/Default/Extensions/abc/1.0/popup.js", isDir: false, root: "/r"))
        XCTAssertTrue(p.isIgnoredIncludingAncestors("/r/app/user-data/Profile 2/Extensions/abc/1.0/popup.js", isDir: false, root: "/r"))
        XCTAssertTrue(p.isIgnoredIncludingAncestors("/r/app/Default/IndexedDB/x.leveldb/000004.log", isDir: false, root: "/r"))
        XCTAssertTrue(p.isIgnoredIncludingAncestors("/r/app/Default/Web Applications/Manifest Resources/x/Icons/128.png", isDir: false, root: "/r"))
        // ...and not their look-alikes.
        XCTAssertFalse(p.isIgnoredIncludingAncestors("/r/Sources/Extensions/String+Trim.swift", isDir: false, root: "/r"))
        XCTAssertFalse(p.isIgnoredIncludingAncestors("/r/logs/server.log", isDir: false, root: "/r"))
        XCTAssertFalse(p.isIgnoredIncludingAncestors("/r/2024/123456-notes.log", isDir: false, root: "/r"))
    }
}
