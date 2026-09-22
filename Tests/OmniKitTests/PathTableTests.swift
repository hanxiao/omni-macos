import XCTest
@testable import OmniKit

/// The path table has to be a drop-in for `[String]` + `[String: Int32]` - including the one
/// property that is easy to lose by going to bytes: lookup by CANONICAL equality.
final class PathTableTests: XCTestCase {

    /// Round trip and lookup against a reference built from the old pair, over paths shaped like a
    /// real tree: shared directories, a root file, a path with no slash, an empty path, non-ASCII.
    func testMatchesTheArrayAndDictionaryItReplaces() {
        var paths = ["", "noslash", "/root.txt", "/a/b/c.txt", "/a/b/d.txt", "/a/x/c.txt",
                     "/Users/u/文档/报告.pdf", "photos://asset/ABC-123", "/a/b/", "/a/b/c.txt.bak"]
        for i in 0 ..< 3000 { paths.append("/deep/dir\(i % 37)/sub\(i % 5)/file-\(i).md") }
        var t = PathTable()
        var refID: [String: Int32] = [:]
        for (i, p) in paths.enumerated() {
            XCTAssertEqual(t.append(p), Int32(i))
            refID[p] = Int32(i)
        }
        XCTAssertEqual(t.count, paths.count)
        XCTAssertEqual(t.keyCount, refID.count)
        for (i, p) in paths.enumerated() {
            XCTAssertEqual(t[i], p, "round trip of id \(i)")
            XCTAssertEqual(t.id(p), refID[p], "lookup of \(p)")
            XCTAssertEqual(t.byteCount(i), p.utf8.count)
        }
        XCTAssertNil(t.id("/a/b/e.txt"))
        XCTAssertNil(t.id("/a/b"))
        // 3000 files over 37 * 5 subdirectories plus the handful above.
        XCTAssertLessThan(t.dirCount, 200)
    }

    /// THE PROPERTY THE STORE DEPENDS ON. An NFC spelling finds the entry an NFD spelling wrote,
    /// and reads back the STORED bytes - that is `storedSpellingLocked`, and without it a watcher
    /// event on an old NFD path re-indexes the file as new.
    func testLookupIsCanonicalAndReturnsTheStoredSpelling() {
        let nfd = "/docs/Eigentu\u{0308}mer.txt", nfc = "/docs/Eigent\u{00FC}mer.txt"
        XCTAssertEqual(nfd, nfc)
        XCTAssertNotEqual(Array(nfd.utf8), Array(nfc.utf8))
        var t = PathTable()
        t.append("/docs/other.txt")
        let id = t.append(nfd)
        XCTAssertEqual(t.id(nfc), id, "the NFC spelling must find the NFD entry")
        XCTAssertEqual(Array(t.storedSpelling(nfc)!.utf8), Array(nfd.utf8), "and hand back the stored bytes")
        // interning the other spelling does not add a file
        XCTAssertEqual(t.intern(nfc).isNew, false)
        XCTAssertEqual(t.count, 2)
        // A singleton decomposition to ASCII: KELVIN SIGN is canonically "K".
        t.append("/k/K.txt")
        XCTAssertNotNil(t.id("/k/\u{212A}.txt"))
    }

    /// A duplicate append re-points the key at the newest id and keeps both ids readable -
    /// `idPath.append(p); pathID[p] = id` - and survives the index growing afterwards.
    func testDuplicateAppendRepointsTheKey() {
        var t = PathTable()
        t.append("/a.txt"); t.append("/b.txt"); t.append("/a.txt")
        XCTAssertEqual(t.count, 3); XCTAssertEqual(t.keyCount, 2)
        XCTAssertEqual(t.id("/a.txt"), 2)
        XCTAssertEqual(t[0], "/a.txt"); XCTAssertEqual(t[2], "/a.txt")
        for i in 0 ..< 500 { t.append("/grow/\(i)") }
        XCTAssertEqual(t.id("/a.txt"), 2, "growth must keep the re-pointed target")
    }

    /// Directory ids are what the folder filter keys on, so they have to be exactly the path up to
    /// and including its last "/".
    func testDirectoryIsTheBytesUpToTheLastSlash() {
        var t = PathTable()
        let a = Int(t.append("/a/b/c.txt")), b = Int(t.append("/a/b/d.txt")), c = Int(t.append("/a/bc.txt"))
        let r = Int(t.append("noslash"))
        XCTAssertEqual(t.dirID(a), t.dirID(b))
        XCTAssertNotEqual(t.dirID(a), t.dirID(c))
        XCTAssertEqual(t.withDirBytes(Int(t.dirID(a))) { String(decoding: $0, as: UTF8.self) }, "/a/b/")
        XCTAssertEqual(t.withDirBytes(Int(t.dirID(r))) { $0.count }, 0)
        XCTAssertEqual(t.withNameBytes(c) { String(decoding: $0, as: UTF8.self) }, "bc.txt")
    }

    func testRemoveAllKeepingCapacityStartsOver() {
        var t = PathTable()
        for i in 0 ..< 100 { t.append("/x/\(i)") }
        t.removeAll(keepingCapacity: true)
        XCTAssertEqual(t.count, 0); XCTAssertNil(t.id("/x/5"))
        XCTAssertEqual(t.append("/y/1"), 0); XCTAssertEqual(t.id("/y/1"), 0); XCTAssertEqual(t[0], "/y/1")
    }

    /// `filesUnder` claims to be EXACTLY the per-file expressions it replaced, under both
    /// semantics. Checked against those expressions on paths chosen to separate them: NFC and NFD
    /// spellings of one directory, a name that starts with a combining mark directly under the
    /// folder (where String.hasPrefix and bytes disagree), the Kelvin sign, CJK, a sibling whose
    /// name extends the folder's, a folder that is itself a file, root files and the empty path.
    func testFilesUnderMatchesTheExpressionsItReplaces() {
        let nfc = "Caf\u{00E9}", nfd = "Cafe\u{0301}"
        let corpus = [
            "", "noslash", "/root.txt", "/a", "/a/b", "/a/bc", "/a/b/c.txt", "/a/bc/d.txt",
            "/Users/u/\(nfc)/x.txt", "/Users/u/\(nfd)/y.txt", "/Users/u/\(nfc)",
            "/Users/u/\(nfc)/\u{0301}lead.txt", "/Users/u/\(nfd)/\u{0301}lead2.txt",
            "/k/\u{212A}elvin/z.txt", "/k/Kelvin/z.txt",
            "/Users/u/\u{6587}\u{6863}/\u{62A5}\u{544A}.pdf", "/Users/u/\u{6587}\u{6863}/sub/n.md",
            "photos://all/ABC%2FL0/IMG_1.HEIC", "/deep/1/2/3/4/5/6/7.txt",
        ]
        let folders = ["/a", "/a/b", "/Users/u/\(nfc)", "/Users/u/\(nfd)", "/Users/u", "/k/Kelvin",
                       "/k/\u{212A}elvin", "/Users/u/\u{6587}\u{6863}", "photos://all", "/deep/1/2",
                       "/nothing", "/a/b/c.txt", "noslash", "/"]
        var t = PathTable()
        for p in corpus { t.append(p) }
        for folder in folders {
            let slash = Array((folder + "/").utf8)
            let byBytes = t.filesUnder(folder, semantics: .bytes)
            let byString = t.filesUnder(folder, semantics: .string)
            for (i, p) in corpus.enumerated() {
                let wantBytes = p == folder || SearchFilter.underFolderBytes(p, slash)
                let wantString = p == folder || p.hasPrefix(folder + "/")
                XCTAssertEqual(byBytes[i], wantBytes, "bytes: \(folder.debugDescription) vs \(p.debugDescription)")
                XCTAssertEqual(byString[i], wantString, "string: \(folder.debugDescription) vs \(p.debugDescription)")
            }
        }
        // The fixture must actually contain a case where the two semantics disagree.
        let disagree = folders.contains { f in
            corpus.contains { p in (p.hasPrefix(f + "/")) != SearchFilter.underFolderBytes(p, Array((f + "/").utf8)) }
        }
        XCTAssertTrue(disagree, "no path separates the two semantics, so this proves one of them twice")
    }

    /// The extension check on name bytes against `SearchFilter.hasExtensionCI` on the full path.
    func testExtensionOnNameBytesMatchesTheFullPath() {
        let corpus = ["/a/b.TXT", "/a/b.txt", "/a.txt/b", "/a/.txt", "/a/txt", "/x/y.tar.gz", "/x/y.", "", "/a/b.Txt"]
        var t = PathTable()
        for p in corpus { t.append(p) }
        for ext in ["txt", "TXT", "gz", "tar.gz", "t", ""] where !ext.isEmpty {
            for (i, p) in corpus.enumerated() {
                XCTAssertEqual(t.hasExtensionCI(i, Array(ext.utf8)), SearchFilter.hasExtensionCI(p, ext),
                               "\(ext) on \(p)")
            }
        }
    }

    /// The sidecar table is byte-identical to the one `[String]` produced, and decodes back to the
    /// same ids and keys.
    func testSidecarTableRoundTripsByteForByte() {
        let corpus = ["/a/b.txt", "/a/c.txt", "noslash", "", "/Users/u/Cafe\u{0301}/x", "/a/b.txt"]
        var t = PathTable()
        for p in corpus { t.append(p) }
        let (offs, blob) = t.sidecarTable()
        var wantBlob = [UInt8](), wantOffs = [UInt8]()
        for p in corpus {
            withUnsafeBytes(of: UInt32(wantBlob.count).littleEndian) { wantOffs.append(contentsOf: $0) }
            wantBlob.append(contentsOf: p.utf8)
        }
        withUnsafeBytes(of: UInt32(wantBlob.count).littleEndian) { wantOffs.append(contentsOf: $0) }
        XCTAssertEqual([UInt8](blob), wantBlob)
        XCTAssertEqual([UInt8](offs), wantOffs)
        guard let back = PathTable.decode(offsets: offs, blob: blob, count: corpus.count) else { return XCTFail("decode") }
        XCTAssertEqual(back.count, corpus.count); XCTAssertEqual(back.keyCount, t.keyCount)
        for (i, p) in corpus.enumerated() { XCTAssertEqual(back[i], p) }
        XCTAssertEqual(back.id("/a/b.txt"), 5, "a duplicate re-points its key, as the dictionary did")
        XCTAssertNil(PathTable.decode(offsets: offs.prefix(4), blob: blob, count: corpus.count),
                     "a short offsets table is rejected, not trapped on")
    }

    private static let tricky: [String] = [
        "", "noslash", "/root.txt", "/a", "/a/b", "/a/bc", "/a/b/c.txt", "/a/bc/d.TXT", "/a/b/c.txt.bak",
        "/Users/u/Caf\u{00E9}/x.txt", "/Users/u/Cafe\u{0301}/y.txt", "/Users/u/Caf\u{00E9}",
        "/Users/u/Caf\u{00E9}/\u{0301}lead.txt", "/k/\u{212A}elvin/z.txt", "/k/Kelvin/z.txt",
        "/Users/u/\u{6587}\u{6863}/\u{62A5}\u{544A}.pdf", "photos://all/ABC%2FL0/IMG_1.HEIC",
        "/a/b/c", "/a/b/c/", "/A/b.txt", "/a/B.txt",
    ]

    /// `less` is Swift's String `<`, on every ordered pair - ASCII by bytes, the rest by String.
    func testLessMatchesStringOrderingOnEveryPair() {
        var t = PathTable()
        for p in Self.tricky { t.append(p) }
        for i in Self.tricky.indices {
            for j in Self.tricky.indices {
                XCTAssertEqual(t.less(i, j), Self.tricky[i] < Self.tricky[j],
                               "\(Self.tricky[i].debugDescription) < \(Self.tricky[j].debugDescription)")
            }
        }
    }

    /// The compiled filter is `acceptsPath`, on every file, across folder / extension / tag
    /// combinations including the Unicode cases.
    func testCompiledFilterMatchesAcceptsPath() {
        var t = PathTable()
        for p in Self.tricky { t.append(p) }
        var filters: [SearchFilter] = []
        for folders in [[], ["/a"], ["/a/b"], ["/Users/u/Caf\u{00E9}"], ["/Users/u/Cafe\u{0301}"],
                        ["/a", "/k/Kelvin"], ["/a/b/c.txt"], ["photos://all"]] {
            for ext in [nil, "txt", "TXT", "pdf", "b/c"] as [String?] {
                var f = SearchFilter(); f.folderPrefixes = folders; f.ext = ext
                filters.append(f)
            }
        }
        var tagged = SearchFilter(); tagged.tagAllow = ["/a/b/c.txt", "/Users/u/Cafe\u{0301}/x.txt"]
        filters.append(tagged)
        var denied = SearchFilter(); denied.folderPrefix = "/a"; denied.tagDeny = ["/a/bc/d.TXT"]
        filters.append(denied)
        for f in filters {
            let c = CompiledPathFilter(f, table: t)
            for (i, p) in Self.tricky.enumerated() {
                XCTAssertEqual(c.accepts(i), f.acceptsPath(p),
                               "folders \(f.folderPrefixes) ext \(f.ext ?? "-") on \(p.debugDescription)")
            }
        }
    }
}
