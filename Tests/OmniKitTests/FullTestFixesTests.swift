import CoreText
import PDFKit
import XCTest
@testable import OmniKit

/// The defects the full test of 2026-11-03 found, each pinned where it lives: `in:` by a folder's
/// own name, a binary file with a text extension, a locked PDF, and the page a PDF passage is on.
final class FullTestFixesTests: XCTestCase {
    private var dir: URL!

    override func setUpWithError() throws {
        dir = FileManager.default.temporaryDirectory.appendingPathComponent("fulltest-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: dir) }

    // MARK: - in: by a folder's name

    /// Every folder whose path ends in the name's components, any case, and only whole components:
    /// "Edge" is not "Edges" or "OldEdge".
    func testFoldersNamed() {
        var t = PathTable()
        for p in ["/r/corpus/Edge/a.txt", "/r/corpus/Edge/deep/b.txt", "/r/other/edge/c.txt",
                  "/r/corpus/Edges/d.txt", "/r/corpus/OldEdge/e.txt", "/r/x/Edge/y/Edge/f.txt",
                  "/r/Dokumente/Gr\u{00FC}n/g.txt", "/r/only/deep/Inbox/sub/h.txt"] {
            _ = t.append(p)
        }
        XCTAssertEqual(t.folders(named: "Edge"),
                       ["/r/corpus/Edge", "/r/other/edge", "/r/x/Edge", "/r/x/Edge/y/Edge"])
        XCTAssertEqual(t.folders(named: "EDGE/deep"), ["/r/corpus/Edge/deep"])
        XCTAssertEqual(t.folders(named: "Edge/"), t.folders(named: "Edge"))
        // A folder holding only folders is found through the paths under it.
        XCTAssertEqual(t.folders(named: "inbox"), ["/r/only/deep/Inbox"])
        // Not ASCII: by String comparison, so an NFD spelling finds the NFC one.
        XCTAssertEqual(t.folders(named: "gru\u{0308}n"), ["/r/Dokumente/Gr\u{00FC}n"])
        XCTAssertEqual(t.folders(named: "Nowhere"), [])
        XCTAssertEqual(t.folders(named: ""), [])
    }

    private func vec(_ i: Int, dim: Int = 64) -> [Float] {
        var x = UInt64(0xA076_1D64_78BD_642F) &+ UInt64(i) &* 0x9E37_79B9_7F4A_7C15
        var v = [Float](repeating: 0, count: dim), n: Float = 0
        for k in 0 ..< dim {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17
            v[k] = Float(Int32(truncatingIfNeeded: x)) / Float(Int32.max); n += v[k] * v[k]
        }
        n = n.squareRoot(); return v.map { $0 / n }
    }

    /// Through the store, on both reducers: `in:Edge` is every folder named Edge, `-in:Edge` none of
    /// them, and a name no folder has matches nothing (it used to be read as the path "/Edge").
    func testSearchScopedByFolderName() throws {
        let store = try VectorStore(dbURL: dir.appendingPathComponent("index.sqlite"))
        defer { store.close(); VectorStore.gpuReduce = true }
        let folders = ["/r/a/Edge", "/r/b/edge", "/r/a/Other"]
        func path(_ i: Int) -> String { "\(folders[i % 3])/f\(i).txt" }
        var batch: [(path: String, chunks: [IndexedChunk])] = []
        for i in 0 ..< 300 {
            batch.append((path(i), [IndexedChunk(path: path(i), modified: 1_700_000_000, size: 1, kind: "text", chunkIndex: 0,
                                                 snippet: "f\(i)", embedding: vec(i), chunkKey: String(format: "%08x", 0x4000_0000 + i))]))
        }
        try store.replaceMany(batch)
        for gpu in [true, false] {
            VectorStore.gpuReduce = gpu
            var inEdge = SearchFilter(); inEdge.folderNames = ["Edge"]
            var notEdge = SearchFilter(); notEdge.excludeFolderNames = ["edge"]
            var nowhere = SearchFilter(); nowhere.folderNames = ["Nowhere"]
            for q in [0, 1, 2, 30, 31, 32] {
                let a = store.search(vec(q), filter: inEdge, topK: 20)
                XCTAssertFalse(a.isEmpty, "gpu=\(gpu) q=\(q)")
                XCTAssertTrue(a.allSatisfy { !$0.path.hasPrefix("/r/a/Other/") }, "gpu=\(gpu): in:Edge let Other in")
                if q % 3 != 2 { XCTAssertEqual(a.first?.path, path(q)) }
                let b = store.search(vec(q), filter: notEdge, topK: 20)
                XCTAssertTrue(b.allSatisfy { $0.path.hasPrefix("/r/a/Other/") }, "gpu=\(gpu): -in:edge kept an Edge file")
                XCTAssertTrue(store.search(vec(q), filter: nowhere, topK: 20).isEmpty, "gpu=\(gpu): a name no folder has")
            }
        }
    }

    // MARK: - A pinned Recents, and every copy of a content key

    private func smallStore(_ paths: [String]) throws -> VectorStore {
        let store = try VectorStore(dbURL: dir.appendingPathComponent("s-\(UUID().uuidString).sqlite"))
        var batch: [(path: String, chunks: [IndexedChunk])] = []
        for (i, p) in paths.enumerated() {
            batch.append((p, [IndexedChunk(path: p, modified: 1_700_000_000, size: 10, kind: "image", chunkIndex: 0,
                                           snippet: "f\(i)", embedding: vec(i), chunkKey: String(format: "%08x", 0x5000_0000 + i))]))
        }
        try store.replaceMany(batch)
        return store
    }

    /// A refresh keeps the Recents its search started with: with `recentsPaths` the scope is those
    /// files, not the newest now.
    func testPinnedRecentsScope() throws {
        let paths = (0 ..< 40).map { "/r/n/f\($0).jpg" }
        let store = try smallStore(paths)
        defer { store.close() }
        var live = SearchFilter(); live.recentsLimit = 100
        XCTAssertEqual(store.search(vec(3), filter: live, topK: 5).first?.path, paths[3])
        var pinned = live; pinned.recentsPaths = [paths[7], paths[8]]
        let hits = store.search(vec(3), filter: pinned, topK: 5)
        XCTAssertEqual(Set(hits.map(\.path)), [paths[7], paths[8]])
        XCTAssertEqual(store.recentPaths(limit: 100).count, 40)
    }

    /// Every copy of a content key, as the filter allows - not only the ones a result pool holds.
    func testCopiesOfAContentKey() throws {
        let paths = ["/r/a/x.jpg", "/r/b/x.jpg", "/r/c/x.jpg", "/r/a/other.jpg"]
        let store = try smallStore(paths)
        defer { store.close() }
        let key = "00112233445566778899aabbccddeeff"
        store.recordContentKeys(paths.prefix(3).map { (path: $0, key: key, modified: 1_700_000_000, size: 10) }
                                + [(path: paths[3], key: "ffeeddccbbaa99887766554433221100", modified: 1_700_000_000, size: 10)])
        // By the key as contentKeys(paths:) gives it (the stored digest, as hex), which is what grouping holds.
        let hex = try XCTUnwrap(store.contentKeys(paths: [paths[0]])[paths[0]]?.key)
        XCTAssertEqual(store.contentKeys(paths: [paths[2]])[paths[2]]?.key, hex)
        XCTAssertEqual(Set(store.copies(ofContentKeys: [hex], filter: SearchFilter())[hex]?.map(\.path) ?? []), Set(paths.prefix(3)))
        var f = SearchFilter(); f.excludeFolders = ["/r/b"]
        XCTAssertEqual(Set(store.copies(ofContentKeys: [hex], filter: f)[hex]?.map(\.path) ?? []), [paths[0], paths[2]])
        var k = SearchFilter(); k.kinds = ["text"]
        XCTAssertNil(store.copies(ofContentKeys: [hex], filter: k)[hex])
        XCTAssertTrue(store.copies(ofContentKeys: ["not hex"], filter: SearchFilter()).isEmpty)
    }

    // MARK: - Binary with a text extension, and a locked PDF

    func testBinaryWithATextExtensionIsNotText() throws {
        var bytes = [UInt8](repeating: 0, count: 200_000)
        var x: UInt64 = 0x9E37_79B9_7F4A_7C15
        for i in bytes.indices { x ^= x << 13; x ^= x >> 7; x ^= x << 17; bytes[i] = UInt8(truncatingIfNeeded: x) }
        let binary = dir.appendingPathComponent("binary-as-text.txt")
        try Data(bytes).write(to: binary)
        XCTAssertTrue(FileExtractor.looksBinary(binary))
        guard case .empty = try FileExtractor.extract(binary) else { return XCTFail("a binary .txt was read as text") }

        let text = dir.appendingPathComponent("notes.txt")
        try Data("caf\u{00E9} notes, line one\nline two\n".utf8).write(to: text)
        XCTAssertFalse(FileExtractor.looksBinary(text))
        // Latin-1 is text: no NUL in it.
        XCTAssertFalse(FileExtractor.looksBinary(Data([0x63, 0x61, 0x66, 0xE9, 0x0A])))
        // UTF-16 has NULs and is text: its byte-order mark says so.
        XCTAssertFalse(FileExtractor.looksBinary(Data([0xFF, 0xFE, 0x61, 0x00, 0x62, 0x00])))
        XCTAssertTrue(FileExtractor.looksBinary(Data([0x61, 0x00, 0x62, 0x00])))
    }

    func testLockedPDFIsNotIndexedAsAScan() throws {
        let url = dir.appendingPathComponent("plain.pdf")
        writePDF(url, pages: [["A page of ordinary words about lighthouses and ferries, long enough to read as text"]])
        let doc = try XCTUnwrap(PDFDocument(url: url))
        let locked = dir.appendingPathComponent("locked.pdf")
        XCTAssertTrue(doc.write(to: locked, withOptions: [.userPasswordOption: "secret", .ownerPasswordOption: "owner"]))
        XCTAssertEqual(PDFDocument(url: locked)?.isLocked, true)
        guard case .empty = try FileExtractor.extract(locked) else { return XCTFail("a locked PDF was indexed") }
        if case .empty = try FileExtractor.extract(url) { XCTFail("the unlocked PDF is text") }
    }

    // MARK: - The page a PDF passage starts on

    private func writePDF(_ url: URL, pages: [[String]]) {
        var box = CGRect(x: 0, y: 0, width: 612, height: 792)
        let ctx = CGContext(url as CFURL, mediaBox: &box, nil)!
        let font = CTFontCreateWithName("Helvetica" as CFString, 9, nil)
        for lines in pages {
            ctx.beginPDFPage(nil)
            var y: CGFloat = 760
            for line in lines {
                let l = CTLineCreateWithAttributedString(NSAttributedString(string: line, attributes: [.init(kCTFontAttributeName as String): font]))
                ctx.textPosition = CGPoint(x: 20, y: y); CTLineDraw(l, ctx); y -= 14
            }
            ctx.endPDFPage()
        }
        ctx.closePDF()
    }

    /// The reported shape: the passage is the sentence at the foot of page 2, its chunk mostly on
    /// page 3, so the chunk's locator says 3. The passage starts on 2; a window running over the
    /// break is on the page it starts on; a passage on no page near keeps the locator.
    func testPassageIsOnThePageItStartsOn() throws {
        let url = dir.appendingPathComponent("book.pdf")
        let filler = (0 ..< 30).map { "ferry bay cargo the ferry and lights and ferry carrying number \($0)" }
        writePDF(url, pages: [
            ["Chapter one opens the book with a different subject entirely"],
            filler + ["The lighthouse keeper hid the brass telescope under the stairs."],
            ["Chapter 24 page three toward ferry bay cargo the ferry and lights"] + filler,
        ])
        let doc = try XCTUnwrap(PDFDocument(url: url))
        XCTAssertEqual(PagePhrase.page(of: "The lighthouse keeper hid the brass telescope under the stairs.", in: doc, near: 3), 2)
        XCTAssertEqual(PagePhrase.page(of: "\u{2026}keeper hid the brass telescope under the stairs. Chapter 24 page three", in: doc, near: 3), 2)
        XCTAssertEqual(PagePhrase.page(of: "Chapter 24 page three toward ferry bay", in: doc, near: 3), 3)
        XCTAssertNil(PagePhrase.page(of: "words that are on no page of this document at all", in: doc, near: 3))
    }
}
