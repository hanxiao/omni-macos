import XCTest
@testable import OmniKit

/// What the transcript cache promises: the same page under the same prompt comes back, and
/// anything that would change the transcript does not.
///
/// These matter because a wrong answer here is silent. A cache that misses costs a decode nobody
/// notices; a cache that HITS when it should not shows a reader the transcript of a document they
/// have since edited, or of a prompt they have since changed, with no sign that anything is stale.
final class OCRCacheTests: XCTestCase {
    private var dir: URL!
    private var corpus: URL!
    private var savedDirectory: String?
    private var savedEnabled: Any?

    override func setUpWithError() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-ocr-cache-\(UUID().uuidString)", isDirectory: true)
        dir = root.appendingPathComponent("cache", isDirectory: true)
        corpus = root.appendingPathComponent("corpus", isDirectory: true)
        try FileManager.default.createDirectory(at: corpus, withIntermediateDirectories: true)
        // The settings live in UserDefaults, so they are put back afterwards rather than left
        // pointing a developer's own app at a deleted temporary directory.
        savedDirectory = UserDefaults.standard.string(forKey: "omni.ocr.cache.dir")
        savedEnabled = UserDefaults.standard.object(forKey: "omni.ocr.cache.enabled")
        OCRCache.directory = dir
        OCRCache.isEnabled = true
    }

    override func tearDownWithError() throws {
        if let savedDirectory { UserDefaults.standard.set(savedDirectory, forKey: "omni.ocr.cache.dir") }
        else { UserDefaults.standard.removeObject(forKey: "omni.ocr.cache.dir") }
        if let savedEnabled { UserDefaults.standard.set(savedEnabled, forKey: "omni.ocr.cache.enabled") }
        else { UserDefaults.standard.removeObject(forKey: "omni.ocr.cache.enabled") }
        try? FileManager.default.removeItem(at: dir.deletingLastPathComponent())
    }

    @discardableResult
    private func write(_ name: String, _ body: String) throws -> URL {
        let url = corpus.appendingPathComponent(name)
        try body.write(to: url, atomically: true, encoding: .utf8)
        return url
    }

    func testAPageComesBack() throws {
        let source = try write("scan.pdf", "pretend this is a scan")
        OCRCache.write("# Page 3", source: source, page: 2, prompt: "P", variant: "balanced")
        XCTAssertEqual(OCRCache.read(source: source, page: 2, prompt: "P", variant: "balanced"),
                       "# Page 3")
    }

    func testPagesDoNotBleedIntoEachOther() throws {
        let source = try write("scan.pdf", "pretend this is a scan")
        OCRCache.write("page one", source: source, page: 0, prompt: "P", variant: "balanced")
        OCRCache.write("page two", source: source, page: 1, prompt: "P", variant: "balanced")
        XCTAssertEqual(OCRCache.read(source: source, page: 0, prompt: "P", variant: "balanced"), "page one")
        XCTAssertEqual(OCRCache.read(source: source, page: 1, prompt: "P", variant: "balanced"), "page two")
    }

    /// The user's stated condition: the cache is valid as long as the prompt has not changed.
    func testADifferentPromptMisses() throws {
        let source = try write("scan.pdf", "pretend this is a scan")
        OCRCache.write("tables only", source: source, page: 0, prompt: "P", variant: "balanced")
        XCTAssertNil(OCRCache.read(source: source, page: 0, prompt: "P but different", variant: "balanced"))
        XCTAssertEqual(OCRCache.read(source: source, page: 0, prompt: "P", variant: "balanced"),
                       "tables only", "the original prompt still hits")
    }

    /// Different weights produce different text, so a transcript is not transferable between them.
    func testADifferentVariantMisses() throws {
        let source = try write("scan.pdf", "pretend this is a scan")
        OCRCache.write("from balanced", source: source, page: 0, prompt: "P", variant: "balanced")
        XCTAssertNil(OCRCache.read(source: source, page: 0, prompt: "P", variant: "compact"))
    }

    /// The case a path-and-date key gets wrong, set up so it cannot pass by accident.
    ///
    /// Both versions are the SAME LENGTH and both modification dates are stamped to the same fixed
    /// instant, so neither half of a size-and-date check can notice. Written that way deliberately:
    /// the first version of this test restored the date with `setAttributes` and passed only
    /// because the restored value came back a few hundred nanoseconds off, which proved nothing.
    func testEditingTheSourceInPlaceMisses() throws {
        let fixed = Date(timeIntervalSince1970: 1_700_000_000)
        let source = try write("scan.pdf", "version one")
        XCTAssertEqual("version one".count, "version two".count, "the two versions must be the same size")
        try FileManager.default.setAttributes([.modificationDate: fixed], ofItemAtPath: source.path)
        OCRCache.write("transcript of version one", source: source, page: 0, prompt: "P", variant: "balanced")
        XCTAssertNotNil(OCRCache.read(source: source, page: 0, prompt: "P", variant: "balanced"))

        try "version two".write(to: source, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.modificationDate: fixed], ofItemAtPath: source.path)
        let after = try FileManager.default.attributesOfItem(atPath: source.path)
        XCTAssertEqual((after[.modificationDate] as? Date), fixed,
                       "precondition: the date really is identical, so only content can be the signal")

        XCTAssertNil(OCRCache.read(source: source, page: 0, prompt: "P", variant: "balanced"),
                     "a rewritten file served the previous version's transcript")
    }

    /// The reason the key is the CONTENT and not the path: the same document in two places is one
    /// document, and the second copy must not cost a second run.
    func testACopyElsewhereHits() throws {
        let source = try write("scan.pdf", "pretend this is a scan")
        OCRCache.write("shared", source: source, page: 0, prompt: "P", variant: "balanced")
        let elsewhere = corpus.appendingPathComponent("sub", isDirectory: true)
        try FileManager.default.createDirectory(at: elsewhere, withIntermediateDirectories: true)
        let copy = elsewhere.appendingPathComponent("scan.pdf")
        try FileManager.default.copyItem(at: source, to: copy)
        XCTAssertEqual(OCRCache.read(source: copy, page: 0, prompt: "P", variant: "balanced"), "shared")
    }

    func testDisabledNeitherReadsNorWrites() throws {
        let source = try write("scan.pdf", "pretend this is a scan")
        OCRCache.write("written while on", source: source, page: 0, prompt: "P", variant: "balanced")
        OCRCache.isEnabled = false
        XCTAssertNil(OCRCache.read(source: source, page: 0, prompt: "P", variant: "balanced"))
        OCRCache.write("written while off", source: source, page: 1, prompt: "P", variant: "balanced")
        OCRCache.isEnabled = true
        XCTAssertNil(OCRCache.read(source: source, page: 1, prompt: "P", variant: "balanced"),
                     "a write made while the cache was off left a file behind")
    }

    /// Clear must not be a way to empty a folder the user chose for something else. Only files
    /// this cache could have written are counted or removed.
    func testClearOnlyTouchesItsOwnFiles() throws {
        let source = try write("scan.pdf", "pretend this is a scan")
        OCRCache.write("mine", source: source, page: 0, prompt: "P", variant: "balanced")
        OCRCache.write("mine too", source: source, page: 1, prompt: "P", variant: "balanced")

        // A hand-written note and a transcript the user renamed - neither matches the name test.
        let note = dir.appendingPathComponent("Notes.md")
        try "not ours".write(to: note, atomically: true, encoding: .utf8)
        let renamed = dir.appendingPathComponent("my transcript.md")
        try "also not ours".write(to: renamed, atomically: true, encoding: .utf8)

        XCTAssertEqual(OCRCache.clear(), 2, "clear removed a file it did not write")
        XCTAssertEqual(try String(contentsOf: note, encoding: .utf8), "not ours")
        XCTAssertEqual(try String(contentsOf: renamed, encoding: .utf8), "also not ours")
        XCTAssertNil(OCRCache.read(source: source, page: 0, prompt: "P", variant: "balanced"))
    }

    func testClearSurvivesAMissingFolder() {
        OCRCache.directory = dir.appendingPathComponent("never-created", isDirectory: true)
        XCTAssertEqual(OCRCache.clear(), 0)
    }

    /// A name is only a name. This pins the shape the Settings pane's Clear button depends on.
    func testFileNameCarriesThePageAndTheKey() {
        let name = OCRCache.fileName(source: URL(fileURLWithPath: "/tmp/Quarterly Report.pdf"),
                                     page: 6, key: "3f9a1c04d8b27e15")
        XCTAssertEqual(name, "Quarterly Report-p7-3f9a1c04d8b27e15.md")
        let image = OCRCache.fileName(source: URL(fileURLWithPath: "/tmp/receipt.png"),
                                      page: nil, key: "0123456789abcdef")
        XCTAssertEqual(image, "receipt-0123456789abcdef.md")
    }

    func testHashingIsStreamedAndStable() throws {
        // Two chunks' worth, so the streaming loop runs more than once.
        let big = String(repeating: "abcdefgh", count: 1 << 20)      // 8 MB
        let source = try write("big.pdf", big)
        let first = OCRCache.digest(of: source)
        XCTAssertEqual(first?.count, 64)
        XCTAssertEqual(first, OCRCache.digest(of: source), "the memo returned a different answer")
        XCTAssertEqual(first, OCRCache.digest(ofText: big),
                       "the file digest disagreed with the digest of its own contents")
    }
}
