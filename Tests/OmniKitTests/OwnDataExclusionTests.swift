import XCTest
@testable import OmniKit

/// Omni must not index its own data. Reported as "I changed the download folder for the models,
/// it ignores it, tries to actually index it" (issue #21's follow-up): a model directory holds
/// tokenizer.json, 16 MB of vocabulary JSON, which is indexable text - so putting the model
/// download anywhere inside an indexed folder fed Omni's own tokenizer into the index.
final class OwnDataExclusionTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-owndata-\(UUID().uuidString)", isDirectory: true)
        let models = root.appendingPathComponent("models", isDirectory: true)
        try FileManager.default.createDirectory(at: models, withIntermediateDirectories: true)
        // What a real model directory actually contains, by extension.
        try "{}".write(to: models.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try "{}".write(to: models.appendingPathComponent("tokenizer.json"), atomically: true, encoding: .utf8)
        // And an ordinary user file beside it, which must still be indexed.
        try "notes".write(to: root.appendingPathComponent("notes.md"), atomically: true, encoding: .utf8)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    private func crawl(ownData: [String]) -> [String] {
        var found: [String] = []
        FileCrawler(roots: [root], enabledKinds: [.text], ownDataPaths: ownData)
            .walk(shouldContinue: { true }) { found.append(($0.path as NSString).lastPathComponent) }
        return found.sorted()
    }

    /// THE BUG. With nothing excluded, the model's own JSON is crawled as user content.
    func testWithoutExclusionTheModelFilesAreCrawled() {
        let found = crawl(ownData: [])
        XCTAssertTrue(found.contains("tokenizer.json"), "this is the reported behaviour")
        XCTAssertTrue(found.contains("config.json"))
    }

    /// THE FIX, and the test that fails if the exclusion is removed.
    func testTheModelDirectoryIsSkipped() {
        let found = crawl(ownData: [root.appendingPathComponent("models").path])
        XCTAssertFalse(found.contains("tokenizer.json"))
        XCTAssertFalse(found.contains("config.json"))
        XCTAssertEqual(found, ["notes.md"], "a real file beside it must still be indexed")
    }

    /// A sibling that merely starts with the same characters must NOT be swallowed: excluding
    /// "/x/models" cannot take "/x/models-notes".
    func testAPrefixSiblingIsNotExcluded() throws {
        let sibling = root.appendingPathComponent("models-notes", isDirectory: true)
        try FileManager.default.createDirectory(at: sibling, withIntermediateDirectories: true)
        try "keep".write(to: sibling.appendingPathComponent("keep.md"), atomically: true, encoding: .utf8)
        let found = crawl(ownData: [root.appendingPathComponent("models").path])
        XCTAssertTrue(found.contains("keep.md"), "boundary test, not a bare hasPrefix")
        XCTAssertFalse(found.contains("tokenizer.json"))
    }
}
