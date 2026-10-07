import XCTest
@testable import OmniKit

/// The label cache is keyed by the weights it was built from, not just their width: a different
/// checkpoint of the same size must not reuse another's labels.
final class TagCacheIdentityTests: XCTestCase {
    private func model(_ bytes: Data) throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("tagid-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try bytes.write(to: dir.appendingPathComponent("model.safetensors"))
        return dir
    }

    func testSameWeightsSameIdDifferentWeightsDifferentId() throws {
        var a = Data((0 ..< 3_000_000).map { UInt8(truncatingIfNeeded: $0 &* 31) })
        let d1 = try model(a), d2 = try model(a)
        a[a.count / 16 * 7 + 5] ^= 0xFF          // one byte inside a sampled slice, same size
        let d3 = try model(a)
        defer { for d in [d1, d2, d3] { try? FileManager.default.removeItem(at: d) } }
        let i1 = try XCTUnwrap(OmniTagger.modelIdentity(modelDir: d1))
        XCTAssertEqual(i1, OmniTagger.modelIdentity(modelDir: d2), "a re-download of the same weights keeps its cache")
        XCTAssertNotEqual(i1, OmniTagger.modelIdentity(modelDir: d3), "other weights of the same size get their own")
        XCTAssertNil(OmniTagger.modelIdentity(modelDir: d1.appendingPathComponent("missing")))
    }
}
