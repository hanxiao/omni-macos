import XCTest
@testable import OmniKit

/// The special tokens come from each model's config, and a model without one has none. They were
/// Small's ids for every model: past the end of Nano's 128,260-row table, so Nano's audio and
/// images were wrapped in undefined memory and embedded differently from one launch to the next.
final class SpecialTokenTests: XCTestCase {
    func testNanoUsesItsOwnTokensAndNoVisionWrappers() throws {
        guard let dir = ProcessInfo.processInfo.environment["OMNI_NANO_MODEL_DIR"] else { throw XCTSkip("OMNI_NANO_MODEL_DIR not set") }
        let cfg = try OmniConfig(modelDir: URL(fileURLWithPath: dir))
        XCTAssertEqual(cfg.audioStartTokenId, 128257)
        XCTAssertEqual(cfg.audioEndTokenId, 128258)
        XCTAssertNil(cfg.visionStartTokenId, "Nano's processor strips <|vision_start|>; there is no such row")
        XCTAssertNil(cfg.visionEndTokenId)
        XCTAssertEqual(cfg.imageTokenId, 128259)
    }

    func testSmallKeepsItsWrappers() throws {
        guard let dir = ProcessInfo.processInfo.environment["OMNI_MODEL_DIR"] else { throw XCTSkip("OMNI_MODEL_DIR not set") }
        let cfg = try OmniConfig(modelDir: URL(fileURLWithPath: dir))
        XCTAssertEqual(cfg.visionStartTokenId, 151652)
        XCTAssertEqual(cfg.visionEndTokenId, 151653)
        XCTAssertEqual(cfg.audioStartTokenId, 151670)
        XCTAssertEqual(cfg.audioEndTokenId, 151671)
    }
}
