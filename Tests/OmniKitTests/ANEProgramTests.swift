import XCTest
@testable import OmniKit

/// The Neural Engine program's weight file, as the app writes it on the user's Mac from its own
/// weights (ANEProgram), is byte for byte the one that was built and measured: its SHA-256 is the
/// recipe's. Without that the app refuses the Neural Engine, so a change to the writer, to fp16
/// rounding (Half) or to the weight load that broke it would silently put everyone on the GPU.
final class ANEProgramTests: XCTestCase {
    private func check(_ env: String, _ name: String) throws {
        guard #available(macOS 15.0, *) else { throw XCTSkip("needs macOS 15") }
        guard let dir = ProcessInfo.processInfo.environment[env], FileManager.default.fileExists(atPath: dir) else {
            throw XCTSkip("\(env) not set")
        }
        let modelDir = URL(fileURLWithPath: dir)
        let cfg = try OmniConfig(modelDir: modelDir)
        XCTAssertEqual(OmniEngine.aneProgramName(cfg), name)
        let ws = try WeightStore(modelDir: modelDir, loraScale: cfg.loraScale, keepVision: false, keepAudio: false)
        let out = FileManager.default.temporaryDirectory.appendingPathComponent("ane-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: out) }
        XCTAssertTrue(try ANEProgram.writeWeightsChecked(name, weights: ws, to: out), "\(name): the weights written are not the ones built")
    }

    func testNanoWeightsAreTheOnesBuilt() throws { try check("OMNI_NANO_MODEL_DIR", "nano_text") }
    func testSmallWeightsAreTheOnesBuilt() throws { try check("OMNI_MODEL_DIR", "small_text") }
}
