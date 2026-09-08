import XCTest
@testable import OmniKit

/// The parts of the OCR port that can be wrong WITHOUT the 4 GB weights being present, and that
/// nothing else would catch. The heavy numeric gate is `ocr-verify` against the torch reference
/// dumps (see docs/OCR.md); these cover the pure-Swift logic around it, where a silent change
/// alters the prompt or the page layout and every tensor check still passes.
final class OCRPortTests: XCTestCase {

    /// The chat template's two escapes after `<image>` and before the assistant prefix are
    /// LITERAL backslash-n - the Jinja source writes `'\\n'`. Emitting real newlines there is a
    /// different token stream and therefore a different transcription, with no error anywhere.
    func testChatTemplateUsesLiteralBackslashN() {
        let text = OCRModel.chatText(prompt: "Do the thing.")
        XCTAssertTrue(text.hasPrefix("<|User|>:\n<image>"))
        XCTAssertTrue(text.contains("<image>\\nDo the thing.\\n<|Assistant|>:\n"),
                      "template must carry literal backslash-n, got: \(text.debugDescription)")
        XCTAssertFalse(text.contains("<image>\nDo the thing"),
                       "a real newline after <image> changes the token stream")
        // Exactly one <image> marker: the visual block is assembled once per request, and a
        // second marker used to surface as a cryptic unpacking error.
        XCTAssertEqual(text.components(separatedBy: "<image>").count - 1, 1)
    }

    func testChatTemplateStripsTrailingWhitespaceFromPrompt() {
        let text = OCRModel.chatText(prompt: "Transcribe.\n\n  ")
        XCTAssertTrue(text.contains("Transcribe.\\n<|Assistant|>:\n"))
    }

    /// Visual queries per side: 1024 -> 16, 640 -> 10. These two numbers set the `<image>` token
    /// count, and a mismatch against the encoder's output is a hard failure at assembly time.
    func testQueryCounts() {
        XCTAssertEqual(OCRPreprocess.queries(size: 1024), 16)
        XCTAssertEqual(OCRPreprocess.queries(size: 640), 10)
    }

    /// Tile grid selection, against the reference's own rule (candidates with
    /// `2 <= i*j <= 9` sorted by area, nearest aspect ratio, ties broken by the area test).
    func testCropGridMatchesReferenceChoices() {
        // The bench pages: 1240x2100 portrait picks 2x3, and its 1007-token prompt depends on it.
        XCTAssertEqual(OCRPreprocess.closestAspectRatio(width: 1240, height: 2100, imageSize: 640).0, 2)
        XCTAssertEqual(OCRPreprocess.closestAspectRatio(width: 1240, height: 2100, imageSize: 640).1, 3)
        // 300 dpi A4 lands on the same grid - the tile layout is fixed by aspect, not by size,
        // which is why a 2480x3508 scan produces the same 1007 prompt tokens as a 1240x2100 page.
        let scan = OCRPreprocess.closestAspectRatio(width: 2480, height: 3508, imageSize: 640)
        XCTAssertEqual(scan.0, 2)
        XCTAssertEqual(scan.1, 3)
        // A wide panorama must not choose a portrait grid.
        let wide = OCRPreprocess.closestAspectRatio(width: 3200, height: 180, imageSize: 640)
        XCTAssertGreaterThan(wide.0, wide.1)
    }

    /// Pillow's `ImageOps.pad`: contain to the box, then centre on the grey canvas. The colour is
    /// 127 because the processor passes `mean * 255 = 127.5` through `int()`.
    func testPadSquareCentresAndUsesTheProcessorGrey() {
        // 100x50 -> contained to 1024x512, then centred vertically in 1024x1024.
        let source = OCRImage(width: 100, height: 50,
                              rgb: [UInt8](repeating: 200, count: 100 * 50 * 3))
        let padded = OCRPreprocess.padSquare(source, size: 1024)
        XCTAssertEqual(padded.width, 1024)
        XCTAssertEqual(padded.height, 1024)
        // Top row is canvas.
        XCTAssertEqual(padded.rgb[0], 127)
        // Centre row is image content.
        let centre = (512 * 1024 + 512) * 3
        XCTAssertEqual(padded.rgb[centre], 200)
    }

    /// A resize to the same size must be the identity, not a filter round-trip: the global view
    /// of an already-square page and the native SAM grid both depend on it.
    func testResizeToSameSizeIsIdentity() {
        var rgb = [UInt8](repeating: 0, count: 8 * 8 * 3)
        for i in 0 ..< rgb.count { rgb[i] = UInt8(i % 251) }
        let image = OCRImage(width: 8, height: 8, rgb: rgb)
        let same = PILResample.resize(image, outW: 8, outH: 8)
        XCTAssertEqual(same.rgb, rgb)
    }

    /// A flat image must survive resampling flat. Pillow's fixed-point accumulation carries a
    /// rounding term precisely so a constant field does not drift by a LSB.
    func testResizePreservesAConstantField() {
        let image = OCRImage(filling: 173, width: 64, height: 40)
        let down = PILResample.resize(image, outW: 40, outH: 25)
        XCTAssertEqual(down.width, 40)
        XCTAssertEqual(down.height, 25)
        XCTAssertTrue(down.rgb.allSatisfy { $0 == 173 },
                      "constant input must resample to a constant field")
    }

    /// The loop guard fires on genuine degeneration and stays silent on legitimate repetition.
    ///
    /// The threshold is 24 because it was measured: real degeneration repeats one block hundreds
    /// of times, while legitimate document structure (repeated table rows) peaks in single
    /// digits. An earlier default of 3 deleted 1670 valid characters from a real 3-page request.
    func testLoopGuardThreshold() {
        let runaway = Array(repeating: [11, 22, 33], count: 40).flatMap { $0 }
        XCTAssertEqual(OCRModel.loopPeriod(runaway, reps: 24), 3)

        // Eight identical table rows are document structure, not a loop.
        let legitimate = Array(repeating: [7, 8, 9, 10], count: 8).flatMap { $0 }
        XCTAssertNil(OCRModel.loopPeriod(legitimate, reps: 24))

        XCTAssertNil(OCRModel.loopPeriod([1, 2, 3], reps: 24))
    }

    /// The manifest names the shards; the downloader has no other source of truth for them.
    func testManifestNamesEveryShard() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-ocr-manifest-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try #"{"shards": 3, "bytes": 4060000000}"#
            .write(to: dir.appendingPathComponent("omni-ocr.json"), atomically: true, encoding: .utf8)

        let manifest = try OCRModelCatalog.readManifest(at: dir)
        XCTAssertTrue(manifest.files.contains("model-00001-of-00003.safetensors"))
        XCTAssertTrue(manifest.files.contains("model-00003-of-00003.safetensors"))
        XCTAssertTrue(manifest.files.contains("tokenizer.json"))
        XCTAssertEqual(manifest.files.filter { $0.hasSuffix(".safetensors") }.count, 3)
    }

    /// A directory with the manifest but a missing shard must NOT read as installed: the loader
    /// would otherwise fail deep inside a safetensors read complaining about a tensor.
    func testPartialInstallDoesNotReadAsInstalled() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("omni-ocr-partial-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try #"{"shards": 2, "bytes": 1}"#
            .write(to: dir.appendingPathComponent("omni-ocr.json"), atomically: true, encoding: .utf8)
        let manifest = try OCRModelCatalog.readManifest(at: dir)
        XCTAssertEqual(manifest.files.filter { $0.hasSuffix(".safetensors") }.count, 2)
        // Nothing else was written, so no variant backed by this directory could be complete.
        XCTAssertFalse(OCRModelCatalog.Variant.allCases.isEmpty)
    }

    /// Asset names are the download contract: variant-prefixed, flat, one tag.
    func testAssetURLShape() throws {
        let url = try XCTUnwrap(OCRModelCatalog.assetURL(variant: .compact, file: "omni-ocr.json"))
        XCTAssertEqual(url.absoluteString,
                       "https://github.com/\(OCRModelCatalog.repository)/releases/download/"
                       + "\(OCRModelCatalog.releaseTag)/jina-ocr-v1-mlx-compact-omni-ocr.json")
    }

    /// Each variant names a policy that `Tools/ocr/convert.py` actually defines. The pairing is
    /// how a published artifact is traced back to how it was built.
    func testVariantPolicies() {
        XCTAssertEqual(OCRModelCatalog.Variant.fidelity.policy, "q8")
        XCTAssertEqual(OCRModelCatalog.Variant.balanced.policy, "dyn-k")
        XCTAssertEqual(OCRModelCatalog.Variant.compact.policy, "dyn-j")
    }
}
