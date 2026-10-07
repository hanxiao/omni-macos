import XCTest
import Foundation
import CoreGraphics
import AVFoundation
@testable import OmniKit

/// Opt-in hardware integration test; runs the official checkpoint through the Swift/MLX pipe.
/// OMNI_GEMMA_INTEGRATION=1 OMNI_MODEL_DIR=<checkpoint> Scripts/run-tests.sh OmniKitTests.EmbeddingGemmaIntegrationTests
final class EmbeddingGemmaIntegrationTests: XCTestCase {
    func testFrenchRetrievalMediaAndIndexer() async throws {
        guard ProcessInfo.processInfo.environment["OMNI_GEMMA_INTEGRATION"] == "1",
              let path = ProcessInfo.processInfo.environment["OMNI_MODEL_DIR"] else {
            throw XCTSkip("Requires the local EmbeddingGemma 2 checkpoint and runtime")
        }
        let engine = try await OmniEngine.loadValidated(modelDir: URL(fileURLWithPath: path))
        XCTAssertTrue(engine.isEmbeddingGemma2)
        XCTAssertEqual(engine.dim, 768)
        let documents = [
            "La facture de mars indique un montant de 240 euros pour la réparation de la pompe.",
            "Le contrat de location de l'appartement prend fin au mois de juillet.",
            "Le rapport de maintenance décrit une panne de roulement sur le moteur industriel.",
            "La recette du gâteau au chocolat nécessite du beurre et trois œufs."
        ]
        let queries = ["Combien coûte la réparation de la pompe ?", "Quand se termine le bail ?",
                       "Quelle panne a été trouvée sur le moteur ?", "Comment préparer un gâteau au chocolat ?"]
        let corpus = engine.embedTextBatch(documents, as: .passage)
        XCTAssertEqual(corpus.count, documents.count)
        for vector in corpus { assertVector(vector) }
        for (expected, query) in queries.enumerated() {
            let vector = engine.embedQuery(query)
            assertVector(vector)
            let scores = corpus.map { dot(vector, $0) }
            let winner = try XCTUnwrap(scores.indices.max(by: { scores[$0] < scores[$1] }))
            XCTAssertEqual(winner, expected, "French retrieval: \(query), scores=\(scores)")
        }
        let fixtureRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Resources")
        let image = try XCTUnwrap(FileExtractor.loadImage(fixtureRoot.appendingPathComponent("test_image.png"), maxDimension: 512))
        let prepared = engine.prepareImage(image)
        XCTAssertNotNil(prepared.encodedImage)
        XCTAssertTrue(prepared.pixels.isEmpty, "Google must not receive Qwen patches")
        assertVector(try XCTUnwrap(engine.embedImages([prepared])?.first))
        assertVector(try XCTUnwrap(engine.embedImageQuery(image)))
        assertVector(try XCTUnwrap(engine.embedVideoFrames([image, image])))
        assertVector(try XCTUnwrap(engine.embedAudio(fixtureRoot.appendingPathComponent("test_audio.wav"))))

        // A >30-second audio file verifies that Google's processor does not truncate to its
        // stale configured 280/750-token cap or consume Jina's log-mels.
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("gemma-integration-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let audio = root.appendingPathComponent("tone.wav")
        let format = try XCTUnwrap(AVAudioFormat(standardFormatWithSampleRate: 16000, channels: 1))
        let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 16000 * 35))
        buffer.frameLength = buffer.frameCapacity
        let channel = try XCTUnwrap(buffer.floatChannelData?[0])
        for i in 0..<Int(buffer.frameLength) { channel[i] = 0.05 * sin(Float(i) * 2 * .pi * 440 / 16000) }
        do {
            let audioFile = try AVAudioFile(forWriting: audio, settings: format.settings)
            try audioFile.write(from: buffer)
        }
        assertVector(try XCTUnwrap(engine.embedAudio(audio)))
        let reader = try XCTUnwrap(OmniAudioPreprocess.AudioSegmentReader(url: audio, rawPCM: true))
        let raw = try XCTUnwrap(reader.nextMelSegment())
        XCTAssertEqual(raw.mel.count, 16000 * 35)
        assertVector(try XCTUnwrap(engine.embedAudioMel(raw.mel, frames: raw.frames)))

        let files = root.appendingPathComponent("corpus")
        try FileManager.default.createDirectory(at: files, withIntermediateDirectories: true)
        for (i, document) in documents.enumerated() {
            try document.write(to: files.appendingPathComponent("document-\(i).txt"), atomically: true, encoding: .utf8)
        }
        try FileManager.default.copyItem(at: fixtureRoot.appendingPathComponent("test_image.png"), to: files.appendingPathComponent("photo.png"))
        try FileManager.default.copyItem(at: fixtureRoot.appendingPathComponent("test_audio.wav"), to: files.appendingPathComponent("recording.wav"))
        let store = try VectorStore(dbURL: root.appendingPathComponent("index.sqlite"))
        let indexer = Indexer(store: store, embedder: engine)
        indexer.index(roots: [files], settings: .default) { _ in }
        let hits = store.search(engine.embedQuery(queries[0]), topK: 6)
        XCTAssertEqual(hits.first.map { URL(fileURLWithPath: $0.path).resolvingSymlinksInPath().path },
                       files.appendingPathComponent("document-0.txt").resolvingSymlinksInPath().path)
        XCTAssertEqual(store.knownFiles().count, 6)
        XCTAssertGreaterThan(engine.tokensProcessed, 0)
        engine.setTowers(keepVision: false, keepAudio: false)
        XCTAssertFalse(engine.supportsImages)
        XCTAssertFalse(engine.supportsAudio)
        XCTAssertNil(engine.embedImage(image))
        assertVector(engine.embedQuery("Bonjour"))
    }
    private func assertVector(_ v: [Float], file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(v.count, 768, file: file, line: line)
        XCTAssertTrue(v.allSatisfy(\.isFinite), file: file, line: line)
        XCTAssertEqual(sqrt(dot(v, v)), 1, accuracy: 0.01, file: file, line: line)
    }
    private func dot(_ a: [Float], _ b: [Float]) -> Float { zip(a, b).reduce(0) { $0 + $1.0 * $1.1 } }
}
