import XCTest
import CoreGraphics
@testable import OmniKit

/// The ingestion-source contract.
///
/// These exist because issue #13 was not a PhotoKit bug, it was an ARCHITECTURE bug: a second
/// ingestion channel restated the first one's policy and the restatement drifted. The protocol
/// removes the opportunity, and these pin the properties a third channel (external disk, special
/// folder, mail store) must not break.
final class ContentSourceTests: XCTestCase {

    private func file(_ path: String, modified: Double = 1_700_000_000, size: Int = 4096) -> CrawledFile {
        CrawledFile(path: path, modified: modified, size: size)
    }

    private func photoPath(_ id: String = "AAAA1111-2222-3333-4444-555566667777/L0/001",
                           source: String = "all", name: String = "IMG_1.HEIC") -> String {
        PhotoLibrary.scheme + PhotoLibrary.esc(source) + "/" + PhotoLibrary.esc(id) + "/" + name
    }

    // MARK: - Routing

    /// Exactly one source owns any item: the claims must partition, not overlap.
    func testClaimsPartitionEveryPath() {
        let disk = file("/Users/x/Pictures/holiday.heic")
        let photo = file(photoPath())

        XCTAssertTrue(FileContentSource.claims(disk))
        XCTAssertFalse(PhotosContentSource.claims(disk))

        XCTAssertTrue(PhotosContentSource.claims(photo))
        XCTAssertFalse(FileContentSource.claims(photo))
    }

    func testResolutionPicksTheOwningSource() {
        XCTAssertTrue(ContentSources.source(for: file("/tmp/a.txt")) is FileContentSource)
        XCTAssertTrue(ContentSources.source(for: file(photoPath())) is PhotosContentSource)
    }

    /// A path that merely LOOKS like a photos path but does not parse must still get a source
    /// rather than trapping - the fallback is what makes routing total.
    func testUnparseablePhotoPathStillResolves() {
        let malformed = file(PhotoLibrary.scheme + "not-a-valid-ref")
        XCTAssertNotNil(ContentSources.source(for: malformed))
    }

    // MARK: - The invariant that broke (#13)

    /// A source must never ask for MORE resolution than the item has. `FileExtractor.loadImage`
    /// gets this free from `kCGImageSourceThumbnailMaxPixelSize` (a cap); the Photos path has to
    /// clamp explicitly, because PhotoKit's `.exact` delivers whatever size it is asked for.
    ///
    /// This is the pipeline-parity property in its most testable form: same picture, same channel-
    /// independent answer.
    func testNoSourceUpscalesBeyondTheItem() {
        for (w, h) in [(4032, 3024), (640, 480), (1568, 1568), (100, 4000)] {
            let side = PhotoLibrary.targetSide(maxDimension: 1568, pixelWidth: w, pixelHeight: h)
            XCTAssertLessThanOrEqual(side, max(w, h), "asked for more pixels than the \(w)x\(h) item has")
            XCTAssertLessThanOrEqual(side, 1568, "exceeded maxImageDimension on a \(w)x\(h) item")
        }
    }

    // MARK: - Content keys

    /// Content keys are the dedup identity: equal key MUST mean equal embedding, so a key has to
    /// move when anything that changes the vectors moves. Checked on the Photos source because its
    /// key is derived (no byte hash available), which is where a drift would hide.
    func testPhotoContentKeyTracksEverythingThatChangesTheVectors() {
        let src = PhotosContentSource(ref: PhotoLibrary.Ref(photoPath())!)
        let f = file(photoPath())
        var settings = IndexSettings()

        let base = src.contentKey(f, kind: .image, dim: 512, chunkOverlap: 200, settings: settings)
        XCTAssertNotNil(base)

        // Model dimension.
        XCTAssertNotEqual(base, src.contentKey(f, kind: .image, dim: 256, chunkOverlap: 200, settings: settings))
        // Preprocess size.
        settings.maxImageDimension = 1024
        XCTAssertNotEqual(base, src.contentKey(f, kind: .image, dim: 512, chunkOverlap: 200, settings: settings))
        settings.maxImageDimension = 1568
        // The tower that will run.
        XCTAssertNotEqual(base, src.contentKey(f, kind: .video, dim: 512, chunkOverlap: 200, settings: settings))
        // The asset's own mtime/size.
        XCTAssertNotEqual(base, src.contentKey(file(photoPath(), modified: 1_700_000_001),
                                               kind: .image, dim: 512, chunkOverlap: 200, settings: settings))
        // A DIFFERENT asset must never share a key with this one.
        let other = photoPath("BBBB1111-2222-3333-4444-555566667777/L0/001")
        XCTAssertNotEqual(base, PhotosContentSource(ref: PhotoLibrary.Ref(other)!)
            .contentKey(file(other), kind: .image, dim: 512, chunkOverlap: 200, settings: settings))
    }

    /// Same asset, same settings, twice: dedup only works if the key is stable.
    func testPhotoContentKeyIsDeterministic() {
        let src = PhotosContentSource(ref: PhotoLibrary.Ref(photoPath())!)
        let f = file(photoPath())
        let s = IndexSettings()
        XCTAssertEqual(src.contentKey(f, kind: .image, dim: 512, chunkOverlap: 200, settings: s),
                       src.contentKey(f, kind: .image, dim: 512, chunkOverlap: 200, settings: s))
    }

    /// The same asset reached through two different albums is the same content and must dedup to
    /// one forward pass - the property that makes overlapping album selections cheap.
    func testSameAssetInTwoAlbumsSharesAContentKey() {
        let id = "CCCC1111-2222-3333-4444-555566667777/L0/001"
        let viaAll = photoPath(id, source: "all")
        let viaAlbum = photoPath(id, source: "5E2F5C3A-0000-4000-8000-000000000001/L0/040")
        let s = IndexSettings()

        let a = PhotosContentSource(ref: PhotoLibrary.Ref(viaAll)!)
            .contentKey(file(viaAll), kind: .image, dim: 512, chunkOverlap: 200, settings: s)
        let b = PhotosContentSource(ref: PhotoLibrary.Ref(viaAlbum)!)
            .contentKey(file(viaAlbum), kind: .image, dim: 512, chunkOverlap: 200, settings: s)
        XCTAssertEqual(a, b, "the same asset in two albums must embed once")
    }

    /// A file key and a photo key must never collide, whatever the inputs.
    func testFileAndPhotoKeyspacesAreDisjoint() {
        let s = IndexSettings()
        let photoKey = PhotosContentSource(ref: PhotoLibrary.Ref(photoPath())!)
            .contentKey(file(photoPath()), kind: .image, dim: 512, chunkOverlap: 200, settings: s)
        XCTAssertEqual(photoKey?.hasPrefix("2|photo|"), true)

        // The file keyspace is "2|<kind>|...", never "2|photo|...", because FileKind has no
        // "photo" case - the kinds are text/image/audio/video/scan.
        XCTAssertFalse(FileKind.allCases.contains { $0.rawValue == "photo" },
                       "a FileKind named photo would collide with the Photos keyspace")
    }

    /// A missing file yields no key rather than a bogus one: no digest, no dedup, and the normal
    /// path decides what to do. A key invented here would alias unrelated files.
    func testFileContentKeyIsNilWhenTheFileCannotBeRead() {
        let missing = file("/nonexistent-\(UUID().uuidString)/nope.txt")
        XCTAssertNil(FileContentSource().contentKey(missing, kind: .text, dim: 512,
                                                    chunkOverlap: 200, settings: IndexSettings()))
    }

    /// Text keys must move with the chunking parameters: the same bytes chunked differently are
    /// different rows. This is the parameter the protocol has to thread through explicitly, so it
    /// is the one most likely to be dropped by a future source.
    func testFileTextKeyTracksChunkingParameters() throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("omni-cs-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("doc.txt")
        try String(repeating: "search index content. ", count: 500).write(to: url, atomically: true, encoding: .utf8)

        let f = CrawledFile(url: url, modified: 1_700_000_000, size: 11_000)
        let src = FileContentSource()
        var settings = IndexSettings()

        let base = src.contentKey(f, kind: .text, dim: 512, chunkOverlap: 200, settings: settings)
        XCTAssertNotNil(base)
        XCTAssertEqual(base, src.contentKey(f, kind: .text, dim: 512, chunkOverlap: 200, settings: settings),
                       "same file, same settings must be stable")
        XCTAssertNotEqual(base, src.contentKey(f, kind: .text, dim: 512, chunkOverlap: 120, settings: settings),
                          "overlap changes the chunks")
        settings.maxCharsPerChunk = 900
        XCTAssertNotEqual(base, src.contentKey(f, kind: .text, dim: 512, chunkOverlap: 200, settings: settings),
                          "chunk size changes the chunks")
    }

    // MARK: - Probe / threshold policy

    /// A dataless file must be refused BEFORE any content read: reading it would silently download
    /// it, which is the whole point of the setting.
    func testDatalessPolicyIsAnsweredByTheProbe() throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("omni-cs-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("plain.txt")
        try "ordinary local text".write(to: url, atomically: true, encoding: .utf8)

        // An ordinary local file is never dataless, under either policy.
        let f = CrawledFile(url: url, modified: 1_700_000_000, size: 19)
        var settings = IndexSettings()
        settings.skipDataless = true
        XCTAssertNotNil(FileContentSource().probe(f, settings: settings))
        settings.skipDataless = false
        XCTAssertNotNil(FileContentSource().probe(f, settings: settings))
    }

    /// `measured` is what stops a threshold from firing on a number nobody read. A text file has no
    /// dimensions, so it must report false - otherwise a minImageDimension setting would start
    /// silently dropping documents.
    func testTextProbeReportsNoMeasurements() throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("omni-cs-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("doc.txt")
        try "hello".write(to: url, atomically: true, encoding: .utf8)

        let p = try XCTUnwrap(FileContentSource().probe(CrawledFile(url: url, modified: 0, size: 5),
                                                        settings: IndexSettings()))
        XCTAssertEqual(p.kind, .text)
        XCTAssertFalse(p.measured)
        XCTAssertEqual(p.width, 0)
        XCTAssertEqual(p.height, 0)
    }

    /// An unreadable image header yields a probe with no measurements rather than nil: the item is
    /// still THERE, it just cannot be sized, and the decode below may still extract something.
    func testUnreadableImageHeaderIsNotTreatedAsMissing() throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("omni-cs-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("broken.png")
        try Data("not actually a png".utf8).write(to: url)

        let p = try XCTUnwrap(FileContentSource().probe(CrawledFile(url: url, modified: 0, size: 18),
                                                        settings: IndexSettings()))
        XCTAssertEqual(p.kind, .image)
        XCTAssertFalse(p.measured, "a header that did not parse must not claim measurements")
    }

    // MARK: - Metadata after decode

    /// Default policy: a row keeps the ORIGINAL's dimensions, because the decode is just a
    /// downscale and the stored size is a quality signal.
    func testDefaultMetaKeepsTheOriginalDimensions() {
        let probe = SourceProbe(kind: .image, width: 4032, height: 3024, measured: true)
        let decoded = CGContext(data: nil, width: 1568, height: 1176, bitsPerComponent: 8, bytesPerRow: 0,
                                space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!.makeImage()!
        let meta = FileContentSource().metaAfterImageDecode(probe: probe, decoded: decoded)
        XCTAssertEqual(meta.width, 4032)
        XCTAssertEqual(meta.height, 3024)
    }

    /// Photos keeps the asset's dimensions for a plain downscale (same aspect ratio)...
    func testPhotoMetaKeepsAssetDimensionsOnAPlainDownscale() {
        let src = PhotosContentSource(ref: PhotoLibrary.Ref(photoPath())!)
        let probe = SourceProbe(kind: .image, width: 4032, height: 3024, measured: true)
        let decoded = CGContext(data: nil, width: 1568, height: 1176, bitsPerComponent: 8, bytesPerRow: 0,
                                space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!.makeImage()!
        let meta = src.metaAfterImageDecode(probe: probe, decoded: decoded)
        XCTAssertEqual(meta.width, 4032, "a downscaled decode must not shrink the reported photo")
        XCTAssertEqual(meta.height, 3024)
    }

    /// ...but takes the decoded ones after an EDIT, where the asset's numbers no longer describe
    /// the picture the user sees.
    func testPhotoMetaFollowsTheDecodeAfterACrop() {
        let src = PhotosContentSource(ref: PhotoLibrary.Ref(photoPath())!)
        let probe = SourceProbe(kind: .image, width: 4032, height: 3024, measured: true)   // 4:3
        let cropped = CGContext(data: nil, width: 1000, height: 1000, bitsPerComponent: 8, bytesPerRow: 0,
                                space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!.makeImage()!
        let meta = src.metaAfterImageDecode(probe: probe, decoded: cropped)
        XCTAssertEqual(meta.width, 1000, "a crop changes the framing; the decode is the truth")
        XCTAssertEqual(meta.height, 1000)
    }

    /// An asset PhotoKit reported no dimensions for must not produce a zero-sized row when the
    /// decode did yield an image.
    func testPhotoMetaFallsBackToTheDecodeWhenTheAssetHasNoDimensions() {
        let src = PhotosContentSource(ref: PhotoLibrary.Ref(photoPath())!)
        let probe = SourceProbe(kind: .image, width: 0, height: 0, measured: true)
        let decoded = CGContext(data: nil, width: 800, height: 600, bitsPerComponent: 8, bytesPerRow: 0,
                                space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!.makeImage()!
        let meta = src.metaAfterImageDecode(probe: probe, decoded: decoded)
        XCTAssertEqual(meta.width, 800)
        XCTAssertEqual(meta.height, 600)
    }

    // MARK: - Modality coverage

    /// The Photos library holds images and videos. Audio must be declined rather than faked, so the
    /// shared decode falls through to "nothing to do" instead of embedding silence.
    func testPhotosDeclinesAudio() {
        let src = PhotosContentSource(ref: PhotoLibrary.Ref(photoPath())!)
        XCTAssertNil(src.audio(file(photoPath()), probe: SourceProbe(kind: .audio), settings: IndexSettings()))
    }

    /// Photos content() only answers for stills - a video goes through video(), and asking
    /// content() for one must not return an image payload.
    func testPhotosContentDeclinesNonImageKinds() {
        let src = PhotosContentSource(ref: PhotoLibrary.Ref(photoPath())!)
        if case .empty = src.content(file(photoPath()), kind: .video, settings: IndexSettings()) {} else {
            XCTFail("video must not be answered by content()")
        }
    }
}
