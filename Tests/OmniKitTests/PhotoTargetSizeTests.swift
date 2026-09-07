import XCTest
import CoreGraphics
@testable import OmniKit

/// The Photos decode has to reduce exactly like the file decode does. `FileExtractor.loadImage`
/// passes `kCGImageSourceThumbnailMaxPixelSize`, which is a CAP: a smaller image comes back at its
/// own size and is never enlarged. PhotoKit's `.exact` resize has no such property - it delivers
/// the size it is asked for - so the target has to be clamped to the asset before the request.
///
/// These pin the pure part of that (no photo library, no authorization needed).
final class PhotoTargetSizeTests: XCTestCase {

    /// A big asset is capped at the setting, exactly like a big file.
    func testLargeAssetIsCappedAtMaxDimension() {
        XCTAssertEqual(PhotoLibrary.targetSide(maxDimension: 1568, pixelWidth: 4032, pixelHeight: 3024), 1568)
        XCTAssertEqual(PhotoLibrary.targetSide(maxDimension: 1568, pixelWidth: 3024, pixelHeight: 4032), 1568)
    }

    /// THE REGRESSION: a small asset must keep its own size. Asking `.exact` for 1568 here would
    /// hand back an upscaled image - more vision tokens carrying no more information, and a
    /// different embedding than the same picture would get as a file on disk.
    func testSmallAssetIsNotUpscaled() {
        XCTAssertEqual(PhotoLibrary.targetSide(maxDimension: 1568, pixelWidth: 640, pixelHeight: 480), 640)
        XCTAssertEqual(PhotoLibrary.targetSide(maxDimension: 1568, pixelWidth: 100, pixelHeight: 900), 900)
    }

    /// The long edge decides, for either orientation - the same thing `contentMode: .aspectFit`
    /// with a square target means.
    func testLongEdgeDecides() {
        XCTAssertEqual(PhotoLibrary.targetSide(maxDimension: 1000, pixelWidth: 1200, pixelHeight: 300), 1000)
        XCTAssertEqual(PhotoLibrary.targetSide(maxDimension: 1000, pixelWidth: 300, pixelHeight: 1200), 1000)
        XCTAssertEqual(PhotoLibrary.targetSide(maxDimension: 2000, pixelWidth: 1200, pixelHeight: 300), 1200)
    }

    /// An asset that reports no dimensions (PhotoKit occasionally has none for a placeholder) must
    /// not collapse the request to zero; fall back to the cap and let PhotoKit answer.
    func testMissingAssetDimensionsFallBackToTheCap() {
        XCTAssertEqual(PhotoLibrary.targetSide(maxDimension: 1568, pixelWidth: 0, pixelHeight: 0), 1568)
    }

    /// The 64 px floor of the old code is preserved, including for a nonsense setting.
    func testFloorHolds() {
        XCTAssertEqual(PhotoLibrary.targetSide(maxDimension: 0, pixelWidth: 4032, pixelHeight: 3024), 64)
        XCTAssertEqual(PhotoLibrary.targetSide(maxDimension: -5, pixelWidth: 4032, pixelHeight: 3024), 64)
    }

    /// A tiny asset under the floor still asks for its own size, not the floor: the clamp is
    /// min(cap, longest), and 32 < 64.
    func testTinyAssetAsksForItsOwnSize() {
        XCTAssertEqual(PhotoLibrary.targetSide(maxDimension: 1568, pixelWidth: 32, pixelHeight: 32), 32)
    }
}
