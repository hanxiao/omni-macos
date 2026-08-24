import XCTest
@testable import OmniKit

/// The Photos path format is load-bearing: it is the store's primary key, the thing the stale
/// sweep matches roots against, and the only record of which asset a row came from. These hold it
/// to a round trip.
final class PhotoLibraryPathTests: XCTestCase {

    private func path(source: String, asset: String, name: String) -> String {
        PhotoLibrary.scheme + PhotoLibrary.esc(source) + "/" + PhotoLibrary.esc(asset) + "/" + name
    }

    func testRefRoundTripsARealLocalIdentifier() {
        // PHAsset.localIdentifier is "<uuid>/L0/001" - it contains the separator, which is the
        // whole reason the component is escaped.
        let id = "24EE2DB1-1AA4-43FF-908B-2A6FC1A4BCE9/L0/001"
        let p = path(source: "all", asset: id, name: "IMG_1234.HEIC")
        XCTAssertFalse(p.dropFirst(PhotoLibrary.scheme.count).contains("/L0/"), "identifier must not split the path")

        let ref = try! XCTUnwrap(PhotoLibrary.Ref(p))
        XCTAssertEqual(ref.sourceID, "all")
        XCTAssertEqual(ref.localIdentifier, id)
        XCTAssertEqual(ref.filename, "IMG_1234.HEIC")
    }

    func testRefRoundTripsAnAlbumSourceWhoseIDAlsoContainsSlashes() {
        let album = "5E2F5C3A-0000-4000-8000-000000000001/L0/040"
        let asset = "AAAA1111-2222-3333-4444-555566667777/L0/001"
        let source = PhotoLibrary.Source(id: album, title: "Iceland")
        let p = source.key + "/" + PhotoLibrary.esc(asset) + "/IMG_9.JPG"

        let ref = try! XCTUnwrap(PhotoLibrary.Ref(p))
        XCTAssertEqual(ref.localIdentifier, asset)
        XCTAssertEqual(PhotoLibrary.sourceKey(ofPath: p), source.key)
        // Containment - what rootOf() and deleteUnderFolder() both rely on.
        XCTAssertTrue(p.hasPrefix(source.key + "/"))
    }

    func testPercentInAnIdentifierSurvives() {
        let odd = "a%b/L0/001"
        let p = path(source: "all", asset: odd, name: "x.jpg")
        XCTAssertEqual(PhotoLibrary.Ref(p)?.localIdentifier, odd)
    }

    func testOrdinaryFilePathsAreNotPhotoPaths() {
        for p in ["/Users/x/Pictures/IMG_1.HEIC", "/tmp/a.jpg", "", "photos:/missing", "photos://all/only-two"] {
            XCTAssertNil(PhotoLibrary.Ref(p), "\(p) must not parse as a Photos ref")
        }
        XCTAssertFalse(PhotoLibrary.isPhotoPath("/Users/x/photos://a"))
    }

    /// The store splits a path into (dir, name) and rebuilds it on the way out. A key that does not
    /// survive that would land rows under a directory row nothing owns.
    func testStorePathSplitRoundTrips() {
        let p = path(source: "all", asset: "24EE2DB1-1AA4-43FF-908B-2A6FC1A4BCE9/L0/001", name: "IMG_1234.HEIC")
        let (dir, name) = StoreSchema.splitPath(p)
        XCTAssertEqual(name, "IMG_1234.HEIC")
        XCTAssertEqual(StoreSchema.joinPath(dir: dir, name: name), p)
    }

    func testCrawledFileReadsNameAndExtensionOffAPhotoPath() {
        let p = path(source: "all", asset: "AAAA/L0/001", name: "IMG_1234.HEIC")
        let f = CrawledFile(path: p, modified: 1, size: 2)
        XCTAssertTrue(f.isPhoto)
        XCTAssertEqual(f.name, "IMG_1234.HEIC")
        XCTAssertEqual(f.ext, "HEIC")
        // The extension is what decides which pipeline a Photos asset goes down.
        XCTAssertEqual(FileExtractor.kind(forExtension: f.ext), .image)
    }

    func testSourceKeysAreDistinctAndNotPrefixesOfEachOther() {
        let a = PhotoLibrary.Source.all
        let b = PhotoLibrary.Source(id: "5E2F5C3A-0000-4000-8000-000000000001/L0/040", title: "Iceland")
        XCTAssertNotEqual(a.key, b.key)
        XCTAssertFalse(b.key.hasPrefix(a.key + "/"))
        XCTAssertFalse(a.key.hasPrefix(b.key + "/"))
    }
}
