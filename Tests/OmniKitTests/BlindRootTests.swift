import XCTest
@testable import OmniKit

/// What may be swept when a root comes back empty. This decides DELETIONS, so each case is the
/// difference between losing rows and keeping ghosts.
final class BlindRootTests: XCTestCase {
    private let photos: Set<String> = ["photos://all"]

    func testEmptyFolderRootIsBlindAndKeepsItsRows() {
        // A folder gives no readability signal, so empty has to mean "probably unreadable".
        let blind = Indexer.blindRoots(totals: ["/Users/x/Docs": 0],
                                       photoRoots: photos, unreadablePhotos: [])
        XCTAssertEqual(blind, ["/Users/x/Docs"])
    }

    func testEmptyButREADABLEPhotoSourceIsNotBlind() {
        // The bug: a library the user emptied kept its rows forever. `enumerate` succeeded, so the
        // source is genuinely empty and the leftover rows are stale.
        let blind = Indexer.blindRoots(totals: ["photos://all": 0],
                                       photoRoots: photos, unreadablePhotos: [])
        XCTAssertTrue(blind.isEmpty, "an empty, readable Photos source must be swept")
    }

    func testUnreadablePhotoSourceStaysBlind() {
        // Access revoked or album deleted: keep the rows, exactly as for a folder.
        let blind = Indexer.blindRoots(totals: ["photos://all": 0],
                                       photoRoots: photos, unreadablePhotos: ["photos://all"])
        XCTAssertEqual(blind, ["photos://all"])
    }

    func testNonEmptyRootsAreNeverBlind() {
        let blind = Indexer.blindRoots(totals: ["/Users/x/Docs": 5, "photos://all": 3],
                                       photoRoots: photos, unreadablePhotos: ["photos://all"])
        XCTAssertTrue(blind.isEmpty, "a root that yielded files is readable by definition")
    }
}
