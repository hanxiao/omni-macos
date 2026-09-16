import XCTest
@testable import OmniKit

/// Issue #21: folders removed from the sidebar came back on every launch, so a user with 30-40
/// folders re-added them at every start. The whole bug was one condition - an empty stored list
/// read as "nothing stored" instead of "nothing wanted".
final class RootSeedTests: XCTestCase {
    /// THE REGRESSION GUARD, and the only test here that fails if the condition comes back.
    ///
    /// The obvious test - empty stored, empty legacy, expect empty - passes with the BUG in place,
    /// because falling through to the legacy keys yields empty too when they are also empty. It
    /// was written first, it could not fail, and it was deleted. The legacy keys have to be
    /// POPULATED for the difference to show, which is also the real case: everyone upgrading has
    /// the old defaults sitting in `omni.roots`, and they are exactly the users who hit this.
    func testAnEmptyStoredListBeatsTheLegacyKeys() {
        XCTAssertEqual(RootSeed.folders(stored: [],
                                        legacyRoots: ["/Users/x/Documents", "/Users/x/Downloads"],
                                        legacyCovered: ["/Users/x/Desktop"]), [])
    }

    /// A first launch adds nothing at all: no folders, and so no permission prompts before the
    /// user has chosen anything.
    func testAFirstLaunchSeedsNothing() {
        XCTAssertEqual(RootSeed.folders(stored: nil, legacyRoots: [], legacyCovered: []), [])
    }

    /// Upgrading from a build that predates `omni.addedFolders` keeps what that build indexed.
    func testAnUpgradeKeepsTheLegacyFolders() {
        XCTAssertEqual(RootSeed.folders(stored: nil,
                                        legacyRoots: ["/a", "/b"], legacyCovered: ["/c"]),
                       ["/a", "/b", "/c"])
    }

    /// The ordinary case: what was stored is what comes back, in order.
    func testStoredFoldersAreReturnedInOrder() {
        XCTAssertEqual(RootSeed.folders(stored: ["/z", "/a"], legacyRoots: ["/ignored"], legacyCovered: []),
                       ["/z", "/a"])
    }
}
