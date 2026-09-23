import XCTest
@testable import OmniKit

/// `PathTable.lowercasedExtension` has to answer exactly what `NSString.pathExtension.lowercased()`
/// answers - the filter menu's per-extension counts are built from it at every launch. It decides
/// only plain alphanumeric extensions itself and defers everything else, so these are the cases
/// where the two could part company.
final class PathExtensionTests: XCTestCase {
    func testMatchesNSString() {
        let paths = [
            "/a/b.txt", "/a/B.JPEG", "/a/b.tar.gz", "/a/noext", "/a/.bashrc", "/a/.hidden.TXT",
            "/a/b.", "/a/b..c", "/a/...", "/a/..", "/a/b. ", "/a/b.c d", "/a/ü.PnG", "/a/B.ÄÖ",
            "/a.dir/file", "/a.dir/file.md", "photos://ABC-123/IMG_1.HEIC", "/", "relative.Swift",
            "/a/b.123", "/a/b.x_y", "/a/b.x-y", "/a/b.🙂",
        ]
        var table = PathTable()
        for p in paths { table.append(p) }
        for (i, p) in paths.enumerated() {
            XCTAssertEqual(table.lowercasedExtension(i), (p as NSString).pathExtension.lowercased(), p)
        }
    }
}
