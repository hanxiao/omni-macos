import XCTest
@testable import OmniKit

/// The tagger's partial top-k selection must return EXACTLY what the deterministic full sort
/// returns (descending value, ties by lower index) - it replaced a full 25k-index sort purely
/// for speed, so any divergence is a correctness bug, not a tuning difference.
final class TaggerSelectionTests: XCTestCase {

    /// Deterministic xorshift so failures reproduce.
    private struct Rng {
        var s: UInt64
        mutating func next() -> UInt64 { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s }
        mutating func float() -> Float { Float(next() % 100_000) / 1000 - 50 }
    }

    private func reference(_ x: [Float], k: Int) -> [Int] {
        Array(x.indices.sorted { x[$0] != x[$1] ? x[$0] > x[$1] : $0 < $1 }.prefix(k))
    }

    func testMatchesFullSortOnRandomVectors() {
        var rng = Rng(s: 0x9E3779B97F4A7C15)
        for (n, k) in [(5, 3), (399, 400), (400, 400), (401, 400), (1000, 400), (25465, 400)] {
            let x = (0 ..< n).map { _ in rng.float() }
            XCTAssertEqual(OmniTagger.topIndices(x, k: k), reference(x, k: k), "n=\(n) k=\(k)")
        }
    }

    func testMatchesFullSortWithHeavyTies() {
        var rng = Rng(s: 42)
        for trial in 0 ..< 50 {
            // Quantize hard so the k-th boundary is dense with exact ties - the case where a
            // sloppy selection silently picks a different (still "correct-looking") subset.
            let n = 2000 + Int(rng.next() % 3000)
            let x = (0 ..< n).map { _ in Float(rng.next() % 8) }
            XCTAssertEqual(OmniTagger.topIndices(x, k: 400), reference(x, k: 400), "trial=\(trial)")
        }
    }

    func testEdgeCases() {
        XCTAssertEqual(OmniTagger.topIndices([], k: 400), [])
        XCTAssertEqual(OmniTagger.topIndices([1.5], k: 0), [])
        XCTAssertEqual(OmniTagger.topIndices([3, 1, 2], k: 5), [0, 2, 1])
        XCTAssertEqual(OmniTagger.topIndices([Float](repeating: 7, count: 500), k: 400),
                       Array(0 ..< 400))   // all-equal: lowest indices win, in order
    }
}
