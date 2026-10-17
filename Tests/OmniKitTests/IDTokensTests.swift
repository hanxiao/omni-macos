import XCTest
@testable import OmniKit

/// IDTokens.tokens reads the UTF-8 in place. It must return exactly what the byte-buffer form it
/// replaced returned, for every input: that form is kept here as the reference.
final class IDTokensTests: XCTestCase {
    private func reference(_ text: String) -> [String] {
        var out: [String] = []
        var cur: [UInt8] = []
        func flush() {
            var lo = 0, hi = cur.count
            while lo < hi, IDTokens.isJoiner(cur[lo]) { lo += 1 }
            while hi > lo, IDTokens.isJoiner(cur[hi - 1]) { hi -= 1 }
            if hi - lo >= 4, hi - lo <= 40 {
                let t = cur[lo ..< hi]
                if IDTokens.isIdentifier(bytes: t) { out.append(String(decoding: t.map(IDTokens.lower), as: UTF8.self)) }
            }
            cur.removeAll(keepingCapacity: true)
        }
        for b in text.utf8 {
            if IDTokens.isTokenByte(b) {
                if cur.count < 64 { cur.append(b) }
            } else if !cur.isEmpty { flush() }
        }
        if !cur.isEmpty { flush() }
        return out
    }

    /// The 64-byte cap only decides a token after many joiners are trimmed, which random text
    /// hardly ever reaches: walk an identifier across the cap one byte at a time.
    func testTheCapBoundary() {
        for lead in 0 ... 70 {
            for body in ["A1b2C3d4", "a1b2c", "1234567", "x9", "Q7-Z9-K2"] {
                for tail in ["", "-", "__##", "ZZ99", " next 4815162342"] {
                    let s = String(repeating: "-", count: lead) + body + tail
                    XCTAssertEqual(IDTokens.tokens(in: s), reference(s), s)
                    let t = "Order " + String(repeating: "#", count: lead) + body + tail + ", thanks"
                    XCTAssertEqual(IDTokens.tokens(in: t), reference(t), t)
                }
            }
        }
    }

    func testMatchesTheByteBufferForm() {
        let pieces = ["INV-2041", "K7Q2LM", "ops-0417", "4815162342", "2026-03-01", "500ml", "1280x720", "4e-06",
                      "abc", "12", "hello", "the", "#", "--", "__", "-#_", "x1", "A1B2C3D4E5", "订单", "번호", "😀",
                      "é", "\u{0301}", "\n", " ", ",", ".", "/", ":", "\t", "-", "_", "#"]
        var rng = SystemRandomNumberGenerator()
        for _ in 0 ..< 4_000 {
            var s = ""
            for _ in 0 ..< Int.random(in: 0 ... 40, using: &rng) {
                if Int.random(in: 0 ..< 12, using: &rng) == 0 {
                    // A long run: past the 64-byte cap, with joiners at either end or in the middle.
                    let alphabet = Array("abcXYZ0123456789-_#")
                    s += String((0 ..< Int.random(in: 30 ... 120, using: &rng)).map { _ in alphabet.randomElement(using: &rng)! })
                } else {
                    s += pieces.randomElement(using: &rng)!
                }
            }
            XCTAssertEqual(IDTokens.tokens(in: s), reference(s), s)
            // A bridged string takes the other branch of withUTF8.
            let bridged = NSString(string: s) as String
            XCTAssertEqual(IDTokens.tokens(in: bridged), reference(s), s)
        }
    }
}
