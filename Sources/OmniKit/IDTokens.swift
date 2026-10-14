import Foundation

/// Identifier-shaped tokens: order numbers, booking codes, ticket IDs, serials, long numbers -
/// what a person types to find one exact file and what an embedding cannot hold ("INV-2041",
/// "K7Q2LM", "OPS-0417", "4815162342"). Readings, dates, sizes and decimals are not identifiers:
/// "21.4C", "101.3kPa", "2026-03-01", "1280x720", "4e-06".
///
/// One pass over the UTF-8 bytes, no regex: this runs over every text the indexer reads. A token
/// is a run of ASCII letters, digits, '-', '_' and '#'; anything else ends it - so a path or a
/// fraction is its parts, and an identifier inside CJK or Arabic text is still found.
public enum IDTokens {
    /// Tokens of `text`, lowercased, in order (repeats included).
    public static func tokens(in text: String) -> [String] {
        var out: [String] = []
        var cur: [UInt8] = []
        cur.reserveCapacity(48)
        func flush() {
            var lo = 0, hi = cur.count
            while lo < hi, isJoiner(cur[lo]) { lo += 1 }
            while hi > lo, isJoiner(cur[hi - 1]) { hi -= 1 }
            if hi - lo >= 4, hi - lo <= 40 {
                let t = cur[lo ..< hi]
                if isIdentifier(bytes: t) { out.append(String(decoding: t.map(lower), as: UTF8.self)) }
            }
            cur.removeAll(keepingCapacity: true)
        }
        for b in text.utf8 {
            if isTokenByte(b) {
                if cur.count < 64 { cur.append(b) }
            } else if !cur.isEmpty { flush() }
        }
        if !cur.isEmpty { flush() }
        return out
    }

    public static func isIdentifier<S: StringProtocol>(_ t: S) -> Bool {
        let b = Array(t.utf8)
        guard b.allSatisfy(isTokenByte) else { return false }
        return isIdentifier(bytes: b[...])
    }

    @inline(__always) static func isDigit(_ c: UInt8) -> Bool { c >= 48 && c <= 57 }
    @inline(__always) static func isLetter(_ c: UInt8) -> Bool { (c >= 65 && c <= 90) || (c >= 97 && c <= 122) }
    @inline(__always) static func isJoiner(_ c: UInt8) -> Bool { c == 45 || c == 95 || c == 35 }   // - _ #
    @inline(__always) static func isTokenByte(_ c: UInt8) -> Bool { isDigit(c) || isLetter(c) || isJoiner(c) }
    @inline(__always) static func lower(_ c: UInt8) -> UInt8 { c >= 65 && c <= 90 ? c + 32 : c }

    /// 4 to 40 bytes with a digit, and a letter too (not a reading, a date, a size or a number in
    /// scientific notation), or seven digits (not a year, a range or a short number).
    static func isIdentifier(bytes t: ArraySlice<UInt8>) -> Bool {
        let n = t.count
        guard n >= 4, n <= 40 else { return false }
        var digits = 0, letters = 0
        for c in t { if isDigit(c) { digits += 1 } else if isLetter(c) { letters += 1 } }
        guard digits > 0 else { return false }
        let a = Array(t)
        var i = 0
        while i < n, isDigit(a[i]) { i += 1 }
        let lead = i
        // A date, with or without an hour after T ("2026-03-01", "2026-03-01T00").
        if n >= 10, lead == 4, a[4] == 45, isDigit(a[5]), isDigit(a[6]), a[7] == 45, isDigit(a[8]), isDigit(a[9]) {
            if n == 10 { return false }
            if a[10] == 116 || a[10] == 84, a[11...].allSatisfy(isDigit) { return false }
        }
        guard letters > 0 else { return digits >= 7 }
        // A reading: digits, then one to four letters and nothing else ("3kPa", "500ml", "16kWp").
        if lead > 0, n - lead <= 4, a[lead...].allSatisfy(isLetter) { return false }
        // A size or scientific notation: digits, x or e, an optional sign, digits ("1280x720", "28e-06").
        if lead > 0, lead < n, a[lead] == 120 || a[lead] == 88 || a[lead] == 101 || a[lead] == 69 {
            var j = lead + 1
            if j < n, a[j] == 45 || a[j] == 43 { j += 1 }
            if j < n, a[j...].allSatisfy(isDigit) { return false }
        }
        return true
    }
}
