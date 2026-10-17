import Foundation

/// Identifier-shaped tokens: order numbers, booking codes, ticket IDs, serials, long numbers -
/// what a person types to find one exact file and what an embedding cannot hold ("INV-2041",
/// "K7Q2LM", "OPS-0417", "4815162342", "2026-03-01", "500ml").
///
/// One pass over the UTF-8 bytes, no regex: this runs over every text the indexer reads. A token
/// is a run of ASCII letters, digits, '-', '_' and '#'; anything else ends it - so a path or a
/// fraction is its parts, and an identifier inside CJK or Arabic text is still found.
public enum IDTokens {
    /// Tokens of `text`, lowercased, in order (repeats included). A run of token bytes is judged
    /// on its first 64 bytes, its joiners trimmed from both ends.
    ///
    /// IN PLACE over the contiguous UTF-8 (`withUTF8`, which makes a bridged string native once):
    /// it runs over every chunk of a file each time the file is saved. Appending each byte to a
    /// buffer and walking `text.utf8` cost 0.7 ms an edit in the save path (40 edits a round,
    /// 0.15.15 against 0.17.1, roadmap.md "Save path"); this reads the bytes where they are and
    /// allocates only for a token it returns. IDTokensTests checks it against the byte-buffer form.
    public static func tokens(in text: String) -> [String] {
        var text = text
        return text.withUTF8 { buf -> [String] in
            var out: [String] = []
            let n = buf.count
            var i = 0
            while i < n {
                guard isTokenByte(buf[i]) else { i += 1; continue }
                let start = i
                while i < n, isTokenByte(buf[i]) { i += 1 }
                var lo = start, hi = Swift.min(i, start + 64)
                while lo < hi, isJoiner(buf[lo]) { lo += 1 }
                while hi > lo, isJoiner(buf[hi - 1]) { hi -= 1 }
                let len = hi - lo
                guard len >= 4, len <= 40 else { continue }
                var digits = 0, letters = 0
                for k in lo ..< hi { let c = buf[k]; if isDigit(c) { digits += 1 } else if isLetter(c) { letters += 1 } }
                guard digits > 0, letters > 0 || digits >= 7 else { continue }
                out.append(String(unsafeUninitializedCapacity: len) { dst in
                    for k in 0 ..< len { dst[k] = lower(buf[lo + k]) }
                    return len
                })
            }
            return out
        }
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
    /// 4 to 40 bytes with a digit, and a letter too or seven digits (not a year, a range or a short
    /// number). NO EXCEPTIONS BY SHAPE (Han, 2026-10-14: no hand-written rules): dates, readings,
    /// sizes and scientific notation used to be cut out ("2026-03-01", "500ml", "1280x720", "4e-06")
    /// by a parser of their forms. Counted as identifiers they cost 17% more postings on real
    /// Desktop and Downloads text (348,851 -> 408,094 file-token pairs, ~0.7% -> 0.8% of the text)
    /// and move no query on the eval (exact 1.00, MRR 0.843, both), and a search for a date or a
    /// reading now finds the files that hold it exactly.
    static func isIdentifier(bytes t: ArraySlice<UInt8>) -> Bool {
        let n = t.count
        guard n >= 4, n <= 40 else { return false }
        var digits = 0, letters = 0
        for c in t { if isDigit(c) { digits += 1 } else if isLetter(c) { letters += 1 } }
        return digits > 0 && (letters > 0 || digits >= 7)
    }
}
