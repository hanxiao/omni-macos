import Foundation

/// Text that is machine payload rather than language: base64 blobs, tokens, signatures, digests.
///
/// These are rare in a corpus and ruinous in a result list. Measured on a 9.7M-chunk index, chunks
/// that are at least half unbroken token characters are 1.8% of the agent-log chunks they mostly
/// come from, yet they took 14% of the top-10 slots across a query sweep - a 14x overrepresentation.
/// The reason is that an embedding of random characters is near-random, and a near-random unit
/// vector has no particular reason to be far from any given query, so it drifts into the top of
/// whatever is asked. A thinkingSignature blob cannot answer a question, and it costs the same
/// tokens to return as a real passage.
///
/// The test is on the RUN, not the character class: a path, a UUID, a hyphenated identifier all
/// break into segments well under the run floor and are never touched. '/' is deliberately not a
/// run character for exactly that reason - counting it would make one long path a single run.
public enum OpaqueText {
    /// Shortest unbroken token that counts as payload. A UUID is 36 characters and stays; the
    /// base64url blobs that prompted this are 90+.
    public static let minRun = 40
    /// A chunk at or above this fraction of payload characters carries nothing worth retrieving.
    public static let dropFraction = 0.5

    /// Characters that continue a payload run. Base64 and base64url, hex, and the '=' padding.
    @inline(__always) private static func isRunChar(_ u: UInt8) -> Bool {
        (u >= 65 && u <= 90) || (u >= 97 && u <= 122) || (u >= 48 && u <= 57)
            || u == 43 || u == 61 || u == 95 || u == 45   // + = _ -
    }

    /// Fraction of `text` that sits inside an unbroken run of at least `minRun` token characters.
    /// Counted over UTF-8 bytes: the run alphabet is ASCII, so any multi-byte scalar is a non-run
    /// byte and correctly breaks a run.
    public static func payloadFraction(_ text: String) -> Double {
        var total = 0, payload = 0, run = 0
        for byte in text.utf8 {
            total += 1
            if isRunChar(byte) {
                run += 1
            } else {
                if run >= minRun { payload += run }
                run = 0
            }
        }
        if run >= minRun { payload += run }
        guard total > 0 else { return 0 }
        return Double(payload) / Double(total)
    }

    /// Whether this chunk should be left out of the index.
    public static func isPayload(_ text: String) -> Bool {
        payloadFraction(text) >= dropFraction
    }

    /// Drop the payload chunks from a chunked file. A file that is payload end to end keeps its
    /// first chunk: its row still has to exist for the filename channel to find it, and one junk
    /// vector for such a file is the price of not making "indexed" and "has chunks" disagree.
    public static func filter<T>(_ pieces: [T], text: (T) -> String) -> [T] {
        guard pieces.count > 1 else { return pieces }
        let kept = pieces.filter { !isPayload(text($0)) }
        return kept.isEmpty ? [pieces[0]] : kept
    }
}
