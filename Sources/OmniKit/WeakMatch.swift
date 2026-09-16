import Foundation

/// When a search found nothing that really matches, and how to say so.
///
/// Dense retrieval has no empty result. It returns its nearest neighbours whatever was asked, so
/// "there is nothing here" and "here are forty things" are the same page, and the failure worth
/// naming is not an empty list - it is a full one that means nothing. This is the one place that
/// judgement is made, so the window and the HTTP/MCP surface agree to the character: one computes
/// it on the main actor, the other on a request thread.
public enum WeakMatch {
    /// Standard deviations above the most confusable impostors (see
    /// `VectorStore.retrievalConfidence`), below which the top hit is not clearly better than what
    /// this query finds anywhere in the index.
    ///
    /// Measured, not chosen. On a frozen 2.68M-file index, 600 probes with a known answer against
    /// 600 with the answer removed but the domain kept:
    ///
    ///     slice     answerable   wrongly warned   near-negatives caught
    ///     text         493           0.8%               20.9%
    ///     media        107           1.9%                8.4%
    ///     latin        411           1.5%               11.9%
    ///     cjk           89           0.0%               48.3%
    ///     cjk-name      68           0.0%               25.0%
    ///
    /// Raising it catches more and costs real answers fast: at 4.7 it is 41.0% caught for 3.7%
    /// wrongly warned. The asymmetry is the point - warning on a good search is the expensive
    /// error, because it teaches the user to ignore the warning.
    /// Overridable with `defaults write io.hanxiao.omni omni.weakMatchThreshold <float>`, which is
    /// how the plumbing gets a POSITIVE CONTROL: an advisory that never appears is indistinguishable
    /// from one that is never computed, and at the shipped value it correctly stays quiet on most
    /// queries. Raise it and every search should warn; if none does, the wiring is broken.
    public static var threshold: Float {
        let v = UserDefaults.standard.object(forKey: "omni.weakMatchThreshold") as? Double
        return v.map(Float.init) ?? 2.7
    }

    /// The advisory line, or nil for "confident" AND for "no opinion".
    ///
    /// Those last two must produce the same silence. An unavailable statistic is the small or
    /// still-filling index - where every index starts - and a caller that rendered "no opinion" as
    /// "no match" would tell a new user their files are missing while they are being indexed.
    public static func notice(_ c: VectorStore.RetrievalConfidence?, hits: [SearchHit]) -> String? {
        guard let c, c.available, !hits.isEmpty, c.tnorm < threshold else { return nil }
        return "Nothing here matches this closely. These are the nearest files, not answers."
    }
}
