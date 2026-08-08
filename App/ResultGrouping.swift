import Foundation
import Accelerate
import OmniKit

/// A search result after duplicate collapsing: one file the user sees, plus the copies it stands
/// for. `members` always contains `representative` as its first element, so a group of one is the
/// ordinary case and needs no special handling anywhere downstream.
struct ResultGroup: Identifiable {
    var members: [SearchHit]
    /// How the members were established - decides the wording ("2 copies" vs "2 near-identical").
    var reason: Reason

    enum Reason { case single, exact, near }

    var representative: SearchHit { members[0] }
    var count: Int { members.count }
    var isStack: Bool { members.count > 1 }
    var id: String { representative.path }
    /// Every path in the group, for actions that deliberately span the stack.
    var paths: [String] { members.map(\.path) }
}

/// Collapses near-identical and byte-identical results into stacks, POST-HOC: it runs on the hits
/// the store already returned and changes nothing about how they were found or scored.
///
/// Two tiers, because they carry different risk:
///
///   exact - same `content_key` AND same byte size. The key is a digest of the bytes the extractor
///           reads, which for text is capped, so equal keys alone do NOT mean equal files; the size
///           is what makes the claim honest. (Indexer stamps size into the key from v2 on, so this
///           is belt and braces for rows written by older builds.)
///
///   near  - cosine >= `threshold` on mean-pooled file vectors, same kind, sizes within
///           `sizeTolerance`. The size guard is load-bearing, not cosmetic: a document and its
///           later draft sit at 0.98+ by construction, so without it seven append-only session
///           checkpoints (4 MB -> 18 MB) collapse into one stack and six real files vanish.
///
/// Clustering is greedy from the top of the ranking: each unclaimed hit becomes an anchor and takes
/// everything that matches IT. Not connected components - those chain (a~b, b~c, a≁c) and merge
/// clusters that share no member with each other. Anchor-first also means the highest-scoring file
/// is always the one shown, which is what the user ranked into that slot.
enum ResultGrouping {
    /// Cosine at or above this is "the same file, near enough". Chosen by sweeping 0.97/0.98/0.99
    /// over real queries on a 212k-file index: 0.97 merged distinct figures, 0.99 missed
    /// punctuation-renamed copies of the same asset.
    static let threshold: Float = 0.98
    /// Sizes must be within 10% of each other. Drafts and checkpoints fail this; re-encodes pass.
    static let sizeTolerance: Double = 0.10

    /// - Parameters:
    ///   - hits: ranked results, best first.
    ///   - vectors: pooled unit vectors by path; a missing path never groups (cannot compare).
    ///   - contentKeys: path -> (key, modified) from the store; used only when `modified` agrees.
    ///   - nearEnabled: false collapses only byte-identical copies.
    static func group(hits: [SearchHit],
                      vectors: [String: [Float]],
                      contentKeys: [String: (key: String, modified: Double)],
                      nearEnabled: Bool) -> [ResultGroup] {
        guard hits.count > 1 else { return hits.map { ResultGroup(members: [$0], reason: .single) } }

        // Exact identity per hit as a (key, size) PAIR, not an interpolated string: this runs on
        // every keystroke, and building n Strings only to compare them allocates and hashes for
        // nothing. Size is part of the identity because the key digests only the bytes the
        // extractor read (capped for text), so equal keys alone do not mean equal files.
        let keyed: [(key: String, size: Int)?] = hits.map { h in
            guard let ck = contentKeys[h.path], ck.modified == h.modified else { return nil }
            return (ck.key, h.size)
        }

        let sim = nearEnabled ? similarityMatrix(hits: hits, vectors: vectors) : nil
        let n = hits.count
        var claimed = [Bool](repeating: false, count: n)
        var groups: [ResultGroup] = []
        groups.reserveCapacity(n)

        for i in 0 ..< n where !claimed[i] {
            claimed[i] = true
            var members = [hits[i]]
            var reason = ResultGroup.Reason.single
            for j in (i + 1) ..< n where !claimed[j] {
                if let k = keyed[i], let kj = keyed[j], k.size == kj.size, k.key == kj.key {
                    claimed[j] = true; members.append(hits[j])
                    if reason == .single { reason = .exact }
                    continue
                }
                guard nearEnabled, let sim,
                      hits[i].kind == hits[j].kind,
                      // Same EXTENSION, not merely the same kind. Both are "text", but a .jsonl
                      // session log that quotes an .html page scores 0.98+ against it while being
                      // a completely different artifact - observed live, three real .html files
                      // hidden behind a log that merely contained them. Costs the .jpg/.png pair
                      // of one image, which is the cheaper mistake.
                      ext(hits[i].path) == ext(hits[j].path),
                      sizesComparable(hits[i].size, hits[j].size),
                      sim[i * n + j] >= threshold else { continue }
                claimed[j] = true; members.append(hits[j])
                reason = .near        // a stack with any near member is described as near
            }
            groups.append(ResultGroup(members: members, reason: members.count > 1 ? reason : .single))
        }
        return groups
    }

    private static func ext(_ path: String) -> String { (path as NSString).pathExtension.lowercased() }

    /// Sizes within tolerance of each other. Unknown size (0, rows predating the column) is treated
    /// as comparable - the cosine and kind guards still apply, and refusing to group everything
    /// older than the column would be worse than the occasional missed guard.
    private static func sizesComparable(_ a: Int, _ b: Int) -> Bool {
        guard a > 0, b > 0 else { return true }
        let lo = Double(min(a, b)), hi = Double(max(a, b))
        return lo / hi >= 1 - sizeTolerance
    }

    /// Full n x n cosine matrix in one GEMM. The vectors are already unit length, so the inner
    /// product IS the cosine.
    ///
    /// On CPU via Accelerate rather than the GPU: at n <= 120 this is ~11 MFLOP, tens of
    /// microseconds, while an MLX round trip would add dispatch plus an eval sync AND queue behind
    /// whatever the embedder is doing - and this runs on every keystroke under instant search.
    /// Rows without a pooled vector get a zero row, so they score 0 against everything and group
    /// with nothing.
    private static func similarityMatrix(hits: [SearchHit], vectors: [String: [Float]]) -> [Float]? {
        guard let dim = vectors.values.first?.count, dim > 0 else { return nil }
        let n = hits.count
        var m = [Float](repeating: 0, count: n * dim)
        var any = false
        for (i, h) in hits.enumerated() {
            guard let v = vectors[h.path], v.count == dim else { continue }
            any = true
            m.replaceSubrange(i * dim ..< (i * dim + dim), with: v)
        }
        guard any else { return nil }
        // SYRK, not GEMM: V·Vt is symmetric, so half the multiply-adds are redundant. Only the
        // upper triangle is written, which is exactly the half the clustering loop reads (j > i).
        var out = [Float](repeating: 0, count: n * n)
        m.withUnsafeBufferPointer { a in
            out.withUnsafeMutableBufferPointer { c in
                cblas_ssyrk(CblasRowMajor, CblasUpper, CblasNoTrans,
                            Int32(n), Int32(dim),
                            1, a.baseAddress, Int32(dim),
                            0, c.baseAddress, Int32(n))
            }
        }
        return out
    }
}
