import Foundation

// The synthetic vectors and query sets the store cases run against.
//
// The construction is NOT new. It is the one omni-verify's `gateparity` mode already uses
// (main.swift:3442-3449): a per-row xorshift64 stream seeded by an LCG step of the row index,
// components uniform in [-1, 1], L2-normalised. It is reproduced here rather than improved on,
// because the paper's Sec. 3.2 and Table 3 numbers were measured against exactly this distribution
// and a "better" one (clustered, as in `searchbench`) would change scan and prune behaviour without
// changing anything the paper claims. A benchmark whose inputs drifted is a benchmark whose old
// numbers cannot be compared with its new ones.
//
// Three properties are load-bearing:
//
//  1. `vec(i)` is a PURE function of `i`. No shared stream, no consumption order, so a 250,000-row
//     store built across eight cores contains the same bytes as one built serially, and a case that
//     rebuilds a store after an arm switch rebuilds the same rows.
//  2. Queries live in a row-index space that is disjoint from every ladder rung
//     (`queryIndexBase = 2^32`, and the largest rung is 4M). `gateparity` used `vec(1_000_000 + q)`,
//     which was off-corpus at its 240,000 rows but would BE a stored row at the 2M rung - and a
//     query that is exactly a stored vector scores a perfect 1.0, which changes what the can't-win
//     gate can prune. Same distribution, disjoint indices.
//  3. Building is bounded and interruptible. Rows are generated and inserted in 8,192-row batches,
//     so the host transient is ~25 MB regardless of the rung, cancel is acknowledged within one
//     batch, and a deadline stops the build with a row count rather than with a fabricated store.

public enum PaperVectors {
    /// The paper's embedding width. Not a default to be overridden casually: every arithmetic peak
    /// in the catalog is computed at 768, and Table 3's rows are keyed on it.
    public static let dim = 768

    /// Query indices start here. Above every ladder rung the suite can reach (p09's 4M is the
    /// largest) and far below the point where the LCG step wraps into anything interesting.
    public static let queryIndexBase = 1 << 32

    /// Cancel and deadline granularity, and the size of the host transient during a build.
    /// 8,192 rows at dim 768 is 25 MB of Float32 before it becomes 12.6 MB of bf16 in the store.
    public static let rowsPerBatch = 8192

    // MARK: - Vectors

    /// Row `i`'s unit vector. The `gateparity` recipe, verbatim: an LCG step of the index seeds a
    /// xorshift64 stream, each component is a signed 32-bit draw scaled into [-1, 1], and the
    /// result is L2-normalised because the store's cosine search assumes unit rows.
    public static func vec(_ i: Int, dim: Int = dim) -> [Float] {
        var state = UInt64(bitPattern: Int64(i &* 6364136223846793005 &+ 1442695040888963407)) | 1
        var v = [Float](repeating: 0, count: dim)
        var norm: Float = 0
        for k in 0 ..< dim {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17
            let x = Float(Int32(truncatingIfNeeded: state)) / Float(Int32.max)
            v[k] = x
            norm += x * x
        }
        norm = norm.squareRoot()
        if norm > 0 { for k in 0 ..< dim { v[k] /= norm } }
        return v
    }

    /// Query `q`'s vector: the same construction at a disjoint index (see the header note).
    public static func query(_ q: Int, dim: Int = dim) -> [Float] { vec(queryIndexBase + q, dim: dim) }

    /// A whole query set. Deterministic and independent of the store, so the same `q` asks the same
    /// question of every arm and of every machine.
    public static func queries(_ n: Int, dim: Int = dim) -> [[Float]] {
        (0 ..< n).map { query($0, dim: dim) }
    }

    // MARK: - Rows

    /// Path of the synthetic file that owns row block `k`. Short and index-derived: the paper's
    /// vector cases never exercise the filename channel (`LexicalIndex` is pinned off for the whole
    /// suite), so a realistic path would add bytes to every row and measure nothing.
    public static func path(file k: Int) -> String { "f\(k)" }

    /// One file's rows. `chunksPerFile` chunks whose vectors are the `chunksPerFile` consecutive
    /// indices starting at `k * chunksPerFile`, so every vector index is used exactly once across
    /// the whole store.
    ///
    /// Several chunks per file on purpose: the store's per-file reduce and its tie-breaks are part
    /// of what search costs, and a one-chunk-per-file store skips both.
    public static func fileGroup(_ k: Int, chunksPerFile: Int = 4, snippetChars: Int = 0,
                                 dim: Int = dim) -> (path: String, chunks: [IndexedChunk]) {
        let p = path(file: k)
        let base = k * chunksPerFile
        return (p, (0 ..< chunksPerFile).map { c in
            IndexedChunk(path: p, modified: 0, size: 0, kind: "text", chunkIndex: c,
                         // Snippets are stored text: p10's compaction case needs the SQLite file to
                         // have realistic bulk, every other case wants the rows as small as they can
                         // be so the measurement is the scan and not the row decode.
                         snippet: snippetChars > 0
                             ? PaperCorpus.filler(characters: snippetChars, stream: UInt64(base + c))
                             : "",
                         embedding: vec(base + c, dim: dim))
        })
    }

    /// Row indices `lo ..< hi` as file groups, exactly as `gateparity` builds them. Both bounds must
    /// be multiples of `chunksPerFile`, because a partial file would give one file a different chunk
    /// count than every other and make the per-file reduce non-uniform.
    public static func rows(_ lo: Int, _ hi: Int, chunksPerFile: Int = 4, snippetChars: Int = 0,
                            dim: Int = dim) -> [(path: String, chunks: [IndexedChunk])] {
        precondition(lo % chunksPerFile == 0 && hi % chunksPerFile == 0,
                     "paper row range \(lo)..<\(hi) is not aligned to \(chunksPerFile) chunks per file")
        return generate(files: (lo / chunksPerFile) ..< (hi / chunksPerFile),
                        chunksPerFile: chunksPerFile, snippetChars: snippetChars, dim: dim)
    }

    /// The first `fraction` of a store's files, for the deletion that creates free pages before a
    /// compaction. Taken from the front rather than sampled: which rows are deleted does not change
    /// what VACUUM costs, and a deterministic set makes the two arms delete identically.
    public static func deletionPaths(rows: Int, chunksPerFile: Int = 4, fraction: Double) -> Set<String> {
        let files = rows / chunksPerFile
        return Set((0 ..< Int(Double(files) * fraction)).map { path(file: $0) })
    }

    // MARK: - Building

    /// Fill `store` with `rows` rows, in bounded batches.
    ///
    /// Returns the number of rows actually inserted, and the CALLER MUST STAMP THAT NUMBER rather
    /// than the one it asked for. It differs in two legitimate ways: the request is rounded down to
    /// a whole number of files (a `--scale` factor can turn 200,000 into a count that is not a
    /// multiple of `chunksPerFile`, and one short file would make the per-file reduce non-uniform),
    /// and a deadline stops the build early. A case that ran out of budget must report the smaller
    /// store and say so, never the full store it did not build. Cancel is not a deadline and throws,
    /// because a cancelled case has no result at all.
    ///
    /// - Parameters:
    ///   - deadline: stop and return early once passed. nil means run to completion.
    ///   - cancelled: polled once per batch; throws `CancellationError` when it goes true.
    ///   - progress: rows inserted so far, at every batch boundary.
    @discardableResult
    public static func buildStore(rows: Int, into store: VectorStore, chunksPerFile: Int = 4,
                                  snippetChars: Int = 0, dim: Int = dim, deadline: Date? = nil,
                                  cancelled: () -> Bool = { false },
                                  progress: (Int) -> Void = { _ in }) throws -> Int {
        // Rounded, not trapped: a precondition here would take the whole app down on a smoke run
        // whose scale factor happened not to divide by four.
        let target = (rows / chunksPerFile) * chunksPerFile
        let batchRows = max(chunksPerFile, (rowsPerBatch / chunksPerFile) * chunksPerFile)
        var done = 0
        while done < target {
            if cancelled() { throw CancellationError() }
            if let deadline, Date() >= deadline { break }
            let next = min(target, done + batchRows)
            try store.replaceMany(generate(files: (done / chunksPerFile) ..< (next / chunksPerFile),
                                           chunksPerFile: chunksPerFile, snippetChars: snippetChars, dim: dim))
            done = next
            progress(done)
        }
        return done
    }

    /// Generate a contiguous block of file groups, across cores when the block is worth it.
    ///
    /// Safe to parallelise precisely because `vec(i)` is pure: each iteration writes one distinct
    /// slot of a pre-sized buffer and reads nothing another iteration writes. Same bridge as
    /// `OmniTextEncoder.tokenIdsBatch` (OmniTextEncoder.swift:98-110); the compiler cannot prove
    /// the slots are disjoint, so the buffer crosses the boundary explicitly.
    private static func generate(files: Range<Int>, chunksPerFile: Int, snippetChars: Int,
                                 dim: Int) -> [(path: String, chunks: [IndexedChunk])] {
        let n = files.count
        guard n > 0 else { return [] }
        var out = [(path: String, chunks: [IndexedChunk])](repeating: ("", []), count: n)
        // Below this the dispatch costs more than the vectors do.
        guard n >= 64 else {
            for k in 0 ..< n {
                out[k] = fileGroup(files.lowerBound + k, chunksPerFile: chunksPerFile,
                                   snippetChars: snippetChars, dim: dim)
            }
            return out
        }
        out.withUnsafeMutableBufferPointer { buf in
            nonisolated(unsafe) let slots = buf
            let first = files.lowerBound
            DispatchQueue.concurrentPerform(iterations: n) { k in
                slots[k] = fileGroup(first + k, chunksPerFile: chunksPerFile,
                                     snippetChars: snippetChars, dim: dim)
            }
        }
        return out
    }

    // MARK: - Selection scores (p09)

    /// The score vector `selbench` selects over (main.swift:2948-2951), reproduced exactly.
    ///
    /// This one stream IS consumption-ordered - a single xorshift64 walk from the golden-ratio
    /// constant - unlike everything else in the paper module. Kept that way on purpose: p09's
    /// numbers are compared against the selection measurements already in the paper, and those were
    /// taken over these values in this order. It is generated serially for the same reason.
    public static func selectionScores(_ n: Int) -> [Float] {
        var state = UInt64(0x9E37_79B9_7F4A_7C15)
        var out = [Float](repeating: 0, count: n)
        for i in 0 ..< n {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17
            out[i] = Float(Int32(truncatingIfNeeded: state)) / Float(Int32.max)
        }
        return out
    }
}
