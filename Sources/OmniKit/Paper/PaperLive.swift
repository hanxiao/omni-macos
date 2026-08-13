import Foundation

// The live index, as a measurable object.
//
// Every case in PaperCasesCompute and PaperCasesStore runs against a corpus the suite generated
// from a seed, because an ablation needs both arms on identical bytes. The cases in
// PaperCasesLive run against the files the machine actually holds, which is a different claim and
// needs a different handle: what a user's own corpus costs to search, to index and to keep up
// with, at that user's own scale.
//
// Two rules make that safe to do from inside the shipping app:
//
//  - READ-ONLY on the live store. Search, find-similar and the index summary only read. Nothing
//    here writes to the user's index, ever. The write-side cases (indexing, re-index after an
//    edit, tagging) copy a SAMPLE of real files into the run's own scratch directory and index
//    those into a throwaway store, exactly as runProfilingPass already does.
//  - NOTHING IDENTIFYING LEAVES. The export carries counts, bytes and latencies. No path, no
//    filename, no snippet, no query text drawn from user content. The sampler below returns URLs
//    for the suite's own use inside the process; the case bodies turn them into numbers.

/// A handle on the machine's real index, plus the roots it was built from. Absent when the suite
/// runs outside the app (omni-verify) or before an index exists, in which case the live cases
/// record `skipped:no-live-index` rather than a measured zero.
public struct PaperLiveIndex: @unchecked Sendable {
    public let store: VectorStore
    public let roots: [URL]
    /// "nano" or "small": the variant the live index was BUILT with, which is not necessarily the
    /// one loaded now. Recorded so a corpus measured under one encoder is never merged with another.
    public let modelVariant: String

    public init(store: VectorStore, roots: [URL], modelVariant: String) {
        self.store = store
        self.roots = roots
        self.modelVariant = modelVariant
    }

    /// What the corpus is, in numbers. One pass over the store, taken once per case that needs it.
    public struct Snapshot: Sendable {
        public var files: Int
        public var chunks: Int
        public var indexBytes: Int64
        public var vectorDim: Int
        /// File counts per kind (text, image, audio, video, scan), the modality mix of the corpus.
        public var kinds: [String]
        public var exts: [String]
    }

    public func snapshot() -> Snapshot {
        let s = store.indexSummary(folders: roots.map(\.path))
        return Snapshot(files: s.fileCount, chunks: s.chunkCount, indexBytes: store.sizeBytes(),
                        vectorDim: store.vectorDim, kinds: s.kinds.sorted(), exts: s.exts.sorted())
    }
}

/// Real files drawn from the machine's own roots, for the cases that must not touch the live store.
///
/// Deterministic given (roots, seed): the pool is sorted by path and the pick is a seeded stride
/// over it rather than a random draw, so re-running a case on the same machine measures the same
/// files. The sort is what makes that true - the crawl delivers from a worker pool, in no
/// particular order. It is NOT comparable across machines, and it is not meant to be: that is the point
/// of measuring a personal corpus.
public struct PaperFileSampler: Sendable {
    public let roots: [URL]
    public let seed: UInt64

    public init(roots: [URL], seed: UInt64 = PaperCaseCatalog.mlxSeed) {
        self.roots = roots
        self.seed = seed
    }

    /// Up to `limit` files of the given kinds, with a size window so the sample is not dominated by
    /// one enormous file. `shouldContinue` is polled: a walk over a real home directory is the one
    /// thing here that can take real time, and a cancel has to land during it.
    public func sample(kinds: Set<FileKind>, limit: Int,
                       minBytes: Int = 1_024, maxBytes: Int = 8_000_000,
                       shouldContinue: () -> Bool = { true }) -> [URL] {
        guard limit > 0 else { return [] }
        var pool: [URL] = []
        // Bounded so a 500k-file corpus does not build a half-million-element array to pick 300
        // files out of: the pool is a wide prefix of the walk, and the stride below spreads the
        // pick across it. 40x the ask, capped, is wide enough to cross directories.
        let poolCap = min(40 * limit, 20_000)
        FileCrawler(roots: roots, enabledKinds: kinds).walk(shouldContinue: {
            shouldContinue() && pool.count < poolCap
        }, onFile: { f in
            guard f.size >= minBytes, f.size <= maxBytes else { return }
            guard let k = FileExtractor.kind(for: f.url), kinds.contains(k) else { return }
            pool.append(f.url)
        })
        guard !pool.isEmpty else { return [] }
        // SORTED BEFORE THE STRIDE, because the walk order is no longer an order.
        //
        // This sampler was written against a serial enumerator, and its contract - same machine,
        // same seed, same files - rested on the walk delivering a repeatable sequence. The crawl is
        // now a bounded pool of workers, so the pool below is whichever `poolCap` files eight
        // threads happened to deliver first, and it differs from run to run. The stride then picked
        // different files, and an indexing rate measured over them moved by 30% between two runs of
        // one machine, which is larger than most of the differences the paper reports between
        // machines. Sorting is O(n log n) on at most 20,000 paths and restores the contract.
        pool.sort { $0.path < $1.path }
        if pool.count <= limit { return pool }
        // Seeded stride: coprime step over the pool so the pick spans it instead of taking a
        // contiguous run out of whichever directory the walk happened to start in.
        var step = Int(seed % UInt64(pool.count))
        if step < 1 { step = 1 }
        while gcd(step, pool.count) != 1 { step += 1 }
        var out: [URL] = []
        var i = Int(seed % UInt64(pool.count))
        for _ in 0 ..< limit {
            out.append(pool[i])
            i = (i + step) % pool.count
        }
        return out
    }

    private func gcd(_ a: Int, _ b: Int) -> Int { b == 0 ? a : gcd(b, a % b) }
}

/// Copy real files into the run's scratch directory so a write-side case can index them without
/// going anywhere near the user's own index. Returns the copies, in the order they were taken.
public func paperStageRealFiles(_ urls: [URL], into dir: URL) throws -> [URL] {
    let fm = FileManager.default
    try fm.createDirectory(at: dir, withIntermediateDirectories: true)
    var staged: [URL] = []
    for (i, src) in urls.enumerated() {
        // Renamed to an index-plus-extension: the copy's NAME carries no user content, so a
        // filename can never reach the export through a log line or an error message.
        let dst = dir.appendingPathComponent(String(format: "f%05d.%@", i, src.pathExtension))
        if fm.fileExists(atPath: dst.path) { try? fm.removeItem(at: dst) }
        do {
            try fm.copyItem(at: src, to: dst)
            staged.append(dst)
        } catch {
            continue   // unreadable file: skip it, the sample is a sample
        }
    }
    return staged
}
