import Foundation

/// Filesystem layout and the non-destructive guarantee for the paper benchmark.
///
/// The user's real index must be untouched under EVERY code path, including cancel and failure.
/// Four independent mechanisms enforce that, so no single mistake can reach it:
///
///  1. One root. Every store, scratch tree and report the run creates lives under
///     $TMPDIR/omni-paper-run-<runId>/. The bulk (stores, scratch) is removed as soon as the run
///     ends; report.txt/json stay, because the dialog shows their path, until the stale sweep.
///  2. One choke point. `store(named:)` is the ONLY way the suite constructs a VectorStore, and it
///     preconditions that the path is inside the run dir, that the run dir is inside $TMPDIR, and
///     that the path is neither equal to nor a parent of any protected (real index) URL. There is
///     no other `VectorStore(dbURL:)` call anywhere in the Paper module.
///  3. The real index is never opened - not read-only, not for metadata. Which is why the export
///     carries no chunk count and no index size.
///  4. A stale sweep at start removes run dirs left by a crashed prior run.
///
/// The CORPUS deliberately lives OUTSIDE the run dir, in its own version-tagged directory, so it
/// survives cleanup and a second run on the same machine does not regenerate 4,616 files. Same
/// cache-stamp shape as ProfilingService.datasetFolder, and the version tag means a corpus change
/// produces a new directory rather than silently reusing old bytes.
public final class PaperFS: Sendable {
    public let runDir: URL
    public let corpusRoot: URL
    /// Paths the suite must never touch (the app's real index and its sidecars). Empty for the
    /// headless runner, which has no app index.
    private let protected: [String]

    public var storesDir: URL { runDir.appendingPathComponent("stores", isDirectory: true) }
    public var scratchDir: URL { runDir.appendingPathComponent("scratch", isDirectory: true) }
    public var reportTxtURL: URL { runDir.appendingPathComponent("report.txt") }
    public var reportJSONURL: URL { runDir.appendingPathComponent("report.json") }

    public init(runId: String, corpusVersion: String, protectedIndexURLs: [URL]) throws {
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
        runDir = tmp.appendingPathComponent("omni-paper-run-\(runId)", isDirectory: true)
        corpusRoot = tmp.appendingPathComponent("omni-paper-corpus-\(corpusVersion)", isDirectory: true)
        protected = protectedIndexURLs.map { $0.standardizedFileURL.path }
        let fm = FileManager.default
        try fm.createDirectory(at: runDir, withIntermediateDirectories: true)
        try fm.createDirectory(at: storesDir, withIntermediateDirectories: true)
        try fm.createDirectory(at: scratchDir, withIntermediateDirectories: true)
    }

    /// The ONLY VectorStore constructor the paper suite may use.
    public func store(named name: String) throws -> VectorStore {
        let url = storesDir.appendingPathComponent(name)
        try assertSafe(url)
        return try VectorStore(dbURL: url)
    }

    /// Close a store and delete it plus every sidecar it may have written (-wal/-shm, the row and
    /// vector sidecars, the quant replica). The suite builds several multi-hundred-MB stores in one
    /// run; leaving them until the final cleanup would need the peak of all of them at once on disk.
    public func discard(_ store: VectorStore, named name: String) {
        store.close()
        let base = storesDir.appendingPathComponent(name).path
        for suffix in ["", "-wal", "-shm", ".rows", ".vecs", ".quant", ".hits", ".vecdump", ".lex", ".lex-wal", ".lex-shm"] {
            try? FileManager.default.removeItem(at: URL(fileURLWithPath: base + suffix))
        }
    }

    /// A scratch subdirectory inside the run dir (p05's copied edit tree lives here).
    public func scratch(named name: String) throws -> URL {
        let url = scratchDir.appendingPathComponent(name, isDirectory: true)
        try assertSafe(url)
        try? FileManager.default.removeItem(at: url)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    public func cleanup() {
        try? FileManager.default.removeItem(at: runDir)
    }

    /// Remove run dirs from a crashed earlier run and corpus dirs from a superseded corpus version.
    /// Never touches anything outside $TMPDIR/omni-paper-*.
    public static func sweepStale(currentCorpusVersion: String) {
        sweepAbandonedRuns()
        let fm = FileManager.default
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
        guard let entries = try? fm.contentsOfDirectory(at: tmp, includingPropertiesForKeys: nil) else { return }
        for url in entries where url.lastPathComponent.hasPrefix("omni-paper-corpus-")
            && url.lastPathComponent != "omni-paper-corpus-\(currentCorpusVersion)" {
            try? fm.removeItem(at: url)
        }
    }

    /// The run-directory half of the sweep, split out because it is the half that must run WITHOUT a
    /// corpus version in hand.
    ///
    /// A killed run leaves its stores behind (measured: 407 MB after a SIGKILL two minutes into a
    /// full-scale run). While this only ran at the start of the NEXT paper run, a user who tried the
    /// feature once and never again kept those bytes until macOS reaped $TMPDIR days later. It is
    /// called at app launch instead, where the corpus version is not the app's business - and must
    /// not be, since sweeping corpus dirs from here would delete the CLI's scaled corpus every time
    /// the app starts.
    public static func sweepAbandonedRuns() {
        let fm = FileManager.default
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
        guard let entries = try? fm.contentsOfDirectory(at: tmp, includingPropertiesForKeys: [.contentModificationDateKey]) else { return }
        let cutoff = Date().addingTimeInterval(-24 * 3600)
        // A run dir with no report never finished: the process was killed or crashed mid-run, and it
        // still holds the stores. Those are dead weight the moment the process died, so they go an
        // hour after their last write - well past the 25-minute wall cap, so a run in flight in
        // another instance is never swept - instead of waiting the 24 h a FINISHED run's report is
        // deliberately kept for.
        let abandonedCutoff = Date().addingTimeInterval(-3600)
        for url in entries where url.lastPathComponent.hasPrefix("omni-paper-run-") {
            let mtime = (try? url.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate
            let hasReport = fm.fileExists(atPath: url.appendingPathComponent("report.json").path)
            if (mtime ?? .distantPast) < (hasReport ? cutoff : abandonedCutoff) { try? fm.removeItem(at: url) }
        }
    }

    // MARK: - The guarantee

    private func assertSafe(_ url: URL) throws {
        let path = url.standardizedFileURL.path
        let root = runDir.standardizedFileURL.path
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true).standardizedFileURL.path
        // Both directions are checked on purpose: "under the run dir" alone would be satisfied by a
        // run dir that had somehow been pointed at the user's home.
        precondition(path == root || path.hasPrefix(root + "/"),
                     "paper benchmark tried to open a store outside its run directory: \(path)")
        precondition(root == tmp || root.hasPrefix(tmp + "/"),
                     "paper benchmark run directory is not under the temporary directory: \(root)")
        for p in protected {
            precondition(path != p && !path.hasPrefix(p + "/") && !p.hasPrefix(path + "/"),
                         "paper benchmark tried to open the real index: \(path)")
        }
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    }
}
