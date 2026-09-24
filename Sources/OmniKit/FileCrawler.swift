import Foundation

/// One crawled file. Stores the PATH, not a URL, and hands out a URL on demand.
///
/// A full index pass holds every crawled file in memory at once (per-root lists, the interleaved
/// list, and the by-kind grouping), so what this struct retains is multiplied by the file count.
/// A Foundation URL is not one object: it brings a CFURL, an NSPathStore2 for the path, and - the
/// moment anything reads `resourceValues` - a CoreServices `_FileCache` holding those values, plus
/// the CFStrings inside it. Measured on a 211k-file index: 264k `_FileCache` (84 MB), 264k NSURL,
/// 597k NSPathStore2 and 1.7M CFStrings alive at once, ~550 MB of heap that the user sees as
/// "Other" and cannot explain. One Swift String per file is a single allocation; the URL is rebuilt
/// where it is used and dies immediately after.
public struct CrawledFile: Sendable {
    public let path: String
    public let modified: Double
    public let size: Int

    /// Materialized per access - deliberately not stored. Callers use it once per file, which is
    /// nothing next to decoding and embedding that file.
    ///
    /// ONLY VALID FOR A FILE ON DISK. A Photos asset rides this same struct with a `photos://` path
    /// (see PhotoLibrary), which is not a filesystem path and must never be turned into a file URL -
    /// the indexer branches on `isPhoto` before anything reaches here.
    public var url: URL { URL(fileURLWithPath: path) }

    /// The last path component, without building a URL to get it. What the UI calls the file.
    public var name: String { (path as NSString).lastPathComponent }

    /// The extension, likewise without a URL. Decides the file's kind everywhere.
    public var ext: String { (path as NSString).pathExtension }

    /// Is this a Photos-library asset rather than a file on disk?
    public var isPhoto: Bool { PhotoLibrary.isPhotoPath(path) }

    public init(path: String, modified: Double, size: Int) {
        self.path = path
        self.modified = modified
        self.size = size
    }

    public init(url: URL, modified: Double, size: Int) {
        self.init(path: url.path, modified: modified, size: size)
    }
}

/// Recursively enumerates supported files under a set of roots, skipping hidden
/// dirs, package bundles, and well-known noise (node_modules, .git, caches).
public struct FileCrawler: Sendable {
    public var roots: [URL]
    public var ignore: OmniIgnore   // the fine exclude policy; index iff kind enabled && !ignored
    /// Modalities the user has turned on. The coarse filter applied BEFORE `ignore`: a file is
    /// indexed iff its kind is in this set AND it is not ignored. Default: all four kinds on.
    public var enabledKinds: Set<FileKind>
    /// OMNI'S OWN DATA, never crawled, wherever the user has put it.
    ///
    /// The index folder and the model folder are both relocatable in Settings, so they are runtime
    /// values a seeded `.omniignore` line cannot track - and unlike the noise-dir patterns, these
    /// are not a matter of taste: a model directory holds `tokenizer.json`, 16 MB of vocabulary
    /// JSON that Omni would happily chunk into thousands of meaningless rows and then return as
    /// search results. Reported as "I changed the download folder for the models... it tries to
    /// actually index it" (issue #21's follow-up).
    ///
    /// The index files themselves survived only by luck - `.sqlite`, `.vecs`, `.quant` and `.rows`
    /// are not indexable extensions - which is not a property to keep relying on.
    public var ownDataPaths: [String] = []
    /// Per-kind file-size ceiling in bytes; a kind with NO entry is uncapped. Video and audio stream
    /// in bounded 240 s segments (embedStreamedVideo/Audio), so a multi-GB file is memory-safe - only
    /// slower to index - and is left uncapped. Text reads only the first maxTextBytes (2 MB) regardless
    /// of file size, so its size is irrelevant - uncapped. Images are decoded by ImageIO (which parses
    /// the whole file), so they keep a guard against pathological inputs. (Was a single 200 MB cap on
    /// ALL kinds, which silently skipped every multi-GB video - the exact files the streamed pipeline
    /// exists for. See issue #9.)
    public var maxFileSize: [FileKind: Int]
    /// Default policy: only images are capped (200 MB); video/audio/text are uncapped.
    public static let defaultMaxFileSize: [FileKind: Int] = [.image: 200_000_000]

    /// Well-known noise directories. No longer special-cased in the crawl - migration SEEDS these as
    /// editable patterns in the default .omniignore (so power users can remove them).
    public static let skipDirNames: [String] = [
        "node_modules", ".git", ".svn", ".hg", "Library", "Pods", ".build",
        "DerivedData", "venv", ".venv", "env", "__pycache__", ".cache",
        "Caches", ".Trash", "vendor", "dist", "build", "_build", ".next", "target",
    ]

    public init(roots: [URL], ignore: OmniIgnore = OmniIgnore(text: ""),
                enabledKinds: Set<FileKind> = [.text, .image, .video, .audio],
                maxFileSize: [FileKind: Int] = FileCrawler.defaultMaxFileSize,
                ownDataPaths: [String] = []) {
        self.roots = roots
        self.ignore = ignore
        self.enabledKinds = enabledKinds
        self.maxFileSize = maxFileSize
        // REAL PATHS, resolved once here, for the same reason walkBulk resolves its roots: the
        // walk compares against paths with every symlink already collapsed (/var/... arrives as
        // /private/var/...). Standardizing without resolving left the two spellings unequal and
        // the exclusion silently matched nothing - which the tests caught only because they run
        // under a temp directory, where that difference is real.
        self.ownDataPaths = ownDataPaths.map { p in
            var buf = [CChar](repeating: 0, count: Int(PATH_MAX))
            if realpath(p, &buf) != nil { return String(cString: buf) }
            return URL(fileURLWithPath: p).standardizedFileURL.path
        }
    }

    /// Is this path Omni's own data? Boundary-aware, so a sibling that merely starts with the same
    /// characters ("/data/omni-index2" under "/data/omni-index") is not swallowed.
    @inline(__always) func isOwnData(_ path: String) -> Bool {
        guard !ownDataPaths.isEmpty else { return false }
        for p in ownDataPaths where path == p || path.hasPrefix(p + "/") { return true }
        return false
    }

    /// Whether a crawl from `root` would reach `path`: the rules `walkBulk` applies on the way
    /// down, for a path that arrives on its own (a watcher event). No hidden component below the
    /// root, no package on the way, not Omni's own data, and a file within its kind's size cap.
    /// The ignore rules are checked by the caller (isIgnoredIncludingAncestors). Without this the
    /// watcher indexed what the crawl refuses - the contents of a .app unzipped into a root, files
    /// under .vscode or .mypy_cache - and the next full pass swept them out again, so every save
    /// paid an embed that was always going to be deleted.
    func admitsEventPath(_ path: String, isDir: Bool, size: Int, root: String) -> Bool {
        if isOwnData(path) { return false }
        if path == root { return true }                       // a root is always descended into
        guard path.hasPrefix(root + "/") else { return true } // not under this root: nothing to add
        var comps = path.dropFirst(root.count + 1).split(separator: "/")
        guard let last = comps.popLast() else { return true }
        var dir = root
        for c in comps {
            dir += "/" + c
            if c.hasPrefix(".") || PackageProbe.isPackage(dir) { return false }
        }
        if last.hasPrefix(".") { return false }
        if isDir { return !PackageProbe.isPackage(path) }
        if let kind = FileExtractor.kind(forExtension: (path as NSString).pathExtension),
           let cap = maxFileSize[kind], size > cap { return false }
        return true
    }

    /// Default user folders to index.
    /// Engine selector. OMNI_CRAWLER=legacy restores the FileManager enumerator, so the two can be
    /// A/B'd in one build - and so a user who hits trouble with the fast walk has a way back.
    static var useLegacyEngine: Bool {
        ProcessInfo.processInfo.environment["OMNI_CRAWLER"] == "legacy"
    }

    /// Walk all roots, invoking `onFile` for each supported file. `shouldContinue`
    /// is polled so indexing can be cancelled.
    ///
    /// `onFile` is called serially, but NOT necessarily on the calling thread: the fast engine runs
    /// a pool of readers and hands their results over one directory at a time under its own lock.
    /// Every caller here appends to a local array, which that guarantee covers.
    public func walk(shouldContinue: () -> Bool = { true }, onFile: (CrawledFile) -> Void) {
        if Self.useLegacyEngine { walkLegacy(shouldContinue: shouldContinue, onFile: onFile) }
        else { walkBulk(shouldContinue: shouldContinue, onFile: onFile) }
    }

    /// The fast engine: getattrlistbulk + a bounded pool. See BulkDirWalker for why it is shaped
    /// this way and what was measured. The POLICY below - what to descend into, what to keep - is
    /// identical to the legacy walk, and PackageProbe/hidden/symlink/volume handling exists purely
    /// so it stays identical; CrawlerParityTests holds the two to the same answer.
    private func walkBulk(shouldContinue: () -> Bool, onFile: (CrawledFile) -> Void) {
        // REAL PATHS. FileManager's enumerator hands back paths with every symlink in the ROOT
        // already resolved (/var/... comes back as /private/var/...), and the fast walk builds its
        // paths by concatenation, so without this the two engines return the same files under
        // different names. Those names are the index's keys: a switch would make every file on the
        // machine look new, and re-embed all of it. Once per root, not per entry.
        var resolved: [(url: URL, path: String)] = roots.map { r in
            var buf = [CChar](repeating: 0, count: Int(PATH_MAX))
            if realpath(r.path, &buf) != nil { return (r, String(cString: buf)) }
            return (r, r.path)
        }
        // DEFENCE IN DEPTH AGAINST OVERLAPPING ROOTS. AppModel already reduces the user's folders
        // to ancestors before any pass starts, and until now that was the ONLY thing standing
        // between "add a folder and its parent" and crawling the same tree twice: BulkDirWalker
        // pushes every root onto one shared stack and has no notion of one containing another, so
        // a nested pair would be walked from both ends - every file hashed twice, and the ones the
        // index had not seen embedded twice.
        //
        // Dropping the nested root here costs one pass over a handful of paths and makes the
        // guarantee local to the thing that does the walking, rather than a property of every
        // caller remembering. Resolved paths, because that is what the walk uses as its keys.
        let keep = Set(RootScope.canonical(resolved.map { URL(fileURLWithPath: $0.path) }).map(\.path))
        resolved = resolved.filter { keep.contains($0.path) }

        // The device of each root, so the walk stays on one volume the way the enumerator does. A
        // mount point inside a root (a disk image, a network share) would otherwise be crawled as
        // part of it - a different volume's worth of files under a root the user never chose.
        var rootDevice: [String: dev_t] = [:]
        for r in resolved {
            var st = stat()
            if stat(r.path, &st) == 0 { rootDevice[r.path] = st.st_dev }
        }
        func rootFor(_ path: String) -> String? {
            resolved.first { path == $0.path || path.hasPrefix($0.path + "/") }?.path
        }

        BulkDirWalker.walk(
            roots: resolved.map { $0.path },
            shouldContinue: shouldContinue,
            shouldDescend: { dir, entry in
                // A root itself is always descended into: the user chose it, hidden or not.
                if rootDevice[dir] != nil { return true }
                let name = (dir as NSString).lastPathComponent
                if name.hasPrefix(".") { return false }            // matches .skipsHiddenFiles
                if ignore.isIgnored(dir, isDir: true) { return false }
                if isOwnData(dir) { return false }
                // The volume check reads the id the SAME syscall already returned. A mount point
                // inside a root (a disk image, a share) is a different volume's worth of files
                // under a root the user never chose.
                if let e = entry, e.device != 0, let r = rootFor(dir), let want = rootDevice[r], want != e.device {
                    return false
                }
                if PackageProbe.isPackage(dir) { return false }    // .app, .photoslibrary, ...
                return true
            },
            keep: { dir, e -> CrawledFile? in
                // IN THE WORKER: everything here is per-file and runs on all cores at once.
                guard !e.isDir, !e.isSymlink, !e.name.hasPrefix(".") else { return nil }
                // The extension comes from the name the syscall returned - no URL is built.
                guard let kind = FileExtractor.kind(forExtension: (e.name as NSString).pathExtension),
                      enabledKinds.contains(kind) else { return nil }
                if let cap = maxFileSize[kind], e.size > cap { return nil }
                let path = dir + "/" + e.name
                guard !ignore.isIgnored(path, isDir: false), !isOwnData(path) else { return nil }
                return CrawledFile(path: path, modified: e.mtime, size: e.size)
            },
            deliver: { batch in for f in batch { onFile(f) } })
    }

    private func walkLegacy(shouldContinue: () -> Bool, onFile: (CrawledFile) -> Void) {
        let fm = FileManager.default
        let keys: [URLResourceKey] = [.isDirectoryKey, .isRegularFileKey, .contentModificationDateKey, .fileSizeKey, .isPackageKey, .isHiddenKey]
        for root in roots {
            guard let en = fm.enumerator(at: root, includingPropertiesForKeys: keys,
                                         options: [.skipsHiddenFiles], errorHandler: { _, _ in true })
            else { continue }
            let keySet = Set(keys)
            for case let url as URL in en {
                if !shouldContinue() { return }
                // Per-iteration pool: the enumerator vends autoreleased URLs and every
                // resourceValues read allocates more Foundation objects behind them. Without a
                // pool here they all pile up until the walk RETURNS - hundreds of thousands of
                // objects on a large root, freed only after the crawl no longer needs them.
                var crawled: CrawledFile?
                autoreleasepool {
                    guard let vals = try? url.resourceValues(forKeys: keySet) else { return }
                    if vals.isDirectory == true {
                        if ignore.isIgnored(url.path, isDir: true) || vals.isPackage == true
                            || isOwnData(url.path) {
                            en.skipDescendants()
                        }
                        return
                    }
                    guard vals.isRegularFile == true,
                          let kind = FileExtractor.kind(for: url), enabledKinds.contains(kind),
                          !ignore.isIgnored(url.path, isDir: false), !isOwnData(url.path) else { return }
                    let size = vals.fileSize ?? 0
                    if let cap = maxFileSize[kind], size > cap { return }   // per-kind cap; uncapped kinds stream
                    let mtime = vals.contentModificationDate?.timeIntervalSince1970 ?? 0
                    crawled = CrawledFile(path: url.path, modified: mtime, size: size)
                }
                // Handed over OUTSIDE the pool: onFile is the caller's pipeline, and its own
                // allocations have nothing to do with this iteration's temporaries.
                if let crawled { onFile(crawled) }
            }
        }
    }
}
