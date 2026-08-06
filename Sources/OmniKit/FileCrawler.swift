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
    public var url: URL { URL(fileURLWithPath: path) }

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
        "Caches", ".Trash", "vendor", "dist", "build", ".next", "target",
    ]

    public init(roots: [URL], ignore: OmniIgnore = OmniIgnore(text: ""),
                enabledKinds: Set<FileKind> = [.text, .image, .video, .audio],
                maxFileSize: [FileKind: Int] = FileCrawler.defaultMaxFileSize) {
        self.roots = roots
        self.ignore = ignore
        self.enabledKinds = enabledKinds
        self.maxFileSize = maxFileSize
    }

    /// Default user folders to index.
    public static func defaultRoots() -> [URL] {
        let fm = FileManager.default
        return [.documentDirectory, .downloadsDirectory, .desktopDirectory]
            .compactMap { try? fm.url(for: $0, in: .userDomainMask, appropriateFor: nil, create: false) }
    }

    /// Walk all roots, invoking `onFile` for each supported file. `shouldContinue`
    /// is polled so indexing can be cancelled.
    public func walk(shouldContinue: () -> Bool = { true }, onFile: (CrawledFile) -> Void) {
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
                        if ignore.isIgnored(url.path, isDir: true) || vals.isPackage == true {
                            en.skipDescendants()
                        }
                        return
                    }
                    guard vals.isRegularFile == true,
                          let kind = FileExtractor.kind(for: url), enabledKinds.contains(kind),
                          !ignore.isIgnored(url.path, isDir: false) else { return }
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
