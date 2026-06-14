import Foundation

public struct CrawledFile: Sendable {
    public let url: URL
    public let modified: Double
    public let size: Int
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
            for case let url as URL in en {
                if !shouldContinue() { return }
                guard let vals = try? url.resourceValues(forKeys: Set(keys)) else { continue }
                if vals.isDirectory == true {
                    if ignore.isIgnored(url.path, isDir: true) || vals.isPackage == true {
                        en.skipDescendants()
                    }
                    continue
                }
                guard vals.isRegularFile == true,
                      let kind = FileExtractor.kind(for: url), enabledKinds.contains(kind),
                      !ignore.isIgnored(url.path, isDir: false) else { continue }
                let size = vals.fileSize ?? 0
                if let cap = maxFileSize[kind], size > cap { continue }   // per-kind cap; uncapped kinds stream
                let mtime = vals.contentModificationDate?.timeIntervalSince1970 ?? 0
                onFile(CrawledFile(url: url, modified: mtime, size: size))
            }
        }
    }
}
