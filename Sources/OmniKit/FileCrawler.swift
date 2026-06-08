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
    public var ignore: OmniIgnore   // the exclude policy; index iff kind(for:) != nil && !ignored
    public var maxFileSize: Int

    /// Well-known noise directories. No longer special-cased in the crawl - migration SEEDS these as
    /// editable patterns in the default .omniignore (so power users can remove them).
    public static let skipDirNames: [String] = [
        "node_modules", ".git", ".svn", ".hg", "Library", "Pods", ".build",
        "DerivedData", "venv", ".venv", "env", "__pycache__", ".cache",
        "Caches", ".Trash", "vendor", "dist", "build", ".next", "target",
    ]

    public init(roots: [URL], ignore: OmniIgnore = OmniIgnore(text: ""), maxFileSize: Int = 200_000_000) {
        self.roots = roots
        self.ignore = ignore
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
                      FileExtractor.kind(for: url) != nil,
                      !ignore.isIgnored(url.path, isDir: false) else { continue }
                let size = vals.fileSize ?? 0
                if size > maxFileSize { continue }
                let mtime = vals.contentModificationDate?.timeIntervalSince1970 ?? 0
                onFile(CrawledFile(url: url, modified: mtime, size: size))
            }
        }
    }
}
