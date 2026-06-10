import Foundation
import CryptoKit

/// Disk cache for folder-map projection results. PCA entries store points only; UMAP entries also store
/// the kNN graph used by click-to-highlight, so they can be much larger for big folders.
public enum ProjectionCache {
    public static let schemaVersion = 1
    /// Bound the on-disk cache: each entry holds a folder's full point cloud (and a UMAP kNN graph),
    /// so a few large folders across model/cap changes could otherwise grow without limit. LRU by
    /// file mtime, pruned after every save. The in-memory LRU (AppModel, cap 6) is the hot tier; this
    /// only has to survive restarts for the handful of folders a user actually maps.
    public static let maxDiskEntries = 16

    public struct Loaded: Sendable {
        public let result: ProjectionResult
        public let total: Int
    }

    private struct Entry: Codable {
        var version: Int
        var mode: String
        var folder: String
        var fingerprint: String
        var mapCap: Int
        var totalCap: Int
        var total: Int
        var signature: FolderVectorSignature
        var points: [Point]
        var knn: [Int32]
        var k: Int
    }

    private struct Point: Codable {
        var path: String
        var kind: String
        var x: Float
        var y: Float
    }

    public static func loadPCA(directory: URL, folder: String, fingerprint: String,
                               mapCap: Int, totalCap: Int, signature: FolderVectorSignature) -> Loaded? {
        load(directory: directory, folder: folder, fingerprint: fingerprint,
             mapCap: mapCap, totalCap: totalCap, signature: signature, mode: "pca")
    }

    public static func loadUMAP(directory: URL, folder: String, fingerprint: String,
                                mapCap: Int, totalCap: Int, signature: FolderVectorSignature) -> Loaded? {
        load(directory: directory, folder: folder, fingerprint: fingerprint,
             mapCap: mapCap, totalCap: totalCap, signature: signature, mode: "umap")
    }

    private static func load(directory: URL, folder: String, fingerprint: String,
                             mapCap: Int, totalCap: Int, signature: FolderVectorSignature,
                             mode: String) -> Loaded? {
        let url = cacheURL(directory: directory, mode: mode, folder: folder,
                           fingerprint: fingerprint, mapCap: mapCap, totalCap: totalCap)
        guard let data = try? Data(contentsOf: url),
              let entry = try? JSONDecoder().decode(Entry.self, from: data),
              entry.version == schemaVersion,
              entry.mode == mode,
              entry.folder == folder,
              entry.fingerprint == fingerprint,
              entry.mapCap == mapCap,
              entry.totalCap == totalCap,
              entry.signature == signature,
              !entry.points.isEmpty
        else { return nil }

        var out: [ProjectionPoint] = []
        out.reserveCapacity(entry.points.count)
        for p in entry.points {
            guard p.x.isFinite, p.y.isFinite else { return nil }
            out.append(ProjectionPoint(position: SIMD2(p.x, p.y), path: p.path, kind: p.kind))
        }
        return Loaded(result: ProjectionResult(points: out, knn: entry.knn, k: entry.k),
                      total: max(entry.total, out.count))
    }

    public static func savePCA(_ result: ProjectionResult, directory: URL, folder: String,
                               fingerprint: String, mapCap: Int, totalCap: Int,
                               signature: FolderVectorSignature, total: Int) {
        save(result, directory: directory, folder: folder, fingerprint: fingerprint,
             mapCap: mapCap, totalCap: totalCap, signature: signature, total: total, mode: "pca")
    }

    public static func saveUMAP(_ result: ProjectionResult, directory: URL, folder: String,
                                fingerprint: String, mapCap: Int, totalCap: Int,
                                signature: FolderVectorSignature, total: Int) {
        save(result, directory: directory, folder: folder, fingerprint: fingerprint,
             mapCap: mapCap, totalCap: totalCap, signature: signature, total: total, mode: "umap")
    }

    private static func save(_ result: ProjectionResult, directory: URL, folder: String,
                             fingerprint: String, mapCap: Int, totalCap: Int,
                             signature: FolderVectorSignature, total: Int, mode: String) {
        guard !result.points.isEmpty else { return }
        let points = result.points.map { Point(path: $0.path, kind: $0.kind, x: $0.position.x, y: $0.position.y) }
        let entry = Entry(version: schemaVersion, mode: mode, folder: folder,
                          fingerprint: fingerprint, mapCap: mapCap, totalCap: totalCap,
                          total: max(total, result.points.count),
                          signature: signature, points: points,
                          knn: mode == "umap" ? result.knn : [], k: mode == "umap" ? result.k : 0)
        let fm = FileManager.default
        do {
            try fm.createDirectory(at: directory, withIntermediateDirectories: true)
            let data = try JSONEncoder().encode(entry)
            try data.write(to: cacheURL(directory: directory, mode: mode, folder: folder,
                                        fingerprint: fingerprint, mapCap: mapCap, totalCap: totalCap),
                           options: .atomic)
            prune(directory: directory)
        } catch {
            // Rebuildable cache: ignore write failures.
        }
    }

    /// Keep at most `maxDiskEntries` cache files, evicting the oldest by modification time. Best-effort.
    private static func prune(directory: URL) {
        let fm = FileManager.default
        guard let urls = try? fm.contentsOfDirectory(at: directory,
                                                     includingPropertiesForKeys: [.contentModificationDateKey],
                                                     options: [.skipsHiddenFiles]) else { return }
        let entries = urls.filter { $0.pathExtension == "json" }
        guard entries.count > maxDiskEntries else { return }
        func mtime(_ u: URL) -> Date {
            (try? u.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
        }
        for u in entries.sorted(by: { mtime($0) < mtime($1) }).prefix(entries.count - maxDiskEntries) {
            try? fm.removeItem(at: u)
        }
    }

    private static func cacheURL(directory: URL, mode: String, folder: String, fingerprint: String,
                                 mapCap: Int, totalCap: Int) -> URL {
        let key = "\(mode)|\(fingerprint)|\(mapCap)|\(totalCap)|\(folder)"
        let name = SHA256.hash(data: Data(key.utf8)).map { String(format: "%02x", $0) }.joined()
        return directory.appendingPathComponent("\(mode)-\(name).json")
    }
}
