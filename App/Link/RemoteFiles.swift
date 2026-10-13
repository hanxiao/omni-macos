import AppKit
import CryptoKit
import Foundation
import ImageIO
import OmniKit
import OmniLink

/// A linked Mac's files on this one: thumbnails, and real copies for Open, Quick Look, Share and
/// drag-out, fetched over the link when asked for and kept in a cache.
///
/// COPIES ARE READ-ONLY. A fetched file is a copy of the one on the other Mac; editing it here
/// would change nothing there, and an app that saved into it would make the user think it had. The
/// copy is made 0444 so an editor opens it read-only and says so.
///
/// Thread-safe and not tied to an actor: Thumbnail asks from a view task, the drag promise from an
/// AppKit operation queue.
final class RemoteFiles: @unchecked Sendable {
    static let shared = RemoteFiles()

    private let lock = NSLock()
    private var node: LinkNode?
    private var inflight: [String: Task<URL, Error>] = [:]
    /// What a running fetch has received, by remote path (0...1), for the progress line.
    private var progress: [String: Double] = [:]

    /// The cache: ~/Library/Caches/<bundle>/Linked/<peer>/<hash of path>/<name>. Bounded; the
    /// oldest copies go first.
    let root: URL = {
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        return caches.appendingPathComponent(Bundle.main.bundleIdentifier ?? "io.hanxiao.omni", isDirectory: true)
            .appendingPathComponent("Linked", isDirectory: true)
    }()
    static let cacheLimit: Int64 = 4 << 30
    /// Slices fetched at once. Two keep the link busy while one is in flight back.
    static let parallel = 3

    func attach(_ node: LinkNode?) { lock.withLock { self.node = node } }
    private var current: LinkNode? { lock.withLock { node } }

    func fraction(_ remote: String) -> Double? { lock.withLock { progress[remote] } }

    // MARK: - Thumbnails

    func thumbnail(_ remote: String, maxPixel: Int) async -> CGImage? {
        guard let (peer, path) = RemotePath.parse(remote), let node = current else { return nil }
        do {
            let reply = try await node.call(peer, "thumb", .json(LinkWire.PathRequest(path: path, px: maxPixel)), timeout: 15)
            guard !reply.blob.isEmpty, let src = CGImageSourceCreateWithData(reply.blob as CFData, nil) else { return nil }
            return CGImageSourceCreateImageAtIndex(src, 0, nil)
        } catch {
            return nil
        }
    }

    // MARK: - Files

    /// A local copy of `remote`, fetched if the cache does not hold the current version. One fetch
    /// per file at a time; a second caller waits for the first.
    func fetch(_ remote: String) async throws -> URL {
        let task: Task<URL, Error> = lock.withLock {
            if let t = inflight[remote] { return t }
            let t = Task<URL, Error> { try await self.download(remote) }
            inflight[remote] = t
            return t
        }
        defer { lock.withLock { inflight[remote] = nil; progress[remote] = nil } }
        return try await task.value
    }

    /// The same, blocking: for a file promise, which AppKit fulfils on its own operation queue.
    func fetchBlocking(_ remote: String) -> URL? {
        let out = LockedBox<URL?>(nil)
        let sem = DispatchSemaphore(value: 0)
        Task.detached { out.value = try? await self.fetch(remote); sem.signal() }
        sem.wait()
        return out.value
    }

    func cacheURL(_ remote: String, name: String) -> URL {
        let (peer, path) = RemotePath.parse(remote) ?? ("unknown", remote)
        let h = SHA256.hash(data: Data(path.utf8)).prefix(10).map { String(format: "%02x", $0) }.joined()
        return root.appendingPathComponent(peer, isDirectory: true)
            .appendingPathComponent(h, isDirectory: true)
            .appendingPathComponent(name)
    }

    private func download(_ remote: String) async throws -> URL {
        guard let (peer, path) = RemotePath.parse(remote) else { throw LinkError.refused("not a linked Mac's file") }
        guard let node = current else { throw LinkError.refused("Linked Macs is off") }
        let infoMsg = try await node.call(peer, "fetch", .json(LinkWire.PathRequest(path: path, length: 0)), timeout: 30)
        let info = try infoMsg.decode(LinkWire.FileInfo.self)
        let dest = cacheURL(remote, name: info.name)
        let fm = FileManager.default
        // The cached copy is current if it is the same size and carries the same date.
        if let a = try? fm.attributesOfItem(atPath: dest.path),
           (a[.size] as? NSNumber)?.int64Value == info.size,
           let d = a[.modificationDate] as? Date, abs(d.timeIntervalSince1970 - info.modified) < 1 {
            try? fm.setAttributes([.modificationDate: Date()], ofItemAtPath: dest.deletingLastPathComponent().path)
            return dest
        }
        try fm.createDirectory(at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)
        let tmp = dest.deletingLastPathComponent().appendingPathComponent(".part-\(UUID().uuidString)")
        fm.createFile(atPath: tmp.path, contents: nil)
        defer { try? fm.removeItem(at: tmp) }
        let fh = try FileHandle(forWritingTo: tmp)
        let slice = Int64(LinkResponder.maxSlice)
        let count = info.size == 0 ? 0 : Int((info.size + slice - 1) / slice)
        var done = 0
        do {
            try await withThrowingTaskGroup(of: (Int, Data).self) { group in
                var next = 0
                func add() {
                    let i = next
                    next += 1
                    group.addTask {
                        let r = try await node.call(peer, "fetch",
                                                    .json(LinkWire.PathRequest(path: path, offset: Int64(i) * slice, length: Int(slice))),
                                                    timeout: 60)
                        return (i, r.blob)
                    }
                }
                while next < min(count, Self.parallel) { add() }
                while let (i, data) = try await group.next() {
                    try fh.seek(toOffset: UInt64(Int64(i) * slice))
                    try fh.write(contentsOf: data)
                    done += 1
                    lock.withLock { progress[remote] = Double(done) / Double(max(count, 1)) }
                    if next < count { add() }
                }
            }
            try fh.close()
        } catch {
            try? fh.close()
            throw error
        }
        let got = (try? fm.attributesOfItem(atPath: tmp.path)[.size] as? NSNumber)?.int64Value ?? -1
        guard got == info.size else { throw LinkError.protocolViolation("received \(got) of \(info.size) bytes") }
        try? fm.removeItem(at: dest)
        try fm.moveItem(at: tmp, to: dest)
        try fm.setAttributes([.modificationDate: Date(timeIntervalSince1970: info.modified),
                              .posixPermissions: 0o444], ofItemAtPath: dest.path)
        prune()
        return dest
    }

    /// Keep the cache under its limit, oldest copies first.
    func prune() {
        let fm = FileManager.default
        guard let e = fm.enumerator(at: root, includingPropertiesForKeys: [.fileSizeKey, .contentAccessDateKey, .isRegularFileKey]) else { return }
        var files: [(URL, Int64, Date)] = []
        var total: Int64 = 0
        for case let u as URL in e {
            guard let v = try? u.resourceValues(forKeys: [.fileSizeKey, .contentAccessDateKey, .isRegularFileKey]),
                  v.isRegularFile == true else { continue }
            let n = Int64(v.fileSize ?? 0)
            total += n
            files.append((u, n, v.contentAccessDate ?? .distantPast))
        }
        guard total > Self.cacheLimit else { return }
        for (u, n, _) in files.sorted(by: { $0.2 < $1.2 }) {
            try? fm.setAttributes([.posixPermissions: 0o644], ofItemAtPath: u.path)
            try? fm.removeItem(at: u)
            total -= n
            if total <= Self.cacheLimit { break }
        }
    }

    /// Forget everything fetched from a Mac (it was unlinked).
    func forget(peer: String) {
        let dir = root.appendingPathComponent(peer, isDirectory: true)
        let fm = FileManager.default
        if let e = fm.enumerator(at: dir, includingPropertiesForKeys: nil) {
            for case let u as URL in e { try? fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: u.path) }
        }
        try? fm.removeItem(at: dir)
    }
}

final class LockedBox<T>: @unchecked Sendable {
    private let l = NSLock()
    private var v: T
    init(_ v: T) { self.v = v }
    var value: T {
        get { l.withLock { v } }
        set { l.withLock { v = newValue } }
    }
}
