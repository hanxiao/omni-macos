import AppKit
import Foundation
import ImageIO
import OmniKit
import OmniLink
import UniformTypeIdentifiers

/// What this Mac answers a linked Mac with: searches over its own index on its own GPU, thumbnails,
/// file contents, its sources and folder listings. LinkNode has already checked that the asker is
/// linked and allowed to search (and, for `fetch`, to open files).
///
/// ONLY WHAT THE INDEX HOLDS. Every path a request names is checked against this Mac's index before
/// anything is read: a linked Mac can reach the files Omni indexed here and nothing else - not an
/// arbitrary path, not a file in an excluded folder, and never the clipboard history, which no
/// served surface returns.
final class LinkResponder: LinkService, @unchecked Sendable {
    private let lock = NSLock()
    private var engine: OmniEngine?
    private var store: VectorStore?
    private var modelName = ""
    private let lanes = SearchLanes()

    func attach(engine: OmniEngine, store: VectorStore, modelName: String) {
        lock.withLock { self.engine = engine; self.store = store; self.modelName = modelName }
    }

    func detach() { lock.withLock { engine = nil; store = nil } }

    private func backend() throws -> (OmniEngine, VectorStore, String) {
        guard let (e, s, m) = lock.withLock({ () -> (OmniEngine, VectorStore, String)? in
            guard let engine, let store else { return nil }
            return (engine, store, modelName)
        }) else { throw LinkError.refused("Omni on that Mac is still starting.") }
        return (e, s, m)
    }

    func serve(_ method: String, _ msg: LinkMessage, peer: LinkPeer) async throws -> LinkMessage {
        switch method {
        case "space": return try await space()
        case "search": return try await search(msg.decode(LinkWire.SearchRequest.self), peer: peer.id)
        case "similar": return try await similar(msg.decode(LinkWire.PathRequest.self))
        case "passages": return try await passages(msg.decode(LinkWire.PathRequest.self))
        case "thumb": return try await thumb(msg.decode(LinkWire.PathRequest.self))
        case "fetch": return try await fetch(msg.decode(LinkWire.PathRequest.self))
        case "sources": return try await sources()
        case "list": return try await list(msg.decode(LinkWire.PathRequest.self))
        default: throw LinkError.refused("Unknown request.")
        }
    }

    // MARK: - Requests

    /// The loaded model's embedding of the space probe text: two Macs whose models give the same
    /// vector (cosine >= 0.999) rank on one scale. The index's stored probe is not used - a new
    /// index has none until its first pass - and an index built by other weights is already
    /// obsolete on its own Mac (AppModel.vectorSpaceDiffers).
    private func space() async throws -> LinkMessage {
        let (engine, _, model) = try backend()
        let probe = try await offPool { LinkResponder.probe(engine) }
        let files = await MainActor.run { AppModel.shared?.indexedFiles ?? 0 }
        return try .json(LinkWire.Space(dim: engine.dim, probe: probe, model: model, files: files, fusesText: true))
    }

    /// The engine keeps it: one forward per encoder, shared with the model stamp.
    static func probe(_ engine: OmniEngine) -> [Float] {
        engine.vectorSpaceProbe(AppModel.vectorSpaceProbeText)
    }

    /// One search per linked Mac at a time, and the newest wins: a Mac typing "b", "be", "bea" must
    /// not queue three searches on this one's GPU, only the last.
    private func search(_ req: LinkWire.SearchRequest, peer: String) async throws -> LinkMessage {
        let (engine, store, _) = try backend()
        let ticket = await lanes.enter(peer)
        defer { Task { await lanes.leave(peer) } }
        guard await lanes.isLatest(peer, ticket), !Task.isCancelled else { throw LinkError.refused("superseded") }
        var f = req.filter.searchFilter
        f.excludeFolders.append(AppModel.clipboardDirectory.path)
        let filter = f
        let topK = max(1, min(req.topK, VectorStore.shippedTopK))
        let exclude = Set(req.exclude ?? [])
        let t0 = DispatchTime.now().uptimeNanoseconds
        let hits: [SearchHit] = try await offPool {
            let q = req.query.flatMap { $0.isEmpty ? nil : $0 }
            if let v = req.vector {
                guard v.count == engine.dim else { throw LinkError.refused("Not from this Mac's model.") }
                return store.search(v, filter: filter, topK: topK + exclude.count, textQuery: q)
            }
            guard let q else { return [] }
            if let v = engine.cachedQueryVector(q) { return store.search(v, filter: filter, topK: topK, textQuery: q) }
            let generation = engine.encoderGeneration
            if let g = engine.queryVectorGraph(q) {
                let (hits, v) = store.search(queryGraph: g, filter: filter, topK: topK, textQuery: q)
                engine.rememberQueryVector(q, v, since: generation)
                return hits
            }
            return store.search(engine.embedQuery(q), filter: filter, topK: topK, textQuery: q)
        }
        let kept = hits.filter { !exclude.contains($0.path) }.prefix(topK)
        let ms = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
        return try .json(LinkWire.SearchReply(hits: kept.map(LinkWire.Hit.init), ms: ms))
    }

    private func similar(_ req: LinkWire.PathRequest) async throws -> LinkMessage {
        let (_, store, _) = try backend()
        try servable(req.path, store)
        let path = req.path
        let clip = AppModel.clipboardDirectory.path
        let (hits, vec): ([SearchHit], [Float]?) = try await offPool {
            guard let v = store.fileVector(path) else { return ([], nil) }
            var f = SearchFilter()
            f.excludeFolders = [clip]
            return (store.search(v, filter: f, topK: VectorStore.shippedTopK + 1).filter { $0.path != path }, v)
        }
        return try .json(LinkWire.Similar(hits: hits.map(LinkWire.Hit.init), vector: vec))
    }

    private func passages(_ req: LinkWire.PathRequest) async throws -> LinkMessage {
        let (engine, store, _) = try backend()
        try servable(req.path, store)
        let path = req.path
        let found: [ChunkHit] = try await offPool {
            let vec: [Float]
            if let v = req.vector, v.count == engine.dim { vec = v }
            else if let q = req.query, !q.isEmpty { vec = engine.embedQuery(q) }
            else { return [] }
            return store.rankChunks(vec, path: path)
        }
        return try .json(LinkWire.Passages(passages: found.map {
            LinkWire.Passage(chunk: $0.chunkIndex, score: $0.score, snippet: $0.snippet, loc: $0.locator)
        }))
    }

    private func thumb(_ req: LinkWire.PathRequest) async throws -> LinkMessage {
        let (_, store, _) = try backend()
        try servable(req.path, store)
        let px = max(32, min(req.px ?? 256, 1024))
        guard let cg = await Thumbnail.render(path: req.path, maxPixel: px, side: CGFloat(px) / 2, scale: 2),
              let jpeg = Self.jpeg(cg) else { return LinkMessage() }
        return LinkMessage(body: Data(), blob: jpeg)
    }

    /// A slice of the file (`length` 0 asks for its size and date only). Photos assets are exported
    /// first, exactly as Quick Look on this Mac would.
    static let maxSlice = 4 << 20

    private func fetch(_ req: LinkWire.PathRequest) async throws -> LinkMessage {
        let (_, store, _) = try backend()
        try servable(req.path, store)
        let path = req.path
        return try await offPool {
            guard let url = PhotoActions.materialize(path) else { throw LinkError.refused("The file is gone.") }
            let attrs = try FileManager.default.attributesOfItem(atPath: url.path)
            let size = (attrs[.size] as? NSNumber)?.int64Value ?? 0
            let modified = (attrs[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
            let info = LinkWire.FileInfo(size: size, modified: modified, name: url.lastPathComponent)
            let offset = max(0, req.offset ?? 0)
            let length = max(0, min(req.length ?? Self.maxSlice, Self.maxSlice))
            guard length > 0, offset < size else { return try .json(info) }
            let fh = try FileHandle(forReadingFrom: url)
            defer { try? fh.close() }
            try fh.seek(toOffset: UInt64(offset))
            let data = try fh.read(upToCount: length) ?? Data()
            return try .json(info, blob: data)
        }
    }

    private func sources() async throws -> LinkMessage {
        _ = try backend()
        let list: [LinkWire.Source] = await MainActor.run {
            guard let m = AppModel.shared else { return [] }
            let clip = AppModel.clipboardDirectory.path
            let folders = m.roots.filter { $0.path != clip }.map {
                LinkWire.Source(key: $0.path, name: $0.lastPathComponent, kind: "folder", files: m.folderFileCounts[$0.path] ?? 0)
            }
            let photos = m.photoSources.map {
                LinkWire.Source(key: $0.key, name: $0.title, kind: "photos", files: m.folderFileCounts[$0.key] ?? 0)
            }
            return folders + photos
        }
        return try .json(LinkWire.Sources(sources: list))
    }

    /// A folder's indexed children, or a Photos source's assets. Only inside this Mac's sources.
    private func list(_ req: LinkWire.PathRequest) async throws -> LinkMessage {
        let (_, store, _) = try backend()
        let folder = req.path
        let clip = AppModel.clipboardDirectory.path
        guard folder != clip, !folder.hasPrefix(clip + "/") else { throw LinkError.refused("Not shared.") }
        let (inside, photo): (Bool, PhotoLibrary.Source?) = await MainActor.run {
            guard let m = AppModel.shared else { return (false, nil) }
            if let s = m.photoSources.first(where: { $0.key == folder }) { return (true, s) }
            return (m.roots.contains { $0.path != clip && (folder == $0.path || folder.hasPrefix($0.path + "/")) }, nil)
        }
        guard inside else { throw LinkError.refused("Not a shared folder.") }
        if let photo {
            let hits = await AppModel.shared?.photoSourceHits(photo, cap: 1000) ?? []
            return try .json(LinkWire.Listing(children: hits.map {
                LinkWire.Child(path: $0.path, folder: false, kind: $0.kind, modified: $0.modified, size: $0.size, files: 0)
            }))
        }
        let kids = try await offPool { store.indexedChildrenDetailed(ofFolder: folder, aggregates: true) }
        return try .json(LinkWire.Listing(children: kids.map {
            LinkWire.Child(path: $0.path, folder: $0.isDirectory, kind: $0.kind, modified: $0.modified,
                           size: $0.size, files: $0.fileCount)
        }))
    }

    // MARK: - Helpers

    /// Throws unless `path` is a file this Mac's index holds, outside the clipboard history.
    private func servable(_ path: String, _ store: VectorStore) throws {
        let clip = AppModel.clipboardDirectory.path
        guard path.hasPrefix("/") || PhotoLibrary.isPhotoPath(path),
              path != clip, !path.hasPrefix(clip + "/"),
              store.fileStatus(paths: [path])[path] != nil
        else { throw LinkError.refused("Not in that Mac's index.") }
    }

    /// Blocking engine and store work, off the cooperative pool (the same rule Thumbnail's decode
    /// gate keeps: a search must not starve the threads every other task runs on). On the served
    /// work queue, which bounds it with the HTTP and socket surfaces' work, instead of a global
    /// queue that answers blocked work with more threads.
    private func offPool<T: Sendable>(_ work: @escaping @Sendable () throws -> T) async throws -> T {
        try await ServedWork.run(work)
    }

    static func jpeg(_ cg: CGImage, quality: Double = 0.8) -> Data? {
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(out, UTType.jpeg.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, cg, [kCGImageDestinationLossyCompressionQuality: quality] as CFDictionary)
        return CGImageDestinationFinalize(dest) ? out as Data : nil
    }
}

/// Per-peer search lanes: one search at a time per linked Mac, and a search that was overtaken by a
/// newer one from the same Mac while it waited does not run.
actor SearchLanes {
    private var latest: [String: Int] = [:]
    private var busy: Set<String> = []
    private var waiting: [String: [CheckedContinuation<Void, Never>]] = [:]

    func enter(_ peer: String) async -> Int {
        let ticket = (latest[peer] ?? 0) + 1
        latest[peer] = ticket
        if busy.contains(peer) {
            await withCheckedContinuation { waiting[peer, default: []].append($0) }
        }
        busy.insert(peer)
        return ticket
    }

    func isLatest(_ peer: String, _ ticket: Int) -> Bool { latest[peer] == ticket }

    func leave(_ peer: String) {
        if var w = waiting[peer], !w.isEmpty {
            let next = w.removeFirst()
            waiting[peer] = w
            next.resume()     // stays busy: handed straight to the next
        } else {
            busy.remove(peer)
        }
    }
}
