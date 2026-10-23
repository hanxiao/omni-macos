import AppKit
import Foundation
import OmniKit
import OmniLink

/// Linked Macs in the window (docs/notes/link.md): a search also goes to every linked Mac that
/// lets this one search it, and their hits join this Mac's as they arrive.
///
/// THIS MAC'S SEARCH NEVER WAITS FOR ANOTHER ONE. The local search runs exactly as before and its
/// results go on screen when they are ready; a linked Mac's reply is merged in when (and if) it
/// comes. A Mac that is off, asleep or frozen costs a request that fails at once, or one that times
/// out on its own with nothing waiting on it.
extension AppModel {
    /// The search is scoped to a linked Mac (its row in the sidebar), or to a folder on it.
    struct LinkedScope: Equatable {
        var peer: String
        var folder: String?
    }

    /// How long a linked Mac gets to answer one search. Nothing waits on it: past this its results
    /// are simply not part of this query.
    static let remoteSearchTimeout: Double = 10

    /// Whether a search with the current filters can include linked Macs. A scope to a folder on
    /// THIS Mac (or this Mac's Recents, its clipboard, a Photos source) is a question about this
    /// Mac only.
    var searchesLinkedMacs: Bool {
        linkedScope != nil || (filterFolders.isEmpty && !filterRecents && browsedPhotoSource == nil)
    }

    /// Start the linked Macs' half of a search. Called by `search()` after it bumps the token.
    /// `text`: a text query, embedded by each Mac's own model. `vector`: a file query's vector,
    /// sent only to Macs in this Mac's vector space.
    func fanOutRemote(text: String?, vector: [Float]?, filter: SearchFilter, token: Int, exclude: String? = nil) {
        guard searchesLinkedMacs else { return }
        let targets = link.searchable(only: linkedScope?.peer)
        guard !targets.isEmpty else { return }
        let folders = linkedScope?.folder.map { [$0] }
        for peer in targets {
            let mixable = link.mixable[peer.id] == true
            if text == nil, !mixable { continue }   // a file's vector means nothing in another space
            // A text query's vector rides beside the text to a Mac in the same space that fuses the
            // two, which then embeds nothing; any other Mac embeds the text with its own model.
            let v = mixable && (text == nil || link.fusesText[peer.id] == true) ? vector : nil
            var excludeHere: [String]? = nil
            if let e = exclude, let (p, path) = RemotePath.parse(e), p == peer.id { excludeHere = [path] }
            let req = LinkWire.SearchRequest(query: text, vector: v,
                                             filter: LinkWire.Filter(filter, folders: folders),
                                             topK: Self.searchTopK, exclude: excludeHere)
            launchRemote(peer.id, token: token) { [link] in
                try await link.search(req, peer: peer.id, timeout: Self.remoteSearchTimeout)
                    .filter { !filter.rejectsByExtension($0.path) }
            }
        }
    }

    /// Find Similar from a linked Mac's file, on a Mac whose vectors this one cannot use: that Mac
    /// searches with its own stored vector for the file.
    func fanOutSimilarByPath(_ remote: String, token: Int) {
        guard let peer = RemotePath.peer(of: remote), link.mixable[peer] != true, link.isOnline(peer) else { return }
        launchRemote(peer, token: token) { [link] in try await link.similar(remote).hits }
    }

    private func launchRemote(_ peer: String, token: Int, _ work: @escaping @Sendable () async throws -> [SearchHit]) {
        remoteInFlight.insert(peer)
        let t = Task { [weak self] in
            let hits: [SearchHit]?
            do { hits = try await work() } catch { hits = nil }
            guard !Task.isCancelled else { return }
            await MainActor.run { self?.remoteArrived(peer, hits, token: token) }
        }
        remoteTasks.append(t)
    }

    /// A new search: the previous one's remote requests are cancelled (each linked Mac is told, and
    /// drops the work if it has not started it) and their results forgotten.
    func resetRemote() {
        for t in remoteTasks { t.cancel() }
        remoteTasks = []
        remoteHits = [:]
        remoteInFlight = []
        localLanded = false
    }

    private func remoteArrived(_ peer: String, _ hits: [SearchHit]?, token: Int) {
        guard token == currentSearchToken else { return }
        if omniPerfEnabled {
            omniPerfLog(String(format: "remote %@ hits=%d since-search=%.1fms local-landed=%@", peer, hits?.count ?? -1,
                               -searchStartedAt.timeIntervalSinceNow * 1000, localLanded ? "yes" : "no"))
        }
        remoteInFlight.remove(peer)
        if let hits { remoteHits[peer] = hits }
        if linkedScope != nil {
            // Scoped to a linked Mac: there is no local half, so its reply IS the result set.
            guard remoteInFlight.isEmpty else { return }
            finishScopedSearch(token: token)
            return
        }
        guard localLanded, hits != nil else { return }
        let combined = combinedHits()
        if rawResults != combined { rawResults = combined }
    }

    /// The scoped search's end: an empty local half, the linked Mac's hits.
    private func finishScopedSearch(token: Int) {
        guard token == currentSearchToken else { return }
        if searching { searching = false }
        applyScopedResults()
    }

    /// This Mac's hits with the linked Macs' joined: mixed in by score from a Mac in the same vector
    /// space, after them as a group of their own from one that is not. Exactly the local list, in
    /// its own order, when no linked Mac contributed.
    func combinedHits() -> [SearchHit] {
        guard !remoteHits.isEmpty else { return localHits }
        var main = localHits
        var mixedIn = false
        var groups: [[SearchHit]] = []
        for peer in remoteHits.keys.sorted(by: { link.name(of: $0) < link.name(of: $1) }) {
            let hits = remoteHits[peer] ?? []
            if hits.isEmpty { continue }
            if link.mixable[peer] == true { main += hits; mixedIn = true } else { groups.append(hits) }
        }
        if mixedIn {
            // Stable: equal scores keep this Mac's hit first.
            main = main.enumerated().sorted { a, b in
                a.element.score != b.element.score ? a.element.score > b.element.score : a.offset < b.offset
            }.map(\.element)
        }
        return main + groups.flatMap { $0 }
    }

    /// Is this hit in a group of its own (a linked Mac in another vector space)?
    func isGroupedRemote(_ path: String) -> String? {
        guard let peer = RemotePath.peer(of: path), link.mixable[peer] != true else { return nil }
        return peer
    }

    // MARK: - Scope

    /// Browse and search a linked Mac, or one of its folders.
    func enterLinkedMac(_ peer: String, folder: String? = nil) {
        showsClipboardOff = false
        selectFolderForVisualization(nil)
        browsedPhotoSource = nil
        clearLocalScopeQuietly()
        let scope = LinkedScope(peer: peer, folder: folder)
        if linkedScope != scope { linkedScope = scope }
        if hasQuery || fileQuery != nil { search() } else { clearResultsQuietly() }
    }

    func leaveLinkedScope() {
        guard linkedScope != nil else { return }
        linkedScope = nil
        if hasQuery { search() }
    }

    // MARK: - Served search (the MCP tool's `macs`, and `omni search --macs`)

    /// The same search on the named linked Macs ("all" for every one), each answering from its own
    /// index. Paths come back as paths on that Mac. A folder scope is a folder on THIS Mac, which
    /// means nothing on another, so a scoped search does not go out.
    func linkedSearchForServing(_ q: String, filter: SearchFilter, topK: Int, macs: [String], explicit: Bool) async -> [LinkedSearchAnswer] {
        let everyone = macs.contains { $0.caseInsensitiveCompare("all") == .orderedSame }
        let peers = link.peers.filter { p in
            everyone || macs.contains { $0.caseInsensitiveCompare(p.name) == .orderedSame || $0 == p.id }
        }
        var out: [LinkedSearchAnswer] = []
        for name in macs where !everyone && !link.peers.contains(where: { $0.name.caseInsensitiveCompare(name) == .orderedSame || $0.id == name }) {
            let known = link.peers.map(\.name).joined(separator: ", ")
            out.append(LinkedSearchAnswer(mac: name, hits: [], note: known.isEmpty
                ? "no Mac is linked with this one (File > Link a Mac)"
                : "no linked Mac is called \(name); linked: \(known)"))
        }
        guard filter.folderPrefixes.isEmpty else {
            return explicit ? out + peers.map { LinkedSearchAnswer(mac: $0.name, hits: [], note: "a folder scope applies to this Mac only") } : out
        }
        await withTaskGroup(of: LinkedSearchAnswer.self) { group in
            for p in peers {
                let online = link.isOnline(p.id), shared = p.theyAllowSearch, link = self.link
                let req = LinkWire.SearchRequest(query: q, vector: nil, filter: LinkWire.Filter(filter, folders: nil),
                                                 topK: topK, exclude: nil)
                group.addTask {
                    guard online else { return LinkedSearchAnswer(mac: p.name, hits: [], note: "offline") }
                    guard shared else { return LinkedSearchAnswer(mac: p.name, hits: [], note: "does not share its files with this Mac") }
                    do {
                        let hits = try await link.search(req, peer: p.id, timeout: Self.servedRemoteTimeout)
                            .filter { !filter.rejectsByExtension($0.path) }
                        return LinkedSearchAnswer(mac: p.name, hits: hits.map { h in
                            SearchHit.make(path: RemotePath.parse(h.path)?.path ?? h.path, score: h.score, snippet: h.snippet,
                                           kind: h.kind, chunkIndex: h.chunkIndex, modified: h.modified, width: h.width,
                                           height: h.height, duration: h.duration, size: h.size, locator: h.locator,
                                           chunkCount: h.chunkCount)
                        }, note: nil, canOpen: p.theyAllowFetch)
                    } catch {
                        return LinkedSearchAnswer(mac: p.name, hits: [], note: LinkController.describe(error))
                    }
                }
            }
            for await a in group { out.append(a) }
        }
        return out.sorted { $0.mac.localizedCaseInsensitiveCompare($1.mac) == .orderedAscending }
    }

    /// An agent waits for the answer, so a served search gives linked Macs less time than the
    /// window, which never waits at all. Replies measured ~80 ms on a LAN-less loopback pair.
    static let servedRemoteTimeout: Double = 3

    /// `get_linked_file`: copy a linked Mac's file (a path from its search results) here, read-only,
    /// and return where. The same fetch Open and Quick Look use, under the same grant.
    func linkedFetchForServing(_ mac: String, path: String) async -> (path: String?, error: String?) {
        guard let p = link.peers.first(where: { $0.name.caseInsensitiveCompare(mac) == .orderedSame || $0.id == mac }) else {
            let known = link.peers.map(\.name).joined(separator: ", ")
            return (nil, known.isEmpty ? "no Mac is linked with this one" : "no linked Mac is called \(mac); linked: \(known)")
        }
        guard link.isOnline(p.id) else { return (nil, "\(p.name) is offline") }
        guard p.theyAllowFetch else { return (nil, "\(p.name) does not allow opening its files") }
        do {
            return (try await RemoteFiles.shared.fetch(RemotePath.make(peer: p.id, path: path)).path, nil)
        } catch {
            return (nil, LinkController.describe(error))
        }
    }

    // MARK: - Acting on a linked Mac's file

    /// Fetch, then hand the local copy to `then`. While it fetches the window says so.
    func withRemoteCopy(_ remote: String, _ then: @escaping @MainActor (URL) -> Void) {
        guard remoteOpening != remote else { return }
        remoteOpening = remote
        Task { @MainActor in
            defer { if remoteOpening == remote { remoteOpening = nil } }
            do {
                let url = try await RemoteFiles.shared.fetch(remote)
                then(url)
            } catch {
                guard remoteOpening == remote else { return }   // cancelled by a newer one
                let name = (remote as NSString).lastPathComponent
                remoteError = "Couldn\u{2019}t open \u{201C}\(name)\u{201D}. \(LinkController.describe(error))"
            }
        }
    }

    func openRemote(_ remote: String) {
        let h = searchHit(for: remote)
        withRemoteCopy(remote) { url in
            if let h { OpenAtHit.open(url.path, locator: h.locator, snippet: h.snippet) }
            else { NSWorkspace.shared.openAsync(url) }
        }
    }

    func previewRemote(_ remote: String) {
        withRemoteCopy(remote) { [weak self] url in
            // Still the file the user is looking at: arrowing on while it fetched moved the preview on.
            guard let self, self.selection == remote || self.selection == nil else { return }
            self.showPreview(url)
        }
    }

    func searchBySimilarRemote(_ remote: String) {
        withRemoteCopy(remote) { [weak self] url in
            self?.setFileQuery(url, similar: true, transient: true, sourcePath: remote)
        }
    }

    func passagesRemote(_ remote: String) async -> [ChunkHit] {
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        return await link.passages(remote, query: q.isEmpty ? nil : q, vector: lastQueryVectorForLink)
    }
}

/// "Is this result a file on this Mac?" in one place: a Photos asset and a linked Mac's file are not.
enum ResultPath {
    static func isRemote(_ p: String) -> Bool { RemotePath.isRemote(p) }
    /// Not a file on this Mac's disk: a Photos asset or a linked Mac's file.
    static func isVirtual(_ p: String) -> Bool { PhotoLibrary.isPhotoPath(p) || RemotePath.isRemote(p) }
}
