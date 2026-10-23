import AppKit
import Foundation
import Network
import OmniKit
import OmniLink

/// Linked Macs on this Mac (docs/notes/link.md): owns the LinkNode, keeps the state the sidebar,
/// the Link window and Settings show, and asks linked Macs for searches, listings and passages on
/// behalf of AppModel.
///
/// NOTHING HERE WAITS ON A LINKED MAC FROM THE MAIN ACTOR. Every request is an async call that
/// fails at once for a Mac that is not connected, and the node's own work runs on its queue. The
/// rest of the app sees a linked Mac only through results arriving late, or not at all.
@MainActor @Observable
final class LinkController {
    static private(set) var shared: LinkController!

    /// A pairing this Mac started: the code to show while the other Mac's user types it.
    struct Outgoing: Equatable {
        var name: String
        var code: String?
        var error: String?
        var linked = false
    }

    private(set) var peers: [LinkPeer] = []
    private(set) var states: [String: LinkNode.PeerState] = [:]
    private(set) var available: [LinkNode.Discovered] = []
    private(set) var port: UInt16?
    private(set) var windowOpen = false
    /// Whether this Mac is in the same vector space as each linked Mac (nil until known). Same
    /// space: its hits mix into this Mac's list by score. Otherwise they are a group of their own.
    private(set) var mixable: [String: Bool] = [:]
    /// Whether each linked Mac fuses a text query's vector with its text (`Space.fusesText`), so a
    /// text search can carry the vector this Mac already holds instead of making it embed again.
    private(set) var fusesText: [String: Bool] = [:]
    /// Each linked Mac's sources, as it last reported them.
    private(set) var sources: [String: [LinkWire.Source]] = [:]
    var incoming: IncomingPairing?
    var incomingMessage: String?
    /// This Mac just accepted a link from the Mac named here (its sheet shows the success view).
    var incomingLinked: String?
    var outgoing: Outgoing?
    var showWindow = false
    /// Grant the Macs this one links with search (and file access) on this Mac too.
    var alsoShare = true
    let name: String
    let model: String

    @ObservationIgnored let responder = LinkResponder()
    @ObservationIgnored private var node: LinkNode?
    @ObservationIgnored private var pairTask: Task<Void, Never>?
    @ObservationIgnored private var cache: [String: (at: Date, hits: [SearchHit])] = [:]
    @ObservationIgnored private var refreshQueued = false
    /// Called when a linked Mac has come online and its vector space is known: AppModel re-runs
    /// the visible search, so a Mac that comes back joins the results without a keystroke.
    @ObservationIgnored var onPeerReady: ((String) -> Void)?
    /// The local index's space probe, read by AppModel when the index opens.
    @ObservationIgnored var localSpace: (dim: Int, probe: [Float]?) = (0, nil)

    init() {
        // The computer's name, as Finder and AirDrop show it; `omni.link.name` overrides it (two
        // test instances on one Mac would otherwise both be called the same thing).
        name = UserDefaults.standard.string(forKey: "omni.link.name").flatMap { $0.isEmpty ? nil : $0 }
            ?? Host.current().localizedName ?? "Mac"
        model = Self.hardwareModel()
        LinkController.shared = self
        start(force: false)
    }

    static func hardwareModel() -> String {
        var size = 0
        sysctlbyname("hw.model", nil, &size, nil, 0)
        var buf = [CChar](repeating: 0, count: max(size, 1))
        sysctlbyname("hw.model", &buf, &size, nil, 0)
        return String(cString: buf)
    }

    static func linkDirectory() -> URL? {
        if let custom = UserDefaults.standard.string(forKey: "omni.link.dir"), !custom.isEmpty {
            return URL(fileURLWithPath: custom, isDirectory: true)
        }
        return (try? AppModel.indexURL())?.deletingLastPathComponent().appendingPathComponent("link", isDirectory: true)
    }

    static var configuredPort: UInt16 {
        let p = UserDefaults.standard.integer(forKey: "omni.link.port")
        return p > 0 && p < 65536 ? UInt16(p) : LinkNode.defaultPort
    }

    /// Bring the node up if this Mac has linked Macs (or `force`, for the Link window). A Mac that
    /// never linked anything never creates an identity, never listens, never browses.
    private func start(force: Bool) {
        guard node == nil, let dir = Self.linkDirectory() else { return }
        let peersURL = dir.appendingPathComponent("peers.json")
        let store = LinkPeerStore(url: peersURL)
        guard force || !store.all.isEmpty else { return }
        do {
            let identity = try LinkIdentity.loadOrCreate(at: dir.appendingPathComponent("identity"))
            let version = (Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String) ?? "dev"
            let n = LinkNode(identity: identity, peers: store,
                             config: .init(name: name, model: model, version: version, port: Self.configuredPort,
                                           bonjour: UserDefaults.standard.object(forKey: "omni.link.bonjour") == nil
                                               || UserDefaults.standard.bool(forKey: "omni.link.bonjour")),
                             service: responder)
            n.onChange = { [weak self] in Task { @MainActor in self?.scheduleRefresh() } }
            n.onLog = { line in if omniPerfEnabled { omniPerfLog("link \(line)") } }
            n.onPairingRequest = { [weak self] req in Task { @MainActor in self?.present(req) } }
            node = n
            RemoteFiles.shared.attach(n)
            n.reconcile()
            refresh()
        } catch {
            omniPerfLog("link: identity failed: \(error)")
        }
    }

    var thisID: String? { node?.identity.id }

    // MARK: - State

    private func scheduleRefresh() {
        guard !refreshQueued else { return }
        refreshQueued = true
        DispatchQueue.main.async { [weak self] in
            MainActor.assumeIsolated {
                self?.refreshQueued = false
                self?.refresh()
            }
        }
    }

    private func refresh() {
        guard let node else { return }
        let p = node.peers.all.sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
        let s = node.states
        let wentOnline = p.map(\.id).filter { s[$0] == .online && states[$0] != .online }
        if peers != p { peers = p }
        if states != s { states = s }
        let a = node.available
        if available != a { available = a }
        let port = node.port
        if self.port != port { self.port = port }
        for id in wentOnline { learn(id) }
        let gone = Set(mixable.keys).subtracting(p.map(\.id))
        for id in gone { mixable[id] = nil; fusesText[id] = nil; sources[id] = nil; RemoteFiles.shared.forget(peer: id) }
    }

    func state(_ id: String) -> LinkNode.PeerState { states[id] ?? .offline }
    func isOnline(_ id: String) -> Bool { states[id] == .online }
    func peer(_ id: String) -> LinkPeer? { peers.first { $0.id == id } }
    func name(of id: String) -> String { peer(id)?.name ?? "Linked Mac" }

    /// Files on the Macs this one may search right now, as their sources last reported them.
    var searchableFiles: Int {
        searchable().reduce(0) { n, p in n + (sources[p.id] ?? []).reduce(0) { $0 + $1.files } }
    }

    /// Macs this one may search right now.
    func searchable(only: String? = nil) -> [LinkPeer] {
        peers.filter { $0.theyAllowSearch && isOnline($0.id) && (only == nil || $0.id == only) }
    }

    /// On connect: the other Mac's vector space (mix or group?) and its sources (the sidebar).
    private func learn(_ id: String) {
        guard let node else { return }
        Task { [weak self] in
            if let r = try? await node.call(id, "space", timeout: 10), let sp = try? r.decode(LinkWire.Space.self) {
                await MainActor.run { self?.noteSpace(id, sp) }
            }
            await self?.reloadSources(id)
        }
    }

    private func noteSpace(_ id: String, _ sp: LinkWire.Space) {
        let mine = localSpace
        var same = false
        if sp.dim == mine.dim, let a = sp.probe, let b = mine.probe, a.count == b.count {
            var dot: Float = 0, na: Float = 0, nb: Float = 0
            for i in 0 ..< a.count { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
            same = dot / max((na * nb).squareRoot(), 1e-12) >= 0.999
        }
        if mixable[id] != same { mixable[id] = same }
        if fusesText[id] != (sp.fusesText == true) { fusesText[id] = sp.fusesText == true }
        omniPerfLog("link \(id): space dim=\(sp.dim) model=\(sp.model) mixable=\(same)")
        onPeerReady?(id)
    }

    /// Re-check every linked Mac's space (this Mac's index or model changed).
    func localSpaceChanged() { for p in peers where isOnline(p.id) { learn(p.id) } }

    func reloadSources(_ id: String) async {
        guard let node, let r = try? await node.call(id, "sources", timeout: 10),
              let s = try? r.decode(LinkWire.Sources.self) else { return }
        if sources[id] != s.sources { sources[id] = s.sources }
    }

    // MARK: - The Link window and pairing

    func openWindow() {
        start(force: true)
        showWindow = true
    }

    /// Called by the sheet as it appears and disappears.
    func setWindowOpen(_ open: Bool) {
        guard windowOpen != open else { return }
        windowOpen = open
        node?.setPairingWindow(open)
        if !open {
            pairTask?.cancel(); pairTask = nil
            outgoing = nil
            incoming?.cancel(); incoming = nil
            incomingMessage = nil; incomingLinked = nil
        }
    }

    func pair(_ endpoint: NWEndpoint, name: String) {
        guard let node else { return }
        pairTask?.cancel()
        outgoing = Outgoing(name: name)
        let share = alsoShare
        pairTask = Task { [weak self] in
            do {
                let peer = try await node.pair(endpoint, alsoShare: share) { code, name in
                    Task { @MainActor in self?.outgoing?.code = code; self?.outgoing?.name = name }
                }
                await MainActor.run {
                    self?.outgoing = Outgoing(name: peer.name, code: nil, error: nil, linked: true)
                    self?.refresh()
                }
            } catch is CancellationError {
            } catch LinkError.cancelled {
            } catch {
                let text = Self.describe(error)
                await MainActor.run { self?.outgoing?.error = text; self?.outgoing?.code = nil }
            }
        }
    }

    func pair(address: String) {
        guard let ep = LinkNode.endpoint(forAddress: address) else {
            outgoing = Outgoing(name: address, error: "Not an address.")
            return
        }
        pair(ep, name: address)
    }

    func cancelPairing() {
        pairTask?.cancel(); pairTask = nil
        outgoing = nil
    }

    private func present(_ req: IncomingPairing) {
        incoming?.cancel()
        incoming = req
        incomingMessage = nil; incomingLinked = nil
        showWindow = true
        req.onEnd = { [weak self] outcome in
            Task { @MainActor in
                guard let self, self.incoming?.id == req.id else { return }
                self.incoming = nil
                switch outcome {
                case .linked: self.incomingMessage = "Linked with \(req.name)"; self.incomingLinked = req.name
                case .wrong: self.incomingMessage = "Wrong code"
                case .cancelled: self.incomingMessage = nil
                }
                self.refresh()
            }
        }
    }

    /// The code typed for an incoming request. Returns what to tell the user, or nil when linked.
    func submit(_ typed: String) -> String? {
        guard let req = incoming else { return nil }
        switch req.submit(typed) {
        case .linked: return nil
        case .wrong(let left): return "Wrong code. \(left) \(left == 1 ? "try" : "tries") left."
        case .over: return "Wrong code."
        }
    }

    func declineIncoming() { incoming?.cancel() }

    nonisolated static func describe(_ error: Error) -> String {
        switch error {
        case let e as LinkError:
            switch e {
            case .refused(let s), .remote(let s): return s
            case .timeout: return "No answer."
            case .closed: return "Connection closed."
            default: return e.description
            }
        case let e as NWError:
            switch e {
            case .posix(let code) where code == .ECONNREFUSED: return "Open Link a Mac on that Mac first."
            case .posix(let code) where code == .ETIMEDOUT: return "No answer."
            case .dns: return "Not found."
            default: return e.localizedDescription
            }
        default:
            return error.localizedDescription
        }
    }

    // MARK: - Grants and unlinking

    func setAllowSearch(_ id: String, _ on: Bool) {
        guard let node else { return }
        node.peers.update(id) { p in
            p.allowSearch = on
            if !on { p.allowFetch = false }
        }
        node.pushGrants(id)
        refresh()
    }

    func setAllowFetch(_ id: String, _ on: Bool) {
        guard let node else { return }
        node.peers.update(id) { p in p.allowFetch = on && p.allowSearch }
        node.pushGrants(id)
        refresh()
    }

    func unlink(_ id: String) {
        guard let node else { return }
        Task {
            await node.unlink(id)
            RemoteFiles.shared.forget(peer: id)
            await MainActor.run { self.refresh() }
        }
    }

    func retry(_ id: String) { node?.retry(id) }

    // MARK: - Asking linked Macs

    /// Search one linked Mac. A repeat of the same request within a few seconds is answered from
    /// the last reply: a live refresh of this Mac's results (its index changed, not the other's)
    /// must not send the other Mac the same search again.
    func search(_ req: LinkWire.SearchRequest, peer: String, timeout: Double) async throws -> [SearchHit] {
        guard let node else { throw LinkError.refused("off") }
        let key = peer + "\u{1}" + ((try? JSONEncoder().encode(req)).map { String(decoding: $0, as: UTF8.self) } ?? "")
        if let c = cache[key], Date().timeIntervalSince(c.at) < 20 { return c.hits }
        let reply = try await node.call(peer, "search", .json(req), timeout: timeout).decode(LinkWire.SearchReply.self)
        let hits = reply.hits.map { $0.hit(peer: peer) }
        if cache.count > 64 { cache = cache.filter { Date().timeIntervalSince($0.value.at) < 20 } }
        cache[key] = (Date(), hits)
        return hits
    }

    func similar(_ remote: String) async throws -> (hits: [SearchHit], vector: [Float]?) {
        guard let node, let (peer, path) = RemotePath.parse(remote) else { throw LinkError.refused("off") }
        let r = try await node.call(peer, "similar", .json(LinkWire.PathRequest(path: path)), timeout: 30)
            .decode(LinkWire.Similar.self)
        return (r.hits.map { $0.hit(peer: peer) }, r.vector)
    }

    func passages(_ remote: String, query: String?, vector: [Float]?) async -> [ChunkHit] {
        guard let node, let (peer, path) = RemotePath.parse(remote) else { return [] }
        let v = mixable[peer] == true ? vector : nil
        guard let r = try? await node.call(peer, "passages",
                                           .json(LinkWire.PathRequest(path: path, query: query, vector: v)), timeout: 30)
            .decode(LinkWire.Passages.self) else { return [] }
        return r.passages.map { ChunkHit.make(chunkIndex: $0.chunk, score: $0.score, snippet: $0.snippet, locator: $0.loc) }
    }

    func list(_ peer: String, folder: String) async throws -> [LinkWire.Child] {
        guard let node else { throw LinkError.refused("off") }
        return try await node.call(peer, "list", .json(LinkWire.PathRequest(path: folder)), timeout: 30)
            .decode(LinkWire.Listing.self).children
    }

    /// Where a linked Mac's file lives, for a result row: "Studio › ~/Documents/Taxes".
    func location(_ remote: String) -> String {
        guard let (peer, path) = RemotePath.parse(remote) else { return remote }
        var dir = (path as NSString).deletingLastPathComponent
        if PhotoLibrary.isPhotoPath(path) {
            dir = "Photos"
        } else {
            let parts = dir.split(separator: "/", omittingEmptySubsequences: false)
            if parts.count >= 3, parts[1] == "Users" {
                dir = "~" + (parts.count > 3 ? "/" + parts[3...].joined(separator: "/") : "")
            }
        }
        return "\(name(of: peer)) \u{203A} \(dir)"
    }
}
