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
    /// Why `available` cannot fill, or nil (LinkNode.discoveryProblem).
    private(set) var discoveryIssue: String?
    /// TEST: what the browser says with Local Network access denied (PerfScript).
    var discoveryIssueForTest: String? { didSet { discoveryIssue = discoveryIssueForTest ?? node?.discoveryIssue } }
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
    /// Why Linked Macs cannot start, while they cannot (no index folder, a key file that will not
    /// read). The Link window says so; Link used to do nothing with no word why.
    private(set) var startFailure: String?
    /// Why the last scoped search reached nothing: "no results" and "could not ask" are not the same.
    var scopeFailure: String?
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
    /// The linked Macs for whoever asks off the main actor: served searches, and the reply cache.
    nonisolated let served = LinkServed()
    @ObservationIgnored private var node: LinkNode?
    @ObservationIgnored private var pairTask: Task<Void, Never>?
    /// Which pairing the sheet is about. A task from an earlier attempt wrote into the current one:
    /// a typed address failing late put its error on the Mac picked after Cancel.
    @ObservationIgnored private var pairAttempt = 0
    @ObservationIgnored private var refreshQueued = false
    /// A linked Mac's facts being learned; each ends known, or gives up only when the Mac goes
    /// offline or stops sharing, and is started again by what ended that.
    @ObservationIgnored private var learning: [String: Task<Void, Never>] = [:]
    /// The other Mac's service generation when its facts were learned.
    @ObservationIgnored private var learnedGeneration: [String: Int] = [:]
    /// Which fetch the status line is about (AppModel.withRemoteCopy). A path alone cannot tell a
    /// fetch from the one started after its Cancel.
    @ObservationIgnored var remoteOpenToken = 0
    /// Called when a linked Mac has come online and its vector space is known: AppModel re-runs
    /// the visible search, so a Mac that comes back joins the results without a keystroke.
    @ObservationIgnored var onPeerReady: ((String) -> Void)?
    /// A linked Mac left the list (unlinked on either side).
    @ObservationIgnored var onPeerGone: ((String) -> Void)?
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

    /// Where the link folder stayed when the index moved before moves carried it along: the folder
    /// the app's index starts in. A test passes `-omni.link.legacyDir`; any other isolated run has no
    /// such place, and must never read the real one.
    private static var legacyDirectory: URL? {
        let args = UserDefaults.standard.volatileDomain(forName: UserDefaults.argumentDomain)
        if let p = args["omni.link.legacyDir"] as? String, !p.isEmpty { return URL(fileURLWithPath: p, isDirectory: true) }
        guard !AppModel.isolatedByLaunchArgument,
              let base = try? FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask,
                                                      appropriateFor: nil, create: false) else { return nil }
        return base.appendingPathComponent("Omni/link", isDirectory: true)
    }

    static var configuredPort: UInt16 {
        let p = UserDefaults.standard.integer(forKey: "omni.link.port")
        return p > 0 && p < 65536 ? UInt16(p) : LinkNode.defaultPort
    }

    /// Bring the node up if this Mac has linked Macs (or `force`, for the Link window). A Mac that
    /// never linked anything never creates an identity, never listens, never browses.
    private func start(force: Bool) {
        guard node == nil else { return }
        guard let dir = Self.linkDirectory() else {
            startFailure = "The index folder is not available."
            return
        }
        // An index moved before moves carried this folder left the key and the peers behind, and the
        // next launch found neither: cut off from every linked Mac, with nothing to say why.
        if let old = Self.legacyDirectory, LinkFile.adopt(from: old, to: dir) {
            omniPerfLog("link: adopted the link folder left at \(old.path)")
        }
        let peersURL = dir.appendingPathComponent("peers.json")
        let store = LinkPeerStore(url: peersURL)
        if let why = store.readFailure { omniPerfLog("link: peers: \(why)") }
        guard force || !store.all.isEmpty else { startFailure = nil; return }
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
            n.onStorageFailure = { [weak self] why in Task { @MainActor in self?.saveFailed(why) } }
            setNode(n)
            startFailure = nil
            n.reconcile()
            // The sheet is open already: a node that arrives under it (the folder came back, the
            // index moved) has to know, or requests hear "Open Link a Mac first" with it on screen.
            if windowOpen { n.setPairingWindow(true) }
            refresh()
        } catch {
            startFailure = LinkController.describe(error)
            omniPerfLog("link: identity failed: \(error)")
        }
    }

    /// The index is here now. `start` runs at init, when the folder may not have been: a volume not
    /// mounted yet at launch left the node down for the session.
    func startIfItCanNow() { start(force: windowOpen) }

    /// The index moved and this folder with it (IndexRelocation.companionFolders). The running node
    /// would go on writing the peers to the old place, and the next launch would read the new one.
    func reopenAfterIndexMove() {
        pairTask?.cancel(); pairTask = nil
        pairAttempt += 1
        node?.retire()
        setNode(nil)
        // `windowOpen`, not `showWindow`: that is a pulse ContentView has already turned back to false.
        start(force: windowOpen)
    }

    private func setNode(_ n: LinkNode?) {
        node = n
        served.attach(n)
        RemoteFiles.shared.attach(n)
    }

    var thisID: String? { node?.identity.id }

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
        let generations = node.serviceGenerations
        // What is known of a linked Mac changes on more than the online edge: a grant switched on, or
        // its service moving under a link that stays up (an engine arriving, a model switch).
        let before = peers
        let wentOnline = p.map(\.id).filter { s[$0] == .online && states[$0] != .online }
        let granted = p.filter { q in q.theyAllowSearch && s[q.id] == .online && before.first { $0.id == q.id }?.theyAllowSearch == false }.map(\.id)
        let moved = p.map(\.id).filter { id in
            guard s[id] == .online, let g = generations[id], let known = learnedGeneration[id] else { return false }
            return g != known
        }
        if peers != p { peers = p }
        if states != s { states = s }
        let a = node.available
        if available != a { available = a }
        let issue = discoveryIssueForTest ?? node.discoveryIssue
        if discoveryIssue != issue { discoveryIssue = issue }
        let port = node.port
        if self.port != port { self.port = port }
        for id in Set(wentOnline + granted) { learn(id) }
        for id in moved where learning[id] == nil { learn(id) }
        let left = Set(before.map(\.id)).subtracting(p.map(\.id))
        let gone = Set(mixable.keys).union(left).subtracting(p.map(\.id))
        for id in gone {
            mixable[id] = nil; fusesText[id] = nil; sources[id] = nil
            learning.removeValue(forKey: id)?.cancel(); learnedGeneration[id] = nil
            RemoteFiles.shared.forget(peer: id)
        }
        for id in left { onPeerGone?(id) }
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

    /// The other Mac's vector space (mix or group?) and its sources (the sidebar), until they are
    /// known. A failure was dropped, so "still starting" or one slow answer lasted as long as the
    /// link did: no mixing, no folder rows, and that Mac embedding every query itself.
    private func learn(_ id: String) {
        guard let node else { return }
        learning[id]?.cancel()
        learning[id] = Task { [weak self] in
            defer { if !Task.isCancelled { self?.learning[id] = nil } }   // cancelled: a newer run holds the entry
            var wait: Double = 2
            while !Task.isCancelled {
                guard let self, self.isOnline(id), self.peer(id)?.theyAllowSearch != false else { return }
                do {
                    let r = try await node.call(id, "space", timeout: 10)
                    let sp = try r.decode(LinkWire.Space.self)
                    guard !Task.isCancelled else { return }
                    self.noteSpace(id, sp)
                    try await self.loadSources(id)
                    return
                } catch {
                    guard !Task.isCancelled else { return }
                    omniPerfLog("link \(id): facts not learned (\(LinkController.describe(error))); again in \(Int(wait)) s")
                    // Not `try? await Task.sleep` as a wait that guards anything: a cancel ends it at
                    // once, and the loop's head then leaves.
                    try? await Task.sleep(for: .seconds(wait))
                    wait = min(wait * 2, 60)
                }
            }
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
        // A reply cached under the old space ranks on the wrong scale.
        if mixable[id] != same || learnedGeneration[id] != sp.generation { served.forgetReplies(peer: id) }
        if mixable[id] != same { mixable[id] = same }
        if fusesText[id] != (sp.fusesText == true) { fusesText[id] = sp.fusesText == true }
        learnedGeneration[id] = sp.generation
        omniPerfLog("link \(id): space dim=\(sp.dim) model=\(sp.model) mixable=\(same)")
        onPeerReady?(id)
    }

    /// Re-check every linked Mac's space (this Mac's index or model changed).
    func localSpaceChanged() { for p in peers where isOnline(p.id) { learn(p.id) } }

    func reloadSources(_ id: String) async { try? await loadSources(id) }

    private func loadSources(_ id: String) async throws {
        guard let node else { return }
        let s = try await node.call(id, "sources", timeout: 10).decode(LinkWire.Sources.self)
        if sources[id] != s.sources { sources[id] = s.sources }
    }

    // MARK: - The Link window and pairing

    func openWindow() {
        start(force: true)
        // The sheet hangs off the main window, and a closed window runs nothing that would show it:
        // File > Link a Mac did nothing, with no sheet and the window flag still false 2 s later.
        if NSApp.windows.first(where: { $0.toolbar != nil })?.isVisible != true { AppDelegate.showMainWindow() }
        showWindow = true
    }

    /// Called by the sheet as it appears and disappears.
    func setWindowOpen(_ open: Bool) {
        guard windowOpen != open else { return }
        windowOpen = open
        node?.setPairingWindow(open)
        if !open {
            pairTask?.cancel(); pairTask = nil
            pairAttempt += 1
            outgoing = nil
            incoming?.cancel(); incoming = nil
            incomingMessage = nil; incomingLinked = nil
        }
    }

    func pair(_ endpoint: NWEndpoint, name: String) {
        guard let node else {
            outgoing = Outgoing(name: name, error: startFailure ?? "Linked Macs could not start.")
            return
        }
        pairTask?.cancel()
        pairAttempt += 1
        let attempt = pairAttempt
        outgoing = Outgoing(name: name)
        let share = alsoShare
        pairTask = Task { [weak self] in
            do {
                let peer = try await node.pair(endpoint, alsoShare: share) { code, name in
                    Task { @MainActor in
                        guard let self, self.pairAttempt == attempt else { return }
                        self.outgoing?.code = code; self.outgoing?.name = name
                    }
                }
                await MainActor.run {
                    self?.refresh()
                    guard self?.pairAttempt == attempt else { return }
                    self?.outgoing = Outgoing(name: peer.name, code: nil, error: nil, linked: true)
                }
            } catch is CancellationError {
            } catch LinkError.cancelled {
            } catch {
                let text = Self.describe(error)
                await MainActor.run {
                    guard let self, self.pairAttempt == attempt else { return }
                    self.outgoing?.error = text; self.outgoing?.code = nil
                }
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
        pairAttempt += 1
        outgoing = nil
    }

    private func present(_ req: IncomingPairing) {
        // It can end between the reveal and here: the other Mac cancelled, or lost the network. A
        // sheet for it would have nothing that could ever close it.
        guard !req.isOver else { return }
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
            case .refused(let s), .remote(let s), .storage(let s), .identity(let s): return s
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
        reportUnsaved(node)
    }

    func setAllowFetch(_ id: String, _ on: Bool) {
        guard let node else { return }
        node.peers.update(id) { p in p.allowFetch = on && p.allowSearch }
        node.pushGrants(id)
        refresh()
        reportUnsaved(node)
    }

    func unlink(_ id: String) {
        guard let node else { return }
        Task {
            await node.unlink(id)
            RemoteFiles.shared.forget(peer: id)
            await MainActor.run {
                self.refresh()
                self.reportUnsaved(node)
            }
        }
    }

    /// A change the disk did not take holds until Omni quits, and then comes back: the user who just
    /// revoked a grant is told, and the store goes on trying (every later write, and here).
    private func reportUnsaved(_ node: LinkNode) {
        guard let why = node.peers.saveFailure else { return }
        saveFailed(why)
        Task { [weak self] in
            for _ in 0 ..< 40 {
                try? await Task.sleep(for: .seconds(30))
                guard self?.node === node else { return }
                if node.peers.retrySave() { return }
            }
        }
    }

    private func saveFailed(_ why: String) {
        omniPerfLog("link: \(why)")
        AppModel.shared?.remoteError = "Linked Macs could not be saved. This change lasts until Omni quits. \(why)"
    }

    func retry(_ id: String) { node?.retry(id) }

    // MARK: - Asking linked Macs

    nonisolated func search(_ req: LinkWire.SearchRequest, peer: String, timeout: Double) async throws -> [SearchHit] {
        try await served.search(req, peer: peer, timeout: timeout).hits
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

/// The linked Macs as code off the main actor sees them, read straight from the node: its peers are
/// behind a lock, and its connection states on a queue that never waits on anything remote.
///
/// NOTHING PER SERVED REQUEST WAITS ON THE MAIN ACTOR. A default MCP or `omni` search asks every
/// linked Mac, so it hopped through AppModel on every request, with no linked Mac at all and even
/// after the sources snapshot had been taken off the main thread: 5.2 s for a served search under a
/// main thread held for 6 s, 19 ms once it was free.
final class LinkServed: @unchecked Sendable {
    private let lock = NSLock()
    private var node: LinkNode?
    private var cache: [String: (at: Date, reply: Reply)] = [:]

    /// One linked Mac's answer: its hits under this Mac's keys for them, what it said of the ones
    /// that are its screenshots (by path on that Mac), and whether it answered a listing as one.
    struct Reply: Sendable {
        var hits: [SearchHit]
        var shots: [String: ScreenHints.Moment] = [:]
        var listed = false
    }

    func attach(_ n: LinkNode?) { lock.withLock { node = n; cache = [:] } }

    /// That Mac's model or index moved: what it answered before ranks on another scale.
    func forgetReplies(peer: String) {
        lock.withLock { cache = cache.filter { !$0.key.hasPrefix(peer + "\u{1}") } }
    }

    fileprivate var current: LinkNode? { lock.withLock { node } }
    var peers: [LinkPeer] { current?.peers.all ?? [] }
    func isOnline(_ id: String) -> Bool { current?.state(of: id) == .online }

    /// Search one linked Mac. A repeat of the same request within a few seconds is answered from
    /// the last reply: a live refresh of this Mac's results (its index changed, not the other's)
    /// must not send the other Mac the same search again.
    func search(_ req: LinkWire.SearchRequest, peer: String, timeout: Double) async throws -> Reply {
        guard let node = current else { throw LinkError.refused("off") }
        let key = peer + "\u{1}" + ((try? JSONEncoder().encode(req)).map { String(decoding: $0, as: UTF8.self) } ?? "")
        let hit: Reply? = lock.withLock {
            guard let c = cache[key], Date().timeIntervalSince(c.at) < 20 else { return nil }
            return c.reply
        }
        if let hit { return hit }
        let wire = try await node.call(peer, "search", .json(req), timeout: timeout).decode(LinkWire.SearchReply.self)
        let reply = Reply(hits: wire.hits.map { $0.hit(peer: peer) }, shots: wire.shots ?? [:], listed: wire.listed == true)
        lock.withLock {
            if cache.count > 64 { cache = cache.filter { Date().timeIntervalSince($0.value.at) < 20 } }
            cache[key] = (Date(), reply)
        }
        return reply
    }
}
