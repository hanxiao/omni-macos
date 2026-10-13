import Foundation
import Network

/// What this Mac answers a linked Mac with. Implemented by the app (search, thumbnails, files). Only
/// called for a linked peer that this Mac lets search it; `fetchMethods` additionally need the
/// peer's "open files" grant. The peer passed in is read at the moment of the call, so a grant
/// switched off takes effect on the next request.
public protocol LinkService: Sendable {
    func serve(_ method: String, _ msg: LinkMessage, peer: LinkPeer) async throws -> LinkMessage
}

/// One Mac on the link: listens, advertises itself over Bonjour, keeps a connection to every Mac
/// it is linked with, and runs both sides of pairing.
///
/// IT IS SILENT UNLESS USED. Nothing listens, advertises or browses while this Mac has no linked
/// Mac and its Link window is closed (`active`): Omni works with the network cable pulled, and an
/// update must not start announcing every user's Mac on their network.
///
/// All mutable state lives on `queue`. Callbacks fire on `queue`; the app hops to its own actor.
public final class LinkNode: @unchecked Sendable {
    public static let serviceType = "_omni-link._tcp"
    public static let defaultPort: UInt16 = 51298
    /// Methods that hand out file contents, and so need the "open files" grant on top of search.
    public static let fetchMethods: Set<String> = ["fetch"]
    /// Liveness: a ping every `pingInterval` s, and a Mac that does not answer within `pingTimeout`
    /// is dropped. Variables so the tests can run them in a second rather than half a minute.
    nonisolated(unsafe) static var pingInterval: Double = 15
    nonisolated(unsafe) static var pingTimeout: Double = 5
    /// How long a connection gets to come up (TCP and TLS).
    nonisolated(unsafe) static var connectTimeout: Double = 10

    public enum PeerState: Equatable, Sendable {
        case offline
        case connecting
        case online
        /// The other Mac no longer knows this one (it was unlinked there).
        case unlinked
    }

    /// A Mac found on the network with its Link window open.
    public struct Discovered: Equatable, Sendable, Identifiable {
        public let id: String
        public let name: String
        public let endpoint: NWEndpoint
    }

    public struct Config: Sendable {
        public var name: String
        public var model: String
        public var version: String
        public var port: UInt16
        /// Advertise and browse with Bonjour. Off, a Mac is reached by address only (an isolated
        /// test instance on loopback, which local network privacy does not cover).
        public var bonjour: Bool
        public init(name: String, model: String, version: String, port: UInt16 = LinkNode.defaultPort, bonjour: Bool = true) {
            self.name = name; self.model = model; self.version = version; self.port = port; self.bonjour = bonjour
        }
    }

    public let identity: LinkIdentity
    public let peers: LinkPeerStore
    public let queue = DispatchQueue(label: "omni.link")
    private let service: LinkService
    private var config: Config

    // Callbacks (fire on `queue`).
    public var onChange: (@Sendable () -> Void)?
    public var onPairingRequest: (@Sendable (IncomingPairing) -> Void)?
    public var onLog: (@Sendable (String) -> Void)?

    private var listener: NWListener?
    private var browser: NWBrowser?
    private var listening: UInt16?
    private var windowOpen = false
    private var running = false
    private var discovered: [String: Discovered] = [:]       // by peer id; every Omni seen, open window or not
    private var conns: [String: PeerConn] = [:]              // outgoing, by peer id
    private var incoming: [ObjectIdentifier: LinkChannel] = [:]
    private var pairingInProgress = false

    private final class PeerConn {
        var state: PeerState = .offline
        var channel: LinkChannel?
        var attempt: NWConnection?
        var backoff: Double = 2
        var retry: DispatchWorkItem?
        var ping: DispatchSourceTimer?
        var generation = 0
    }

    public init(identity: LinkIdentity, peers: LinkPeerStore, config: Config, service: LinkService) {
        self.identity = identity
        self.peers = peers
        self.config = config
        self.service = service
    }

    // MARK: - Public surface

    /// Whether the link should be up at all: some Mac is linked, or the Link window is open.
    public var active: Bool { queue.sync { windowOpen || !peers.all.isEmpty } }

    /// The port actually listened on (nil while down).
    public var port: UInt16? { queue.sync { listening } }

    public func state(of id: String) -> PeerState { queue.sync { conns[id]?.state ?? .offline } }
    public var states: [String: PeerState] { queue.sync { conns.mapValues(\.state) } }

    /// Macs with their Link window open that are not linked yet, nearest name first.
    public var available: [Discovered] {
        queue.sync {
            let linked = Set(peers.all.map(\.id))
            return discovered.values
                .filter { $0.id != identity.id && !linked.contains($0.id) && $0.name != $0.id }
                .sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
        }
    }

    public func setName(_ name: String) {
        queue.async { [self] in
            guard config.name != name else { return }
            config.name = name
            advertise()
        }
    }

    /// Bring the link up or down to match `active`. Call after anything that changes it.
    public func reconcile() {
        queue.async { [self] in reconcileLocked() }
    }

    /// Open or close the Link window: while open this Mac advertises its name and accepts pairing.
    public func setPairingWindow(_ open: Bool) {
        queue.async { [self] in
            guard windowOpen != open else { return }
            windowOpen = open
            reconcileLocked()
            advertise()
            notify()
        }
    }

    public func stop() {
        queue.sync { stopLocked() }
    }

    /// Call `method` on a linked Mac. Fails at once when it is not connected: a search never waits
    /// for a Mac to come back.
    public func call(_ id: String, _ method: String, _ msg: LinkMessage = LinkMessage(), timeout: Double) async throws -> LinkMessage {
        let ch: LinkChannel? = queue.sync { conns[id]?.state == .online ? conns[id]?.channel : nil }
        guard let ch else { throw LinkError.refused("offline") }
        return try await ch.call(method, msg, timeout: timeout)
    }

    /// Reconnect now (a Mac reappeared, or the user asked).
    public func retry(_ id: String) {
        queue.async { [self] in
            guard let c = conns[id], c.state == .offline || c.state == .unlinked else { return }
            if c.state == .unlinked { c.state = .offline }
            dial(id)
        }
    }

    /// Forget a linked Mac, and tell it so if it is reachable, so it forgets this one too.
    public func unlink(_ id: String) async {
        let ch: LinkChannel? = queue.sync { conns[id]?.channel }
        if let ch { _ = try? await ch.call("unlink", timeout: 3) }
        queue.sync {
            dropConn(id)
            peers.remove(id)
            reconcileLocked()
            notify()
        }
    }

    /// Pair with a Mac whose Link window is open. `showCode` is called with the six digits to show,
    /// and that Mac's name, while its user types them; returns once they did (or it failed).
    /// `alsoShare`: let that Mac search (and open files on) this one as well.
    public func pair(_ endpoint: NWEndpoint, alsoShare: Bool,
                     showCode: @escaping @Sendable (_ code: String, _ name: String) -> Void) async throws -> LinkPeer {
        let seen = Locked<Data?>(nil)
        let params = LinkTLS.parameters(identity: identity, queue: queue) { key in seen.set(key); return true }
        let conn = NWConnection(to: endpoint, using: params)
        let ch = try await connect(conn, timeout: 10)
        defer { ch.close() }
        guard let serverKey = LinkTLS.peerKey(conn) ?? seen.get() else { throw LinkError.protocolViolation("no server key") }
        guard serverKey != identity.publicKey else { throw LinkError.refused("That's this Mac.") }
        guard let exporter = LinkTLS.exporter(conn) else { throw LinkError.protocolViolation("no exporter") }
        let nonce = LinkPairing.nonce()
        let begin = try await ch.call("pair.begin", .json(PairBegin(name: config.name, model: config.model,
                                                                   commit: LinkPairing.commitment(nonce),
                                                                   share: alsoShare)), timeout: 10)
            .decode(PairBeginReply.self)
        let code = LinkPairing.code(exporter: exporter, clientKey: identity.publicKey, serverKey: serverKey,
                                    clientNonce: nonce, serverNonce: begin.nonce)
        showCode(code, begin.name)
        let result = try await ch.call("pair.reveal", .json(PairReveal(nonce: nonce)), timeout: IncomingPairing.lifetime + 10)
            .decode(PairResult.self)
        guard result.ok else { throw LinkError.refused(result.reason ?? "Wrong code.") }
        var host: String?, port: UInt16?
        if case .hostPort(let h, let p) = endpoint { host = "\(h)"; port = p.rawValue }
        let peer = LinkPeer(id: LinkIdentity.id(of: serverKey), publicKey: serverKey, name: begin.name,
                            model: begin.model, host: host, port: port,
                            allowSearch: alsoShare, allowFetch: alsoShare,
                            theyAllowSearch: true, theyAllowFetch: true)
        peers.put(peer)
        queue.async { [self] in
            reconcileLocked()
            dial(peer.id)
            notify()
        }
        return peer
    }

    /// An endpoint for an address the user typed: "host", "host:port", "[v6]:port".
    public static func endpoint(forAddress raw: String) -> NWEndpoint? {
        var s = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !s.isEmpty else { return nil }
        var port = defaultPort
        if s.hasPrefix("[") , let close = s.firstIndex(of: "]") {
            let host = String(s[s.index(after: s.startIndex) ..< close])
            let rest = s[s.index(after: close)...]
            if rest.hasPrefix(":"), let p = UInt16(rest.dropFirst()) { port = p }
            return .hostPort(host: NWEndpoint.Host(host), port: NWEndpoint.Port(rawValue: port)!)
        }
        if s.filter({ $0 == ":" }).count == 1, let colon = s.lastIndex(of: ":") {
            guard let p = UInt16(s[s.index(after: colon)...]), p > 0 else { return nil }
            port = p
            s = String(s[..<colon])
        }
        guard !s.isEmpty, let p = NWEndpoint.Port(rawValue: port) else { return nil }
        return .hostPort(host: NWEndpoint.Host(s), port: p)
    }

    // MARK: - Up and down

    private func reconcileLocked() {
        let want = windowOpen || !peers.all.isEmpty
        if want && !running { startLocked() }
        if !want && running { stopLocked() }
        guard running else { return }
        // Dial every linked Mac not already connected; drop connections to Macs no longer linked.
        let ids = Set(peers.all.map(\.id))
        for id in conns.keys where !ids.contains(id) { dropConn(id) }
        for id in ids where conns[id] == nil { conns[id] = PeerConn(); dial(id) }
    }

    private func startLocked() {
        running = true
        startListener(port: config.port)
        if config.bonjour { startBrowser() }
    }

    private func stopLocked() {
        running = false
        listener?.cancel(); listener = nil; listening = nil
        browser?.cancel(); browser = nil
        discovered = [:]
        for id in conns.keys { dropConn(id) }
        for ch in incoming.values { ch.close() }
        incoming = [:]
        notify()
    }

    private func startListener(port: UInt16, fallback: Bool = true) {
        let params = LinkTLS.parameters(identity: identity, queue: queue) { _ in true }
        params.allowLocalEndpointReuse = true
        let l: NWListener
        do {
            l = try port == 0 ? NWListener(using: params) : NWListener(using: params, on: NWEndpoint.Port(rawValue: port)!)
        } catch {
            log("listener: \(error)")
            if fallback { startListener(port: 0, fallback: false) }
            return
        }
        listener = l
        if config.bonjour { l.service = bonjourService() }
        l.stateUpdateHandler = { [weak self, weak l] state in
            guard let self, let l, self.listener === l else { return }
            switch state {
            case .ready:
                self.listening = l.port?.rawValue
                self.log("listening on \(self.listening ?? 0)")
                self.notify()
            case .failed(let e):
                self.log("listener failed: \(e)")
                l.cancel()
                self.listener = nil; self.listening = nil
                // The fixed port is taken (a second copy of Omni, say): any port still works over
                // Bonjour, and the Link window shows which one for typing an address.
                if fallback, self.running { self.startListener(port: 0, fallback: false) }
                self.notify()
            default: break
            }
        }
        l.newConnectionHandler = { [weak self] c in self?.accept(c) }
        l.start(queue: queue)
    }

    private func bonjourService() -> NWListener.Service {
        NWListener.Service(name: windowOpen ? config.name : identity.id, type: Self.serviceType,
                           domain: nil, txtRecord: NWTXTRecord(["id": identity.id, "v": "1"]))
    }

    private func advertise() {
        guard config.bonjour else { return }
        listener?.service = bonjourService()
    }

    private func startBrowser() {
        let b = NWBrowser(for: .bonjourWithTXTRecord(type: Self.serviceType, domain: nil), using: .tcp)
        browser = b
        b.browseResultsChangedHandler = { [weak self, weak b] results, _ in
            guard let self, let b, self.browser === b else { return }
            var next: [String: Discovered] = [:]
            for r in results {
                guard case .service(let name, _, _, _) = r.endpoint,
                      case .bonjour(let txt) = r.metadata, let id = txt["id"], id != self.identity.id else { continue }
                // One Mac shows up once per interface; keep the first.
                if next[id] == nil { next[id] = Discovered(id: id, name: name, endpoint: r.endpoint) }
            }
            let appeared = Set(next.keys).subtracting(self.discovered.keys)
            self.discovered = next
            // A linked Mac that just appeared is dialled now, not at the end of its backoff.
            for id in appeared where self.conns[id]?.state == .offline { self.dial(id) }
            self.notify()
        }
        b.stateUpdateHandler = { [weak self] s in
            if case .failed(let e) = s { self?.log("browser failed: \(e)") }
        }
        b.start(queue: queue)
    }

    // MARK: - Outgoing: one connection per linked Mac

    private func dial(_ id: String) {
        guard running, let peer = peers.peer(id) else { return }
        let c = conns[id] ?? { let n = PeerConn(); conns[id] = n; return n }()
        guard c.state == .offline else { return }
        c.retry?.cancel(); c.retry = nil
        let endpoint: NWEndpoint
        if let d = discovered[id] {
            endpoint = d.endpoint
        } else if let h = peer.host, let p = peer.port, let port = NWEndpoint.Port(rawValue: p) {
            endpoint = .hostPort(host: NWEndpoint.Host(h), port: port)
        } else {
            return   // nowhere to dial until Bonjour sees it
        }
        c.state = .connecting
        c.generation += 1
        let gen = c.generation
        notify()
        let key = peer.publicKey
        let conn = NWConnection(to: endpoint, using: LinkTLS.parameters(identity: identity, queue: queue) { $0 == key })
        c.attempt = conn
        Task { [weak self] in
            guard let self else { return }
            do {
                let ch = try await self.connect(conn, timeout: Self.connectTimeout)
                let hello = try await ch.call("hello", .json(self.queue.sync { self.helloLocked() }), timeout: 10).decode(HelloReply.self)
                self.queue.async { self.connected(id, gen: gen, channel: ch, hello: hello, conn: conn) }
            } catch {
                self.queue.async {
                    guard let c = self.conns[id], c.generation == gen else { conn.cancel(); return }
                    conn.cancel()
                    c.attempt = nil
                    if case LinkError.remote(let text) = error, text == Self.notLinked {
                        c.state = .unlinked
                        self.log("\(id): unlinked on the other Mac")
                        self.notify()
                        return
                    }
                    self.failed(id, why: "\(error)")
                }
            }
        }
    }

    private func connected(_ id: String, gen: Int, channel ch: LinkChannel, hello: HelloReply, conn: NWConnection) {
        guard let c = conns[id], c.generation == gen, running else { ch.close(); return }
        c.attempt = nil
        c.channel = ch
        c.state = .online
        c.backoff = 2
        peers.update(id) { p in
            p.name = hello.name
            p.model = hello.model
            p.theyAllowSearch = hello.allowSearch
            p.theyAllowFetch = hello.allowFetch
            if case .hostPort(let h, let port)? = conn.currentPath?.remoteEndpoint {
                p.host = "\(h)"   // with its interface scope, which a link-local address needs
                p.port = port.rawValue
            }
        }
        ch.setHandler(handler(forLinked: id, channel: ch))
        ch.onClose { [weak self] _ in
            guard let self else { return }
            self.queue.async {
                guard let c = self.conns[id], c.generation == gen else { return }
                c.channel = nil
                c.ping?.cancel(); c.ping = nil
                self.failed(id, why: "closed")
            }
        }
        // A frozen Mac keeps its TCP connection up - the kernel answers for it - so liveness is a
        // request it has to answer, not the socket.
        let t = DispatchSource.makeTimerSource(queue: queue)
        t.schedule(deadline: .now() + Self.pingInterval, repeating: Self.pingInterval)
        let timeout = Self.pingTimeout
        t.setEventHandler { [weak ch] in
            guard let ch else { return }
            Task { do { _ = try await ch.call("ping", timeout: timeout) } catch { ch.close() } }
        }
        t.resume()
        c.ping = t
        log("\(id): online")
        notify()
    }

    private func failed(_ id: String, why: String) {
        guard let c = conns[id] else { return }
        c.state = .offline
        c.channel = nil
        notify()
        guard running else { return }
        let delay = c.backoff
        c.backoff = min(c.backoff * 2, 60)
        let item = DispatchWorkItem { [weak self] in self?.dial(id) }
        c.retry = item
        queue.asyncAfter(deadline: .now() + delay, execute: item)
        log("\(id): offline (\(why)); retry in \(Int(delay)) s")
    }

    private func dropConn(_ id: String) {
        guard let c = conns.removeValue(forKey: id) else { return }
        c.generation += 1
        c.retry?.cancel()
        c.ping?.cancel()
        c.attempt?.cancel()
        c.channel?.close()
    }

    private func helloLocked() -> Hello {
        Hello(name: config.name, model: config.model, version: config.version, proto: 1, port: listening)
    }

    /// Bring `conn` to ready and wrap it, or throw. TLS 1.3 lets the client report ready before
    /// the server has judged its certificate; a refused client then sees its first request fail.
    private func connect(_ conn: NWConnection, timeout: Double) async throws -> LinkChannel {
        let q = queue
        let ch: LinkChannel = try await withCheckedThrowingContinuation { cont in
            let done = Locked(false)
            @Sendable func finish(_ r: Result<LinkChannel, Error>) {
                guard !done.get() else { return }
                done.set(true)
                cont.resume(with: r)
            }
            conn.stateUpdateHandler = { state in
                switch state {
                case .ready:
                    guard let key = LinkTLS.peerKey(conn) else { finish(.failure(LinkError.protocolViolation("no peer key"))); return }
                    finish(.success(LinkChannel(connection: conn, peerKey: key, queue: q) { _, _ in
                        throw LinkError.refused("not serving on this connection")
                    }))
                case .failed(let e): finish(.failure(e))
                case .waiting(let e): conn.cancel(); finish(.failure(e))
                case .cancelled: finish(.failure(LinkError.closed))
                default: break
                }
            }
            // Only a connection that has NOT come up is cut: this used to cancel unconditionally, so
            // every link died `timeout` seconds after it connected.
            q.asyncAfter(deadline: .now() + timeout) {
                guard !done.get() else { return }
                conn.cancel()
                finish(.failure(LinkError.timeout))
            }
            conn.start(queue: q)
        }
        ch.start()
        return ch
    }

    // MARK: - Incoming

    private func accept(_ conn: NWConnection) {
        guard running else { conn.cancel(); return }
        conn.stateUpdateHandler = { [weak self] state in
            guard let self else { return }
            switch state {
            case .ready:
                guard let key = LinkTLS.peerKey(conn) else { conn.cancel(); return }
                let ch = LinkChannel(connection: conn, peerKey: key, queue: self.queue) { _, _ in throw LinkError.closed }
                let oid = ObjectIdentifier(ch)
                self.incoming[oid] = ch
                if let peer = self.peers.peer(key: key) {
                    // A Mac this one still trusts can pair again: it may have unlinked this one
                    // while this one was away, and a new code replaces the old link.
                    let linked = self.handler(forLinked: peer.id, channel: ch), pairing = self.pairingHandler(ch)
                    ch.setHandler { m, msg in m.hasPrefix("pair.") ? try await pairing(m, msg) : try await linked(m, msg) }
                } else {
                    ch.setHandler(self.pairingHandler(ch))
                }
                ch.onClose { [weak self] _ in
                    guard let self else { return }
                    self.queue.async { self.incoming[oid] = nil }
                }
                ch.start()
            case .failed, .cancelled:
                break
            case .waiting:
                conn.cancel()
            default: break
            }
        }
        // A client that connects and never finishes its handshake does not hold the socket.
        queue.asyncAfter(deadline: .now() + 15) {
            if conn.state != .ready { conn.cancel() }
        }
        conn.start(queue: queue)
    }

    static let notLinked = "not linked"

    private func handler(forLinked id: String, channel ch: LinkChannel) -> LinkChannel.Handler {
        { [weak self, weak ch] method, msg in
            guard let self else { throw LinkError.closed }
            guard let peer = self.peers.peer(id), peer.publicKey == ch?.peerKey else { throw LinkError.remote(Self.notLinked) }
            switch method {
            case "hello":
                let h = try msg.decode(Hello.self)
                let remote = ch?.connection.currentPath?.remoteEndpoint
                self.peers.update(id) { p in
                    p.name = h.name; p.model = h.model
                    if let port = h.port, case .hostPort(let host, _)? = remote { p.host = "\(host)"; p.port = port }
                }
                // A Mac that paired by coming to this one is only now reachable from here: dial it back.
                self.queue.async { if self.conns[id]?.state == .offline { self.dial(id) } }
                self.queue.async { self.notify() }
                let c = self.queue.sync { self.config }
                return try .json(HelloReply(name: c.name, model: c.model, version: c.version, proto: 1,
                                            allowSearch: peer.allowSearch, allowFetch: peer.allowFetch))
            case "ping":
                return LinkMessage()
            case "grants":
                // The other Mac changed what it lets this one do.
                let g = try msg.decode(Grants.self)
                self.peers.update(id) { p in p.theyAllowSearch = g.allowSearch; p.theyAllowFetch = g.allowFetch }
                self.queue.async { self.notify() }
                return LinkMessage()
            case "unlink":
                self.queue.async {
                    self.log("\(id): unlinked by the other Mac")
                    self.dropConn(id)
                    self.peers.remove(id)
                    self.reconcileLocked()
                    self.notify()
                }
                return LinkMessage()
            default:
                guard peer.allowSearch else { throw LinkError.refused("Not sharing with \(peer.name).") }
                if Self.fetchMethods.contains(method), !peer.allowFetch {
                    throw LinkError.refused("Opening files is off.")
                }
                return try await self.service.serve(method, msg, peer: peer)
            }
        }
    }

    /// Tell a linked Mac what this one now grants it (after a switch in the sidebar).
    public func pushGrants(_ id: String) {
        guard let p = peers.peer(id) else { return }
        let ch: LinkChannel? = queue.sync {
            incoming.values.first { $0.peerKey == p.publicKey && !$0.isClosedOnQueue } ?? conns[id]?.channel
        }
        guard let ch else { return }
        Task { _ = try? await ch.call("grants", .json(Grants(allowSearch: p.allowSearch, allowFetch: p.allowFetch)), timeout: 5) }
    }

    // MARK: - Pairing, server side

    private final class ServerPairing: @unchecked Sendable {
        var begin: PairBegin?
        var nonce = Data()
    }

    private func pairingHandler(_ ch: LinkChannel) -> LinkChannel.Handler {
        let st = ServerPairing()
        return { [weak self, weak ch] method, msg in
            guard let self, let ch else { throw LinkError.closed }
            switch method {
            case "pair.begin":
                let ok: String? = self.queue.sync {
                    if !self.windowOpen { return "Open Link a Mac on \(self.config.name) first." }
                    if self.pairingInProgress { return "\(self.config.name) is busy linking another Mac." }
                    self.pairingInProgress = true
                    return nil
                }
                if let ok { throw LinkError.refused(ok) }
                let b = try msg.decode(PairBegin.self)
                st.begin = b
                st.nonce = LinkPairing.nonce()
                let c = self.queue.sync { self.config }
                // A client that begins and walks away must not hold the pairing slot.
                ch.onClose { [weak self] _ in
                    guard let self else { return }
                    self.queue.async { self.pairingInProgress = false }
                }
                return try .json(PairBeginReply(name: c.name, model: c.model, nonce: st.nonce))
            case "pair.reveal":
                guard let b = st.begin else { throw LinkError.protocolViolation("reveal before begin") }
                st.begin = nil   // one reveal per begin
                let r = try msg.decode(PairReveal.self)
                guard LinkPairing.commitment(r.nonce) == b.commit else {
                    self.queue.async { self.pairingInProgress = false }
                    throw LinkError.refused("Pairing failed.")
                }
                guard let exporter = LinkTLS.exporter(ch.connection) else { throw LinkError.protocolViolation("no exporter") }
                let code = LinkPairing.code(exporter: exporter, clientKey: ch.peerKey, serverKey: self.identity.publicKey,
                                            clientNonce: r.nonce, serverNonce: st.nonce)
                let req = IncomingPairing(name: b.name, model: b.model, code: code)
                ch.onClose { _ in req.finish(.cancelled) }
                self.queue.async { self.onPairingRequest?(req) }
                let outcome = await req.wait()
                self.queue.async { self.pairingInProgress = false }
                switch outcome {
                case .linked:
                    let peer = LinkPeer(id: LinkIdentity.id(of: ch.peerKey), publicKey: ch.peerKey, name: b.name, model: b.model,
                                        allowSearch: true, allowFetch: true,
                                        theyAllowSearch: b.share, theyAllowFetch: b.share)
                    self.peers.put(peer)
                    self.queue.async { self.reconcileLocked(); self.notify() }
                    return try .json(PairResult(ok: true, reason: nil))
                case .wrong:
                    return try .json(PairResult(ok: false, reason: "Wrong code."))
                case .cancelled:
                    return try .json(PairResult(ok: false, reason: "Declined on \(self.queue.sync { self.config.name })."))
                }
            case "hello":
                throw LinkError.remote(Self.notLinked)
            default:
                throw LinkError.remote(Self.notLinked)
            }
        }
    }

    private func notify() { onChange?() }
    private func log(_ s: String) { onLog?(s) }
}

/// A Mac asking to link, while this Mac's Link window is open. The UI shows `name` and asks for the
/// code shown on that Mac; three wrong tries end it.
public final class IncomingPairing: @unchecked Sendable, Identifiable {
    public enum Outcome: Sendable { case linked, wrong, cancelled }
    public enum Submit: Sendable, Equatable { case linked, wrong(triesLeft: Int), over }
    public static let lifetime: Double = 180
    public static let tries = 3

    public let id = UUID()
    public let name: String
    public let model: String
    let code: String   // internal: the tests compare it with what the other Mac showed
    private let lock = NSLock()
    private var left = IncomingPairing.tries
    private var outcome: Outcome?
    private var waiter: CheckedContinuation<Outcome, Never>?
    /// Called once when the request ends for any reason (so the sheet closes).
    public var onEnd: (@Sendable (Outcome) -> Void)?

    init(name: String, model: String, code: String) {
        self.name = name; self.model = model; self.code = code
        DispatchQueue.global().asyncAfter(deadline: .now() + Self.lifetime) { [weak self] in self?.finish(.cancelled) }
    }

    public func submit(_ typed: String) -> Submit {
        let result: Submit = lock.withLock {
            guard outcome == nil else { return .over }
            if LinkPairing.matches(typed, code) { return .linked }
            left -= 1
            return left > 0 ? .wrong(triesLeft: left) : .over
        }
        switch result {
        case .linked: finish(.linked)
        case .over: finish(.wrong)
        case .wrong: break
        }
        return result
    }

    public func cancel() { finish(.cancelled) }

    func finish(_ o: Outcome) {
        let (w, end): (CheckedContinuation<Outcome, Never>?, (@Sendable (Outcome) -> Void)?) = lock.withLock {
            guard outcome == nil else { return (nil, nil) }
            outcome = o
            defer { waiter = nil }
            return (waiter, onEnd)
        }
        w?.resume(returning: o)
        end?(o)
    }

    func wait() async -> Outcome {
        await withCheckedContinuation { c in
            let done: Outcome? = lock.withLock {
                if let o = outcome { return o }
                waiter = c
                return nil
            }
            if let done { c.resume(returning: done) }
        }
    }
}

// MARK: - Messages

struct PairBegin: Codable { var name: String; var model: String; var commit: Data; var share: Bool }
struct PairBeginReply: Codable { var name: String; var model: String; var nonce: Data }
struct PairReveal: Codable { var nonce: Data }
struct PairResult: Codable { var ok: Bool; var reason: String? }
struct Hello: Codable {
    var name: String; var model: String; var version: String; var proto: Int
    /// The port this Mac listens on, so the other one can dial back without Bonjour (a VPN).
    var port: UInt16?
}
struct HelloReply: Codable {
    var name: String; var model: String; var version: String; var proto: Int
    var allowSearch: Bool; var allowFetch: Bool
}
struct Grants: Codable { var allowSearch: Bool; var allowFetch: Bool }

final class Locked<T>: @unchecked Sendable {
    private var v: T
    private let l = NSLock()
    init(_ v: T) { self.v = v }
    func get() -> T { l.lock(); defer { l.unlock() }; return v }
    func set(_ n: T) { l.lock(); v = n; l.unlock() }
}
