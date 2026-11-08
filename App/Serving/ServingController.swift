import Foundation
import OmniKit
import Observation
import Security

/// The single serving controller AppModel owns. The SwiftUI tab binds only to the members
/// documented here; AppModel wires the engine/store via attach(). @MainActor so all
/// published state lives on the main actor; the HTTP server and its handler run entirely
/// off it (on the server's own DispatchQueue and detached Tasks), and only LogEntry values are
/// marshalled back here.
@MainActor
@Observable
final class ServingController {

    // MARK: Persisted settings (UserDefaults "omni.serving.*")

    var enabled: Bool = false { didSet { persist(); reconcile() } }
    /// Held off for a benchmark run. Never persisted and never written to `enabled`, so a run that
    /// dies part-way leaves Serving exactly as the owner set it, and the next launch serves again.
    var pausedForBenchmark = false { didSet { held.set(pausedForBenchmark); reconcile() } }
    /// What the toolbar and the tab show: off while a benchmark holds it.
    var isOn: Bool { enabled && !pausedForBenchmark }
    var scope: ServingScope = .local { didSet { persist(); restartIfServing() } }
    var port: Int = 51234 { didSet { persist(); restartIfServing() } }
    // The auth closure snapshots the token at start, so a running server MUST restart to apply a
    // change - otherwise "switch to LAN, then generate a token" leaves the listener unauthenticated
    // while the UI shows a token set.
    var bearerToken: String = "" { didSet { persist(); restartIfServing() } }

    // MARK: Published runtime state (read-only for the view)

    enum State: Equatable {
        case stopped
        /// The listener exists and has not bound yet. Not green: a busy port fails from here.
        case starting
        case running
        case portInUse
        case failed(String)
    }

    private(set) var state: State = .stopped
    private(set) var isRunning: Bool = false      // a server object is live (starting or running)
    private(set) var boundAddress: String = ""    // e.g. "http://127.0.0.1:51234"; "" when stopped

    // MARK: Private state

    /// Set by AppModel: what to do with the query text of a served search. Kept as a closure so
    /// Serving stays unaware of AppModel, which is the whole point of the ServingBackend seam.
    var onServedSearch: ((String, ServedSurface) -> Void)?

    /// What Omni indexes, and how to change it - supplied by AppModel, for the same reason
    /// `onServedSearch` is: Serving must not know what AppModel is. Set before attach(); a running
    /// server picks it up on the next restart, which attach() already performs.
    var sources: SourcesControl?

    private var backend: (any ServingBackend)?
    private var server: HTTPServer?
    /// The `omni` command line's socket: the same router, ALWAYS up while a backend is, whatever
    /// the Serving toggle says - the toggle is about opening a port, and the socket opens none.
    private var socketServer: HTTPServer?
    /// Asking again for a socket another Omni holds, or that failed to come up. Yielding was for good:
    /// when the other copy quit, `omni` found a running app with no socket until a relaunch.
    private var socketRetry: Task<Void, Never>?
    private var serverRetry: Task<Void, Never>?
    /// Set for a benchmark run. The hold reached only the TCP server, and the socket went on answering,
    /// though a client mid-run shares the engine with every case. Answered, not closed: `omni` then
    /// says why instead of waiting a minute for a socket it reads as still loading.
    private nonisolated let held = ServingHold()

    /// The log lives only in the file; the Serving tab tails it.
    private nonisolated let logFile = ServingLogFile()
    private let defaults = UserDefaults.standard
    /// True while load() is assigning persisted values, so each property's didSet does not
    /// persist() back a half-loaded snapshot (which would clobber fields not yet read).
    private var isLoading = false

    init() {
        load()
    }

    /// Served searches on their way to the history, between batches (see `attach`).
    private nonisolated let servedSearches = PendingServedSearches()

    // MARK: Lifecycle (wired from AppModel, not bound by the view)

    /// Build the backend from the engine/store and reconcile: auto-start if previously
    /// enabled. Safe to call again after a model swap - it replaces the backend and, if a
    /// server is running, restarts it against the new backend.
    func attach(engine: OmniEngine, store: VectorStore, modelName: String, passages: PassageRecovery? = nil) {
        var b = EngineServingBackend(engine: engine, store: store, modelName: modelName)
        b.passageRecovery = passages
        // The hop off the connection's Task and onto the main actor, in one place, and in BATCHES.
        // One hop per search inserted each query into the sidebar's history on its own, and at agent
        // speed (50-150 searches a second) the sidebar's list re-diffed its rows continuously: 39% of
        // main-thread samples under a 32-client load, and every main-actor hop queued behind it
        // (index.md, "Served search under load"). Queries now wait here and reach the history together
        // every two seconds - one list update for the lot. Not twice a second: with the history kept
        // forever the sidebar's diff grows with the square of its rows, and at 20,000 searches each
        // update cost the main thread ~100 ms (ui.md, "Renders per step"). `onServedSearch` is
        // weak-captured through self so a torn-down controller cannot resurrect the model.
        let pending = servedSearches
        b.onSearch = { [weak self] q, surface in
            guard pending.add(q, surface) else { return }
            Task { @MainActor [weak self] in
                try? await Task.sleep(nanoseconds: 2_000_000_000)
                for (q, surface) in pending.take() { self?.onServedSearch?(q, surface) }
            }
        }
        backend = b
        // In place, when a server is up: what changed is what answers, not the port (HTTPServer.setHandler).
        if isRunning, let server {
            server.setHandler(httpHandler(backend: b, auth: serverAuth()))
        } else {
            reconcile()
        }
        if let socketServer {
            socketServer.setHandler(socketHandler(backend: b))
        } else {
            startSocket()
        }
    }

    /// A request while a benchmark holds serving: what to tell it.
    private nonisolated static var heldResponse: HTTPResponse {
        var r = HTTPResponse.json(["error": ["message": "Omni is running its benchmark; try again when it finishes",
                                             "type": "service_unavailable"]], status: 503)
        r.headers["Retry-After"] = "60"
        return r
    }

    private func socketHandler(backend: any ServingBackend) -> HTTPServer.Handler {
        let router = Router(backend: backend, auth: { _ in true }, appVersion: AppModel.appVersion, sources: sources,
                            mcpSurface: .cli)
        let held = self.held
        return { req in held.isOn ? Self.heldResponse : await router.handle(req) }
    }

    private func httpHandler(backend: any ServingBackend, auth: @escaping @Sendable (HTTPRequest) -> Bool) -> HTTPServer.Handler {
        var built = Router(backend: backend, auth: auth, appVersion: AppModel.appVersion, sources: sources)
        // Local scope: answer only requests addressed to this Mac (DNS rebinding; Router.handle).
        // LAN scope already requires the token, which a rebinding page does not have.
        built.loopbackOnly = scope != .public
        let router = built
        return { req in await router.handle(req) }
    }

    /// Localhost never requires a token; a public (LAN) bind ALWAYS requires one.
    private func serverAuth() -> @Sendable (HTTPRequest) -> Bool {
        let token = bearerToken
        let requireToken = scope == .public
        return { req in
            guard requireToken else { return true }
            if req.bearer == token { return true }
            if req.googApiKey == token { return true }
            if req.query["key"] == token { return true }
            return false
        }
    }

    /// The served search for App Intents (Shortcuts, Siri, Spotlight): the same search, grouping
    /// and passages an agent gets over MCP, on the served work queue. Empty until the engine is up.
    func searchForIntents(_ query: String, limit: Int, width: Int = Indexer.defaultSnippetLength) async -> [(hit: SearchHit, passage: String?)] {
        guard let backend, !held.isOn else { return [] }
        return await ServedWork.run {
            let (hits, vector) = backend.searchReturningVector(query, topK: min(limit * 3, 150), filter: SearchFilter(), surface: .intent)
            let reps = backend.groupedResults(hits, enabled: true, limit: limit).map(\.representative)
            let passages = SearchAdapter.textPassages(reps, query: query, vector: vector, width: width, backend: backend)
            return Array(zip(reps, passages)).map { hit, p in
                var hit = hit
                if let l = p?.locator { hit.locator = l }
                return (hit: hit, passage: p?.text)
            }
        }
    }

    /// Tear down on model teardown: stop the server and drop the backend.
    func detach() {
        stopServer()
        stopSocket()
        backend = nil
    }

    /// Stop the socket at quit, so the file goes with the app rather than waiting for the next
    /// launch to find it dead. The searches still waiting for their batch go to the history now:
    /// the two seconds of them before a quit never reached it.
    func shutdown() {
        for (q, surface) in servedSearches.take() { onServedSearch?(q, surface) }
        stopSocket()
    }

    // MARK: The command line socket

    /// Where the `omni` command line finds the app. Fixed - not beside a moved index - so the
    /// command needs no configuration. An isolated run (`-omni.dbDir`) is not the user's app and
    /// gets no socket unless its own arguments name one (`-omni.socketPath`).
    nonisolated static var socketPath: String? {
        let args = UserDefaults.standard.volatileDomain(forName: UserDefaults.argumentDomain)
        if let p = args["omni.socketPath"] as? String, !p.isEmpty { return p }
        guard ServingLogFile.isolatedDir == nil,
              let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
        else { return nil }
        return support.appendingPathComponent("Omni/omni.sock").path
    }

    private func startSocket() {
        stopSocket()
        guard let backend, let path = Self.socketPath else { return }
        // sockaddr_un holds 104 bytes with the terminator; a longer path cannot be bound at all.
        guard path.utf8.count < 104 else { note(.error, "Command line socket path too long: \(path)"); return }
        try? FileManager.default.createDirectory(atPath: (path as NSString).deletingLastPathComponent,
                                                 withIntermediateDirectories: true)
        // A file already there is either another live Omni's (leave it: two apps on one socket
        // would answer from whichever index the kernel picked) or left by a crash (clear it).
        if FileManager.default.fileExists(atPath: path) {
            if Self.socketAnswers(path) {
                note(.error, "Command line socket in use by another Omni: \(path)")
                retrySocket(path)
                return
            }
            unlink(path)
        }
        let log = logFile
        let srv = HTTPServer(handler: socketHandler(backend: backend), onLog: { log.append($0) })
        srv.admitsLargeBody = { _ in true }
        srv.onFailure = { [weak self, weak srv] msg in
            Task { @MainActor in
                self?.note(.error, "Command line socket failed: \(msg)")
                guard let self, let srv, self.socketServer === srv else { return }
                srv.stop()
                self.socketServer = nil
                self.retrySocket(path)
            }
        }
        do {
            try srv.start(unixPath: path)
            socketServer = srv
        } catch {
            note(.error, "Command line socket failed: \(error)")
            retrySocket(path)
        }
    }

    private func retrySocket(_ path: String) {
        socketRetry?.cancel()
        socketRetry = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(Self.socketRetrySeconds))
                guard !Task.isCancelled else { return }
                let taken = await Task.detached { Self.socketAnswers(path) }.value
                guard !Task.isCancelled else { return }
                if !taken { self?.startSocket(); return }
            }
        }
    }

    private static let socketRetrySeconds: Double = 5

    private func stopSocket() {
        socketRetry?.cancel(); socketRetry = nil
        socketServer?.stop()
        socketServer = nil
    }

    /// Whether something is accepting on the socket at `path` right now.
    nonisolated static func socketAnswers(_ path: String) -> Bool {
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else { return false }
        defer { close(fd) }
        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let bytes = Array(path.utf8)
        guard bytes.count < MemoryLayout.size(ofValue: addr.sun_path) else { return false }
        withUnsafeMutableBytes(of: &addr.sun_path) { raw in
            raw.copyBytes(from: bytes)
            raw[bytes.count] = 0
        }
        let len = socklen_t(MemoryLayout<sockaddr_un>.size)
        return withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { connect(fd, $0, len) == 0 }
        }
    }

    // MARK: Reconciliation

    private func reconcile() {
        if isOn, !isRunning, backend != nil {
            startServer()
        } else if !isOn, isRunning || server != nil || state != .stopped {
            // A server that failed is not running and is still here, with its failure on show.
            stopServer()
        }
    }

    /// A new port, scope or token applies now: to a running server, and to one that failed while
    /// switched on, for which a new port is the retry. `isRunning` alone left "Port in use" standing
    /// after the port was changed, until the next launch.
    private func restartIfServing() {
        let failed: Bool = switch state { case .portInUse, .failed: true; default: false }
        if isRunning || (failed && isOn) { restart() }
    }

    private func restart() {
        stopServer()
        startServer()
    }

    private func startServer() {
        guard let backend else { return }
        stopServer()   // cancel any prior/orphan server before binding a new listener (avoids EADDRINUSE)

        // Snapshot settings into a Sendable auth closure: localhost never requires a token;
        // a public (LAN) bind ALWAYS requires one. Anyone on the network could otherwise call
        // /v1/search and enumerate file paths + snippets, so an empty token on public scope is
        // filled with a generated one rather than served open.
        let isPublic = (scope == .public)
        if isPublic && bearerToken.isEmpty {
            bearerToken = Self.generateToken()   // didSet persists; restart guard below is no-op while starting
        }
        let requireToken = isPublic
        let auth = serverAuth()
        let log = logFile
        let srv = HTTPServer(handler: httpHandler(backend: backend, auth: auth), onLog: { log.append($0) })
        srv.admitsLargeBody = auth
        srv.onFailure = { [weak self, weak srv] msg in
            Task { @MainActor in
                // Ignore a late failure from a DISCARDED server (rapid toggle / port edit): it must not
                // clobber the live server's state, or reconcile would show Stopped while the real
                // listener keeps serving, and the next toggle binds over the orphan -> EADDRINUSE wedge.
                guard let self, let srv, self.server === srv else { return }
                guard self.state == .starting || self.state == .running else { return }   // reported once
                self.state = msg == "port in use" ? .portInUse : .failed(msg)
                self.note(.error, msg == "port in use" ? "Port \(self.port) is in use" : "Server failed: \(msg)")
                self.isRunning = false
                self.boundAddress = ""
                self.retryServerWhenPortIsFree()
                // Deliberately do NOT flip `enabled` (it persists to defaults): a transient bind
                // failure at launch was silently disabling the feature forever, breaking the
                // agents whose SKILL.md points at this server. The toggle stays on, the status
                // dot shows the failure, and it starts by itself once the port is free.
            }
        }

        let host = isPublic ? "0.0.0.0" : "127.0.0.1"
        let address = "http://\(isPublic ? lanAddress() : "127.0.0.1"):\(port)"
        let port = self.port, tokenNote = requireToken ? " with a bearer token" : ""
        // GREEN ONLY ONCE THE LISTENER IS READY. It was set the moment `start` returned, which only
        // means the listener object exists: a port another process holds then showed "Running"
        // beside "Server failed: port in use" in the log.
        srv.onReady = { [weak self, weak srv] in
            Task { @MainActor in
                guard let self, let srv, self.server === srv else { return }
                // After a failure too: a listener that waited and then bound is serving, and said so.
                let waiting: Bool = switch self.state { case .starting, .portInUse, .failed: true; default: false }
                guard waiting else { return }
                self.state = .running
                self.isRunning = true
                self.boundAddress = address
                self.note(.info, "Listening on \(host):\(port)\(tokenNote)")
            }
        }
        do {
            try srv.start(host: host, port: UInt16(port))
            server = srv
            state = .starting
            isRunning = true
        } catch {
            note(.error, "Port \(port) is in use")
            state = .portInUse
            isRunning = false
            boundAddress = ""
            server = nil
            // `enabled` stays as the user set it - see onFailure above.
            retryServerWhenPortIsFree()
        }
    }

    /// A listener that found its port taken never binds once the port comes free (measured: 50 s after
    /// the other process let go, still nothing on the port and "Port in use" on show), and nothing else
    /// tried again until a toggle, a port edit or an engine reload. So this asks: quietly, by binding a
    /// socket of its own, since a restart per try would write two log lines every few seconds for as
    /// long as another app keeps the port.
    private func retryServerWhenPortIsFree() {
        serverRetry?.cancel()
        let port = self.port, isPublic = scope == .public
        serverRetry = Task { [weak self] in
            // The first look soon: a restart under a connected client finds its own old connection
            // still holding the port, gone within a second (measured: down for 5 s with one wait, for
            // good before this existed).
            var wait: Double = 1
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(wait))
                wait = Self.serverRetrySeconds
                guard !Task.isCancelled else { return }
                let free = await Task.detached { Self.canBind(port: port, loopbackOnly: !isPublic) }.value
                guard !Task.isCancelled, let self else { return }
                let waiting: Bool = switch self.state { case .portInUse, .failed: true; default: false }
                guard waiting, self.isOn else { return }
                if free {
                    self.note(.info, "Port \(port) is free")
                    self.startServer()
                    return
                }
            }
        }
    }

    private static let serverRetrySeconds: Double = 5

    /// Whether a listener could take this port right now. The same reuse option the server's listener
    /// sets, so a connection still closing does not read as taken.
    nonisolated private static func canBind(port: Int, loopbackOnly: Bool) -> Bool {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { return false }
        defer { close(fd) }
        var on: Int32 = 1
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, socklen_t(MemoryLayout<Int32>.size))
        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = in_port_t(port).bigEndian
        addr.sin_addr.s_addr = loopbackOnly ? in_addr_t(INADDR_LOOPBACK).bigEndian : in_addr_t(INADDR_ANY).bigEndian
        return withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) == 0 }
        }
    }

    private func stopServer() {
        serverRetry?.cancel(); serverRetry = nil
        if server != nil { note(.info, "Stopped") }
        server?.stop()
        server = nil
        isRunning = false
        boundAddress = ""
        if state != .stopped { state = .stopped }
    }

    private func note(_ level: ServingLogFile.Level, _ message: String) {
        append(ServingLogFile.line(level, message))
    }

    private func append(_ line: String) {
        logFile.append(line)
    }

    // MARK: Persistence

    private func load() {
        isLoading = true
        defer { isLoading = false }
        if ServingLogFile.isolatedDir != nil {
            // AN ISOLATED RUN (`-omni.dbDir`) SERVES ONLY WHEN ITS OWN ARGUMENTS SAY SO. Reading the
            // user's "on" made every test launch try the user's port, fail on it while their app
            // held it, and log that into their serving log beside a green light that was right.
            let args = UserDefaults.standard.volatileDomain(forName: UserDefaults.argumentDomain)
            enabled = (args["omni.serving.enabled"] as? String).map { ["YES", "1", "true"].contains($0) } ?? false
        } else if defaults.object(forKey: "omni.serving.enabled") != nil {
            enabled = defaults.bool(forKey: "omni.serving.enabled")
        }
        if let raw = defaults.string(forKey: "omni.serving.scope"), let s = ServingScope(rawValue: raw) {
            scope = s
        }
        if defaults.object(forKey: "omni.serving.port") != nil {
            let p = defaults.integer(forKey: "omni.serving.port")
            if p > 0 && p <= 65535 { port = p }
        }
        if let t = defaults.string(forKey: "omni.serving.token") {
            bearerToken = t
        }
    }

    private func persist() {
        guard !isLoading else { return }   // don't write a half-loaded snapshot back over saved values
        guard ServingLogFile.isolatedDir == nil else { return }   // an isolated run is not the user's
        OmniPrefs.set(enabled, forKey: "omni.serving.enabled")
        OmniPrefs.set(scope.rawValue, forKey: "omni.serving.scope")
        OmniPrefs.set(port, forKey: "omni.serving.port")
        OmniPrefs.set(bearerToken, forKey: "omni.serving.token")
    }

    /// URL-safe 192-bit random token (shared by the Generate button and the public-scope autofill).
    static func generateToken() -> String {
        var bytes = [UInt8](repeating: 0, count: 24)
        _ = SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes)
        return Data(bytes).base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }

    /// Best-effort LAN IPv4 for display when bound publicly. Falls back to 0.0.0.0.
    private func lanAddress() -> String {
        var address = "0.0.0.0"
        var ifaddr: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&ifaddr) == 0, let first = ifaddr else { return address }
        defer { freeifaddrs(ifaddr) }
        var ptr: UnsafeMutablePointer<ifaddrs>? = first
        while let p = ptr {
            let flags = Int32(p.pointee.ifa_flags)
            let addr = p.pointee.ifa_addr
            if let addr, addr.pointee.sa_family == UInt8(AF_INET),
               (flags & IFF_UP) == IFF_UP, (flags & IFF_LOOPBACK) == 0 {
                let name = String(cString: p.pointee.ifa_name)
                if name == "en0" || name == "en1" {
                    var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
                    if getnameinfo(addr, socklen_t(addr.pointee.sa_len), &host, socklen_t(host.count), nil, 0, NI_NUMERICHOST) == 0 {
                        address = String(decoding: host.prefix(while: { $0 != 0 }).map { UInt8(bitPattern: $0) }, as: UTF8.self)
                    }
                }
            }
            ptr = p.pointee.ifa_next
        }
        return address
    }
}

/// Whether a benchmark holds serving, readable from any request.
final class ServingHold: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false
    var isOn: Bool { lock.withLock { value } }
    func set(_ on: Bool) { lock.withLock { value = on } }
}

/// Served queries waiting for their batch into the history. `add` returns true for the first query
/// since the last `take`, which is the caller's cue to schedule the batch.
final class PendingServedSearches: @unchecked Sendable {
    private let lock = NSLock()
    private var items: [(String, ServedSurface)] = []
    private var armed = false

    func add(_ q: String, _ surface: ServedSurface) -> Bool {
        lock.withLock {
            // Every one: the history keeps every search (no count cap since 2026-10-25), and a batch
            // is bounded by its two seconds.
            items.append((q, surface))
            if armed { return false }
            armed = true
            return true
        }
    }

    func take() -> [(String, ServedSurface)] {
        lock.withLock {
            defer { items = []; armed = false }
            return items
        }
    }
}
