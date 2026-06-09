import Foundation
import OmniKit
import Observation
import Security

/// The single serving controller AppModel owns. The SwiftUI tab binds only to the members
/// documented here; AppModel wires the engine/store via attach(). @MainActor so all
/// published state lives on the main actor; the HTTP server and its handler run entirely
/// off it (on the server's own DispatchQueue and detached Tasks), and only LogEntry +
/// counters are marshalled back here, coalesced one invalidation per runloop tick.
@MainActor
@Observable
final class ServingController {

    // MARK: Persisted settings (UserDefaults "omni.serving.*")

    var enabled: Bool = false { didSet { persist(); reconcile() } }
    var scope: ServingScope = .local { didSet { persist(); if isRunning { restart() } } }
    var port: Int = 51234 { didSet { persist(); if isRunning { restart() } } }
    // The auth closure snapshots the token at start, so a running server MUST restart to apply a
    // change - otherwise "switch to LAN, then generate a token" leaves the listener unauthenticated
    // while the UI shows a token set.
    var bearerToken: String = "" { didSet { persist(); if isRunning { restart() } } }

    // MARK: Published runtime state (read-only for the view)

    enum State: Equatable {
        case stopped
        case running
        case portInUse
        case failed(String)
    }

    private(set) var state: State = .stopped
    private(set) var isRunning: Bool = false      // mirrors state == .running
    private(set) var boundAddress: String = ""    // e.g. "http://127.0.0.1:51234"; "" when stopped
    private(set) var requestCount: Int = 0
    private(set) var errorCount: Int = 0
    private(set) var log: [LogEntry] = []         // newest first, capped at 200

    // MARK: Private state

    private var backend: (any ServingBackend)?
    private var server: HTTPServer?

    private let logCap = 200
    private let defaults = UserDefaults.standard
    /// True while load() is assigning persisted values, so each property's didSet does not
    /// persist() back a half-loaded snapshot (which would clobber fields not yet read).
    private var isLoading = false

    init() {
        load()
    }

    // MARK: Lifecycle (wired from AppModel, not bound by the view)

    /// Build the backend from the engine/store and reconcile: auto-start if previously
    /// enabled. Safe to call again after a model swap - it replaces the backend and, if a
    /// server is running, restarts it against the new backend.
    func attach(engine: OmniEngine, store: VectorStore, modelName: String) {
        backend = EngineServingBackend(engine: engine, store: store, modelName: modelName)
        if isRunning {
            restart()
        } else {
            reconcile()
        }
    }

    /// Tear down on model teardown: stop the server and drop the backend.
    func detach() {
        stopServer()
        backend = nil
    }

    // MARK: Actions the view triggers

    func clearLog() {
        log.removeAll()
        requestCount = 0
        errorCount = 0
    }

    // MARK: Reconciliation

    private func reconcile() {
        if enabled, !isRunning, backend != nil {
            startServer()
        } else if !enabled, isRunning {
            stopServer()
        }
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
        let token = bearerToken
        let requireToken = isPublic
        let auth: @Sendable (HTTPRequest) -> Bool = { req in
            guard requireToken else { return true }
            if req.bearer == token { return true }
            if req.googApiKey == token { return true }
            if req.query["key"] == token { return true }
            return false
        }

        let router = Router(backend: backend, auth: auth)
        let sink: @Sendable (LogEntry) -> Void = { [weak self] entry in
            Task { @MainActor in self?.ingest(entry) }
        }

        let srv = HTTPServer(handler: { req in await router.handle(req) }, onLog: sink)
        srv.onFailure = { [weak self, weak srv] msg in
            Task { @MainActor in
                // Ignore a late failure from a DISCARDED server (rapid toggle / port edit): it must not
                // clobber the live server's state, or reconcile would show Stopped while the real
                // listener keeps serving, and the next toggle binds over the orphan -> EADDRINUSE wedge.
                guard let self, let srv, self.server === srv else { return }
                self.state = msg == "port in use" ? .portInUse : .failed(msg)
                self.isRunning = false
                self.boundAddress = ""
                self.enabled = false
            }
        }

        let host = isPublic ? "0.0.0.0" : "127.0.0.1"
        do {
            try srv.start(host: host, port: UInt16(port))
            server = srv
            state = .running
            isRunning = true
            boundAddress = "http://\(isPublic ? lanAddress() : "127.0.0.1"):\(port)"
        } catch {
            state = .portInUse
            isRunning = false
            boundAddress = ""
            enabled = false
            server = nil
        }
    }

    private func stopServer() {
        server?.stop()
        server = nil
        isRunning = false
        boundAddress = ""
        if state == .running { state = .stopped }
    }

    /// Coalesced log ingest: one main-actor invalidation per tick. Newest first, capped.
    private func ingest(_ e: LogEntry) {
        requestCount += 1
        if e.status >= 400 { errorCount += 1 }
        log.insert(e, at: 0)
        if log.count > logCap { log.removeLast(log.count - logCap) }
    }

    // MARK: Persistence

    private func load() {
        isLoading = true
        defer { isLoading = false }
        if defaults.object(forKey: "omni.serving.enabled") != nil {
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
        defaults.set(enabled, forKey: "omni.serving.enabled")
        defaults.set(scope.rawValue, forKey: "omni.serving.scope")
        defaults.set(port, forKey: "omni.serving.port")
        defaults.set(bearerToken, forKey: "omni.serving.token")
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
                        address = String(cString: host)
                    }
                }
            }
            ptr = p.pointee.ifa_next
        }
        return address
    }
}
