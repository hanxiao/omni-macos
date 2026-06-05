import Foundation
import Network
import os

/// Minimal in-process HTTP/1.1 server on Network.framework. Pure transport: it knows
/// nothing about embeddings or routing. All connection callbacks fire on a dedicated
/// serial DispatchQueue (matching Network.framework's model), so there is zero actor-hop
/// overhead on the hot path. Per-request work is dispatched into a detached Task that
/// awaits the async handler; only the LogEntry callback crosses to the main actor (the
/// controller coalesces it).
final class HTTPServer: @unchecked Sendable {
    static let log = Logger(subsystem: "ai.jina.omni", category: "serving")

    typealias Handler = @Sendable (HTTPRequest) async -> HTTPResponse

    /// Called when the listener fails to come up or dies (e.g. port in use). The
    /// controller installs this to flip its state. Always invoked on `queue`.
    var onFailure: (@Sendable (String) -> Void)?

    private let handler: Handler
    private let onLog: @Sendable (LogEntry) -> Void
    private let queue = DispatchQueue(label: "omni.serving.http")
    private let maxBody = 8 * 1024 * 1024 // 8 MB hard cap -> 413
    private let receiveChunk = 64 * 1024

    private var listener: NWListener?
    /// When bound for local scope we accept the listener on all interfaces (the reliable
    /// NWListener path) but drop any connection whose peer is not loopback - functionally
    /// local-only without the fragile requiredLocalEndpoint bind.
    private var loopbackOnly = false

    init(handler: @escaping Handler, onLog: @escaping @Sendable (LogEntry) -> Void) {
        self.handler = handler
        self.onLog = onLog
    }

    /// Bind and start accepting. Throws if NWListener can't be created. Runtime bind
    /// failures (port in use) arrive asynchronously via `onFailure`.
    func start(host: String, port: UInt16) throws {
        guard let nwPort = NWEndpoint.Port(rawValue: port) else {
            throw HTTPError.badRequest
        }
        loopbackOnly = (host == "127.0.0.1")

        // Canonical bind: NWListener(using:on:) on all interfaces. requiredLocalEndpoint with
        // a port throws EINVAL here, so loopback scope is enforced per-connection in accept().
        let params = NWParameters.tcp
        params.allowLocalEndpointReuse = true

        let l = try NWListener(using: params, on: nwPort)
        self.listener = l

        l.stateUpdateHandler = { [weak self] state in
            guard let self else { return }
            switch state {
            case .ready:
                Self.log.info("serving listener ready on port \(port, privacy: .public)")
            case .failed(let error):
                Self.log.error("serving listener failed: \(String(describing: error), privacy: .public)")
                self.onFailure?(self.describe(error))
            case .waiting(let error):
                Self.log.error("serving listener waiting: \(String(describing: error), privacy: .public)")
            case .cancelled:
                break
            default:
                break
            }
        }

        l.newConnectionHandler = { [weak self] conn in
            self?.accept(conn)
        }

        l.start(queue: queue)
    }

    /// True if the connection's peer is the loopback interface (127.0.0.1 / ::1 / localhost).
    private func isLoopback(_ conn: NWConnection) -> Bool {
        guard case .hostPort(let host, _) = conn.endpoint else { return false }
        switch host {
        case .ipv4(let a): return a.isLoopback || a == .loopback
        case .ipv6(let a): return a.isLoopback || a == .loopback
        case .name(let n, _): return n == "localhost"
        @unknown default: return false
        }
    }

    func stop() {
        listener?.cancel()
        listener = nil
    }

    // MARK: - Connection handling

    private func accept(_ conn: NWConnection) {
        // Local scope: refuse anything not coming from loopback.
        if loopbackOnly && !isLoopback(conn) {
            conn.cancel()
            return
        }
        conn.stateUpdateHandler = { [weak self] state in
            switch state {
            case .ready:
                self?.readRequest(on: conn, buffer: Data())
            case .failed, .cancelled:
                conn.cancel()
            default:
                break
            }
        }
        conn.start(queue: queue)
    }

    /// Accumulate bytes until a full request is buffered, then service it. Drains
    /// pipelined requests already sitting in `buffer` before reading more.
    private func readRequest(on conn: NWConnection, buffer: Data) {
        // First, try to satisfy from what we already have (handles pipelining and the
        // case where the head+body arrived in one receive).
        if drain(on: conn, buffer: buffer) { return }

        conn.receive(minimumIncompleteLength: 1, maximumLength: receiveChunk) { [weak self] data, _, isComplete, error in
            guard let self else { return }
            var buf = buffer
            if let data, !data.isEmpty { buf.append(data) }

            if buf.count > self.maxBody {
                self.write(HTTPResponse.json(["error": "payload too large"], status: 413), to: conn, keepAlive: false) {
                    conn.cancel()
                }
                return
            }

            if error != nil {
                conn.cancel()
                return
            }

            if self.drain(on: conn, buffer: buf) { return }

            if isComplete {
                // Peer closed without a complete request.
                conn.cancel()
                return
            }

            // Need more bytes.
            self.readRequest(on: conn, buffer: buf)
        }
    }

    /// Attempt to parse and service exactly one request from `buffer`. Returns true if a
    /// request was found and handled (the continuation owns the next read), false if more
    /// bytes are needed (caller should read on).
    private func drain(on conn: NWConnection, buffer: Data) -> Bool {
        let parsed: (HTTPRequest, Int)?
        do {
            parsed = try HTTPParse.tryParse(buffer)
        } catch {
            write(HTTPResponse.json(["error": "bad request"], status: 400), to: conn, keepAlive: false) {
                conn.cancel()
            }
            return true
        }

        guard let (req, consumed) = parsed else { return false }

        let residual = consumed < buffer.count
            ? buffer.subdata(in: buffer.index(buffer.startIndex, offsetBy: consumed)..<buffer.endIndex)
            : Data()
        let client = self.remoteDescription(conn)
        let keepAlive = req.wantsKeepAlive
        let started = DispatchTime.now()

        // Hand off the (thread-safe) handler to a detached Task. The engine/store are
        // documented thread-safe, so they are called directly off the main actor here.
        Task { [weak self] in
            guard let self else { return }
            let resp = await self.handler(req)
            let ms = Double(DispatchTime.now().uptimeNanoseconds - started.uptimeNanoseconds) / 1_000_000.0
            let entry = LogEntry(
                time: Date(),
                method: req.method,
                path: req.routePath,
                status: resp.status,
                ms: ms,
                client: client
            )
            self.onLog(entry)

            // Hop back to the network queue to write and continue the loop.
            self.queue.async {
                self.write(resp, to: conn, keepAlive: keepAlive) {
                    if keepAlive {
                        self.readRequest(on: conn, buffer: residual)
                    } else {
                        conn.cancel()
                    }
                }
            }
        }
        return true
    }

    private func write(_ resp: HTTPResponse, to conn: NWConnection, keepAlive: Bool, completion: @escaping () -> Void) {
        let bytes = resp.serialize(keepAlive: keepAlive)
        conn.send(content: bytes, completion: .contentProcessed { _ in
            completion()
        })
    }

    // MARK: - Helpers

    private func remoteDescription(_ conn: NWConnection) -> String {
        switch conn.endpoint {
        case .hostPort(let host, _):
            return "\(host)"
        default:
            return "\(conn.endpoint)"
        }
    }

    private func describe(_ error: NWError) -> String {
        switch error {
        case .posix(let code) where code == .EADDRINUSE:
            return "port in use"
        case .posix(let code):
            return "posix \(code.rawValue)"
        default:
            return "\(error)"
        }
    }
}
