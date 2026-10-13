import Foundation
import Network

/// One request or reply on the link. `body` is the method's JSON payload; `blob` carries raw bytes
/// (a thumbnail, a slice of a file) so they never pass through JSON.
public struct LinkMessage: Sendable {
    public var body: Data
    public var blob: Data
    public init(body: Data = Data(), blob: Data = Data()) { self.body = body; self.blob = blob }

    public static func json<T: Encodable>(_ v: T, blob: Data = Data()) throws -> LinkMessage {
        LinkMessage(body: try JSONEncoder().encode(v), blob: blob)
    }
    public func decode<T: Decodable>(_ t: T.Type) throws -> T {
        do { return try JSONDecoder().decode(t, from: body) }
        catch { throw LinkError.protocolViolation("\(T.self) did not decode") }
    }
}

/// Request/response over one established TLS connection, both directions at once. Frames are
///
///     u32 length of the rest | u32 header length | header JSON | u32 body length | body | blob
///
/// big-endian. A request carries an id the reply echoes, so several calls are in flight together
/// and a cancelled one is told to the other end (which cancels the handler's task).
///
/// All state lives on `queue`; the class is Sendable because nothing touches it anywhere else.
public final class LinkChannel: @unchecked Sendable {
    public typealias Handler = @Sendable (_ method: String, _ msg: LinkMessage) async throws -> LinkMessage

    /// Biggest frame either end accepts. A file is fetched in slices well under this, so a peer
    /// that declares more is broken or hostile, and the connection is dropped before buffering it.
    static let maxFrame = 24 << 20

    public let connection: NWConnection
    public let peerKey: Data
    public var peerID: String { LinkIdentity.id(of: peerKey) }
    let queue: DispatchQueue
    private var handler: Handler
    private var nextID: UInt64 = 1
    private var pending: [UInt64: CheckedContinuation<LinkMessage, Error>] = [:]
    private var serving: [UInt64: Task<Void, Never>] = [:]
    private var buffer = Data()
    private var closed = false
    private var closeHandlers: [@Sendable (Error?) -> Void] = []

    struct Header: Codable {
        var id: UInt64
        var t: String            // "q" request, "r" reply, "e" error, "c" cancel
        var m: String? = nil     // method, on a request
        var e: String? = nil     // error text, on an error
    }

    init(connection: NWConnection, peerKey: Data, queue: DispatchQueue, handler: @escaping Handler) {
        self.connection = connection
        self.peerKey = peerKey
        self.queue = queue
        self.handler = handler
    }

    /// Start reading. The connection must already be ready.
    func start() {
        queue.async { [self] in
            connection.stateUpdateHandler = { [weak self] state in
                switch state {
                case .failed(let e): self?.shutdown(e)
                case .cancelled: self?.shutdown(nil)
                default: break
                }
            }
            receive()
        }
    }

    func setHandler(_ h: @escaping Handler) { queue.async { self.handler = h } }

    /// Called once when the channel closes, from either end or by error. Added after the close it
    /// runs at once.
    public func onClose(_ f: @escaping @Sendable (Error?) -> Void) {
        queue.async { [self] in
            if closed { f(nil) } else { closeHandlers.append(f) }
        }
    }

    public var isClosed: Bool { queue.sync { closed } }
    /// The same, for code already running on `queue` (the node shares its queue with its channels).
    var isClosedOnQueue: Bool { closed }

    public func close() { queue.async { [self] in shutdown(nil) } }

    /// Call `method` on the other end. Throws `.timeout` past `timeout` seconds, `.closed` if the
    /// connection goes, `.remote` with the other end's message if its handler threw, and
    /// `.cancelled` if the calling task is cancelled (the other end is told).
    public func call(_ method: String, _ msg: LinkMessage = LinkMessage(), timeout: Double) async throws -> LinkMessage {
        let id: UInt64 = queue.sync { defer { nextID += 1 }; return nextID }
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (cont: CheckedContinuation<LinkMessage, Error>) in
                queue.async { [self] in
                    guard !closed else { cont.resume(throwing: LinkError.closed); return }
                    pending[id] = cont
                    send(Header(id: id, t: "q", m: method), msg)
                    queue.asyncAfter(deadline: .now() + timeout) { [weak self] in
                        guard let self, let c = self.pending.removeValue(forKey: id) else { return }
                        self.send(Header(id: id, t: "c"), LinkMessage())
                        c.resume(throwing: LinkError.timeout)
                    }
                }
            }
        } onCancel: {
            queue.async { [self] in
                guard let c = pending.removeValue(forKey: id) else { return }
                send(Header(id: id, t: "c"), LinkMessage())
                c.resume(throwing: LinkError.cancelled)
            }
        }
    }

    // MARK: - On the queue

    private func send(_ h: Header, _ msg: LinkMessage) {
        guard !closed, let hd = try? JSONEncoder().encode(h) else { return }
        var frame = Data(capacity: 12 + hd.count + msg.body.count + msg.blob.count)
        let rest = 4 + hd.count + 4 + msg.body.count + msg.blob.count
        frame.append(be32(rest)); frame.append(be32(hd.count)); frame.append(hd)
        frame.append(be32(msg.body.count)); frame.append(msg.body); frame.append(msg.blob)
        connection.send(content: frame, completion: .contentProcessed { [weak self] err in
            guard let err, let self else { return }
            self.queue.async { self.shutdown(err) }
        })
    }

    private func receive() {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 1 << 20) { [weak self] data, _, complete, err in
            guard let self else { return }
            if let data, !data.isEmpty { self.buffer.append(data); self.drain() }
            if let err { self.shutdown(err); return }
            if complete { self.shutdown(nil); return }
            if !self.closed { self.receive() }
        }
    }

    private func drain() {
        while buffer.count >= 4 {
            let rest = Int(u32(buffer, buffer.startIndex))
            guard rest >= 8, rest <= Self.maxFrame else { shutdown(LinkError.protocolViolation("frame of \(rest) bytes")); return }
            guard buffer.count >= 4 + rest else { return }
            let s = buffer.startIndex + 4
            let hlen = Int(u32(buffer, s))
            guard hlen <= rest - 8 else { shutdown(LinkError.protocolViolation("header length")); return }
            let hd = buffer.subdata(in: s + 4 ..< s + 4 + hlen)
            let bs = s + 4 + hlen
            let blen = Int(u32(buffer, bs))
            guard 4 + hlen + 4 + blen <= rest else { shutdown(LinkError.protocolViolation("body length")); return }
            let body = buffer.subdata(in: bs + 4 ..< bs + 4 + blen)
            let blob = buffer.subdata(in: bs + 4 + blen ..< s + rest)
            buffer = buffer.subdata(in: s + rest ..< buffer.endIndex)
            guard let h = try? JSONDecoder().decode(Header.self, from: hd) else {
                shutdown(LinkError.protocolViolation("header")); return
            }
            dispatch(h, LinkMessage(body: body, blob: blob))
            if closed { return }
        }
    }

    private func dispatch(_ h: Header, _ msg: LinkMessage) {
        switch h.t {
        case "q":
            let method = h.m ?? ""
            let handler = self.handler
            serving[h.id] = Task { [weak self] in
                let reply: Result<LinkMessage, Error>
                do { reply = .success(try await handler(method, msg)) } catch { reply = .failure(error) }
                guard let self else { return }
                self.queue.async {
                    guard self.serving.removeValue(forKey: h.id) != nil else { return }   // cancelled
                    switch reply {
                    case .success(let m): self.send(Header(id: h.id, t: "r"), m)
                    case .failure(let e):
                        let text = (e as? LinkError).map(\.description) ?? String(describing: e)
                        self.send(Header(id: h.id, t: "e", e: text), LinkMessage())
                    }
                }
            }
        case "r":
            pending.removeValue(forKey: h.id)?.resume(returning: msg)
        case "e":
            pending.removeValue(forKey: h.id)?.resume(throwing: LinkError.remote(h.e ?? "failed"))
        case "c":
            serving.removeValue(forKey: h.id)?.cancel()
        default:
            shutdown(LinkError.protocolViolation("frame type \(h.t)"))
        }
    }

    private func shutdown(_ err: Error?) {
        guard !closed else { return }
        closed = true
        connection.cancel()
        let p = pending; pending = [:]
        for (_, c) in p { c.resume(throwing: LinkError.closed) }
        for (_, t) in serving { t.cancel() }
        serving = [:]
        buffer = Data()
        let hs = closeHandlers; closeHandlers = []
        for f in hs { f(err) }
    }

    private func be32(_ n: Int) -> Data { withUnsafeBytes(of: UInt32(n).bigEndian) { Data($0) } }
    private func u32(_ d: Data, _ i: Data.Index) -> UInt32 {
        UInt32(d[i]) << 24 | UInt32(d[i + 1]) << 16 | UInt32(d[i + 2]) << 8 | UInt32(d[i + 3])
    }
}
