import CryptoKit
import Network
import XCTest
@testable import OmniLink

/// Mutual TLS 1.3 with in-memory self-signed identities, trust by pinned key alone.
final class TLSTests: XCTestCase {
    private func listen(_ id: LinkIdentity, accept: @escaping @Sendable (Data) -> Bool,
                        onConn: @escaping @Sendable (NWConnection) -> Void) throws -> (NWListener, UInt16) {
        let q = DispatchQueue(label: "t.server")
        let l = try NWListener(using: LinkTLS.parameters(identity: id, queue: q, accept: accept), on: .any)
        let ready = expectation(description: "listening")
        l.stateUpdateHandler = { if case .ready = $0 { ready.fulfill() } }
        l.newConnectionHandler = { c in onConn(c); c.start(queue: q) }
        l.start(queue: q)
        wait(for: [ready], timeout: 5)
        return (l, l.port!.rawValue)
    }

    private func dial(_ id: LinkIdentity, port: UInt16, accept: @escaping @Sendable (Data) -> Bool) -> (NWConnection, Bool) {
        let q = DispatchQueue(label: "t.client")
        let c = NWConnection(host: "127.0.0.1", port: NWEndpoint.Port(rawValue: port)!,
                             using: LinkTLS.parameters(identity: id, queue: q, accept: accept))
        let done = expectation(description: "client state")
        let ok = Locked(false), once = Locked(false)
        let finish = { @Sendable in if !once.get() { once.set(true); done.fulfill() } }
        c.stateUpdateHandler = { s in
            switch s {
            case .ready: ok.set(true); finish()
            case .failed, .cancelled: finish()
            case .waiting: c.cancel()
            default: break
            }
        }
        c.start(queue: q)
        wait(for: [done], timeout: 10)
        return (c, ok.get())
    }

    func testMutualPinnedHandshake() throws {
        let a = try LinkIdentity(key: P256.Signing.PrivateKey())
        let b = try LinkIdentity(key: P256.Signing.PrivateKey())
        let serverSawConn = Locked<NWConnection?>(nil)
        let (l, port) = try listen(b, accept: { $0 == a.publicKey }, onConn: { serverSawConn.set($0) })
        defer { l.cancel() }
        let (c, ok) = dial(a, port: port, accept: { $0 == b.publicKey })
        defer { c.cancel() }
        XCTAssertTrue(ok, "the handshake completes when both keys are the pinned ones")
        XCTAssertEqual(LinkTLS.peerKey(c), b.publicKey)
        // Give the server side a moment to reach ready, then compare exporters.
        let deadline = Date().addingTimeInterval(5)
        while Date() < deadline, serverSawConn.get()?.state != .ready { usleep(10_000) }
        let s = try XCTUnwrap(serverSawConn.get())
        XCTAssertEqual(LinkTLS.peerKey(s), a.publicKey)
        let ec = try XCTUnwrap(LinkTLS.exporter(c)), es = try XCTUnwrap(LinkTLS.exporter(s))
        XCTAssertEqual(ec, es, "one session, one exporter")
        XCTAssertEqual(ec.count, 32)
    }

    func testWrongServerKeyRefused() throws {
        let a = try LinkIdentity(key: P256.Signing.PrivateKey())
        let b = try LinkIdentity(key: P256.Signing.PrivateKey())
        let other = try LinkIdentity(key: P256.Signing.PrivateKey())
        let (l, port) = try listen(b, accept: { _ in true }, onConn: { _ in })
        defer { l.cancel() }
        let (c, ok) = dial(a, port: port, accept: { $0 == other.publicKey })
        c.cancel()
        XCTAssertFalse(ok, "a server whose key is not the pinned one is refused")
    }

    func testUnknownClientRefused() throws {
        let a = try LinkIdentity(key: P256.Signing.PrivateKey())
        let b = try LinkIdentity(key: P256.Signing.PrivateKey())
        let (l, port) = try listen(b, accept: { _ in false }, onConn: { _ in })
        defer { l.cancel() }
        let (c, ok) = dial(a, port: port, accept: { $0 == b.publicKey })
        // TLS 1.3: the client may report ready before the server has judged its certificate; the
        // refusal then arrives as the connection dying. Either way nothing gets through.
        if ok {
            let dead = expectation(description: "closed by the server")
            c.receive(minimumIncompleteLength: 1, maximumLength: 1) { _, _, complete, err in
                if err != nil || complete { dead.fulfill() }
            }
            wait(for: [dead], timeout: 5)
        }
        c.cancel()
    }

    func testIdentityPersists() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("lid-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("identity")
        let one = try LinkIdentity.loadOrCreate(at: url)
        let two = try LinkIdentity.loadOrCreate(at: url)
        XCTAssertEqual(one.publicKey, two.publicKey)
        XCTAssertEqual(one.id, two.id)
        XCTAssertEqual(one.id.count, 32)
        let perms = try FileManager.default.attributesOfItem(atPath: url.path)[.posixPermissions] as? Int
        XCTAssertEqual(perms, 0o600)
    }
}
