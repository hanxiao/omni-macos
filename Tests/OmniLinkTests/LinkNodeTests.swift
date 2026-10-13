import CryptoKit
import Network
import XCTest
@testable import OmniLink

/// A service that echoes, sleeps on request, and records what it saw.
final class TestService: LinkService, @unchecked Sendable {
    let lock = NSLock()
    var calls: [(String, String)] = []    // (method, peer id)
    var cancelled = 0
    func serve(_ method: String, _ msg: LinkMessage, peer: LinkPeer) async throws -> LinkMessage {
        lock.withLock { calls.append((method, peer.id)) }
        switch method {
        case "echo", "fetch": return msg
        case "sleep":
            do { try await Task.sleep(for: .seconds(30)) } catch { lock.withLock { cancelled += 1 }; throw error }
            return msg
        default: throw LinkError.refused("no such method")
        }
    }
}

final class Node: @unchecked Sendable {
    let node: LinkNode
    let service = TestService()
    let dir: URL
    let requests = Locked<[IncomingPairing]>([])
    init(_ name: String) throws {
        dir = FileManager.default.temporaryDirectory.appendingPathComponent("link-\(name)-\(UUID().uuidString)")
        let id = try LinkIdentity.loadOrCreate(at: dir.appendingPathComponent("identity"))
        node = LinkNode(identity: id, peers: LinkPeerStore(url: dir.appendingPathComponent("peers.json")),
                        config: .init(name: name, model: "Mac15,14", version: "test", port: 0), service: service)
        let r = requests
        node.onPairingRequest = { req in r.set(r.get() + [req]) }
    }
    var id: String { node.identity.id }
    var endpoint: NWEndpoint {
        .hostPort(host: "127.0.0.1", port: NWEndpoint.Port(rawValue: node.port!)!)
    }
    deinit { node.stop(); try? FileManager.default.removeItem(at: dir) }
}

final class LinkNodeTests: XCTestCase, @unchecked Sendable {
    private func until(_ what: String, timeout: Double = 10, _ cond: () -> Bool) {
        let end = Date().addingTimeInterval(timeout)
        while !cond() && Date() < end { usleep(20_000) }
        XCTAssertTrue(cond(), what)
    }

    /// B's window open and listening; A pairs by address; the test types the code A shows on B.
    private func pair(_ a: Node, _ b: Node, share: Bool = true, typed: (@Sendable (String) -> String)? = nil) async throws -> LinkPeer {
        b.node.setPairingWindow(true)
        until("B listens") { b.node.port != nil }
        let shown = Locked<String?>(nil)
        let typer = Task.detached {
            // Wait for B's request, then type what A shows (or what the test says).
            for _ in 0 ..< 500 {
                if let req = b.requests.get().last, let code = shown.get() {
                    let t = typed?(code) ?? code
                    var r = req.submit(t)
                    while case .wrong = r { r = req.submit(t) }
                    return
                }
                try? await Task.sleep(for: .milliseconds(20))
            }
        }
        defer { typer.cancel() }
        return try await a.node.pair(b.endpoint, alsoShare: share) { code, _ in shown.set(code) }
    }

    func testPairThenCallBothWays() async throws {
        let a = try Node("A"), b = try Node("B")
        let peer = try await pair(a, b)
        XCTAssertEqual(peer.id, b.id)
        XCTAssertEqual(peer.name, "B")
        XCTAssertEqual(b.node.peers.peer(a.id)?.name, "A", "B stored A")
        XCTAssertEqual(b.node.peers.peer(a.id)?.allowSearch, true, "typing the code is B's consent")
        until("A is online to B") { a.node.state(of: b.id) == .online }
        let r = try await a.node.call(b.id, "echo", LinkMessage(body: Data("hi".utf8), blob: Data([1, 2, 3])), timeout: 5)
        XCTAssertEqual(r.body, Data("hi".utf8))
        XCTAssertEqual(r.blob, Data([1, 2, 3]))
        XCTAssertEqual(b.service.calls.last?.1, a.id, "B's service saw A as the caller")
        // The other direction: B learned A's listening port from hello and dials back.
        b.node.setPairingWindow(false)
        until("B is online to A") { b.node.state(of: a.id) == .online }
        _ = try await b.node.call(a.id, "echo", timeout: 5)
        XCTAssertEqual(a.service.calls.last?.1, b.id)
    }

    func testOneSidedShare() async throws {
        let a = try Node("A"), b = try Node("B")
        _ = try await pair(a, b, share: false)
        b.node.setPairingWindow(false)
        until("both online") { a.node.state(of: b.id) == .online && b.node.state(of: a.id) == .online }
        _ = try await a.node.call(b.id, "echo", timeout: 5)
        do {
            _ = try await b.node.call(a.id, "echo", timeout: 5)
            XCTFail("A did not share, so B may not search it")
        } catch LinkError.remote(let text) {
            XCTAssertTrue(text.contains("Not sharing"), text)
        }
        XCTAssertEqual(b.node.peers.peer(a.id)?.theyAllowSearch, false, "B knows A does not share")
    }

    func testWrongCodeLinksNothing() async throws {
        let a = try Node("A"), b = try Node("B")
        do {
            _ = try await pair(a, b) { code in code == "000000" ? "111111" : "000000" }
            XCTFail("a wrong code must not link")
        } catch LinkError.refused(let why) {
            XCTAssertTrue(why.contains("Wrong code"), why)
        }
        XCTAssertTrue(a.node.peers.all.isEmpty)
        XCTAssertTrue(b.node.peers.all.isEmpty)
    }

    func testWindowClosedRefuses() async throws {
        let a = try Node("A"), b = try Node("B")
        // B is up (it has some other link) but its Link window is closed.
        b.node.setPairingWindow(true)
        until("B listens") { b.node.port != nil }
        b.node.setPairingWindow(false)
        b.node.peers.put(LinkPeer(id: "x", publicKey: Data([1]), name: "x", model: "", allowSearch: false,
                                  allowFetch: false, theyAllowSearch: false, theyAllowFetch: false))
        b.node.reconcile()
        until("B still listens") { b.node.port != nil }
        do {
            _ = try await a.node.pair(b.endpoint, alsoShare: true) { _, _ in }
            XCTFail("pairing needs the window open")
        } catch LinkError.remote(let why) {
            XCTAssertTrue(why.contains("Open Link a Mac"), why)
        }
        XCTAssertTrue(b.requests.get().isEmpty, "no sheet popped on B")
    }

    /// A relay that terminates TLS on both legs and forwards the pairing messages: A shows one code,
    /// B expects another, so the user typing A's code on B fails. The positive control: the same
    /// relay against a code made from the nonces alone WOULD match - the exporter and keys are what
    /// defeat it.
    func testRelayInTheMiddleFails() async throws {
        let a = try Node("A"), b = try Node("B")
        b.node.setPairingWindow(true)
        until("B listens") { b.node.port != nil }
        let m = try LinkIdentity(key: P256.Signing.PrivateKey())
        let q = DispatchQueue(label: "relay")
        let relay = try NWListener(using: LinkTLS.parameters(identity: m, queue: q) { _ in true }, on: .any)
        let upstream = Locked<LinkChannel?>(nil)
        let keep = Locked<[LinkChannel]>([])   // channels hold themselves only weakly
        let legs = Locked<(Data, Data)?>(nil)   // (exporter A-M, exporter M-B)
        let bPort = b.node.port!
        relay.newConnectionHandler = { conn in
            conn.stateUpdateHandler = { st in
                guard case .ready = st, let key = LinkTLS.peerKey(conn) else { return }
                let down = LinkChannel(connection: conn, peerKey: key, queue: q) { method, msg in
                    if upstream.get() == nil {
                        let up = NWConnection(host: "127.0.0.1", port: NWEndpoint.Port(rawValue: bPort)!,
                                              using: LinkTLS.parameters(identity: m, queue: q) { _ in true })
                        let ready = Locked(false)
                        up.stateUpdateHandler = { if case .ready = $0 { ready.set(true) } }
                        up.start(queue: q)
                        for _ in 0 ..< 250 where !ready.get() { try await Task.sleep(for: .milliseconds(20)) }
                        let ch = LinkChannel(connection: up, peerKey: LinkTLS.peerKey(up)!, queue: q) { _, _ in throw LinkError.closed }
                        ch.start()
                        upstream.set(ch)
                        legs.set((LinkTLS.exporter(conn)!, LinkTLS.exporter(up)!))
                    }
                    return try await upstream.get()!.call(method, msg, timeout: 200)
                }
                keep.set(keep.get() + [down])
                down.start()
            }
            conn.start(queue: q)
        }
        let rready = Locked(false)
        relay.stateUpdateHandler = { if case .ready = $0 { rready.set(true) } }
        relay.start(queue: q)
        until("relay listens") { rready.get() }
        defer { relay.cancel() }

        let shown = Locked<String?>(nil)
        let typer = Task.detached {
            for _ in 0 ..< 500 {
                if let req = b.requests.get().last, let code = shown.get() {
                    var r = req.submit(code)
                    while case .wrong = r { r = req.submit(code) }
                    return
                }
                try? await Task.sleep(for: .milliseconds(20))
            }
        }
        defer { typer.cancel() }
        do {
            _ = try await a.node.pair(.hostPort(host: "127.0.0.1", port: relay.port!), alsoShare: true) { code, _ in shown.set(code) }
            XCTFail("a relay in the middle must not link")
        } catch LinkError.refused(let why) {
            XCTAssertTrue(why.contains("Wrong code"), why)
        }
        let codeA = try XCTUnwrap(shown.get()), codeB = try XCTUnwrap(b.requests.get().last?.code)
        XCTAssertNotEqual(codeA, codeB, "the two legs give two codes")
        XCTAssertTrue(a.node.peers.all.isEmpty)
        XCTAssertTrue(b.node.peers.all.isEmpty)
        // Positive control: the legs' exporters differ, and a code over the nonces alone is blind to it.
        let (e1, e2) = try XCTUnwrap(legs.get())
        XCTAssertNotEqual(e1, e2)
        let n1 = LinkPairing.nonce(), n2 = LinkPairing.nonce()
        XCTAssertEqual(LinkPairing.code(exporter: Data(), clientKey: Data(), serverKey: Data(), clientNonce: n1, serverNonce: n2),
                       LinkPairing.code(exporter: Data(), clientKey: Data(), serverKey: Data(), clientNonce: n1, serverNonce: n2))
        XCTAssertNotEqual(LinkPairing.code(exporter: e1, clientKey: a.node.identity.publicKey, serverKey: m.publicKey, clientNonce: n1, serverNonce: n2),
                          LinkPairing.code(exporter: e2, clientKey: m.publicKey, serverKey: b.node.identity.publicKey, clientNonce: n1, serverNonce: n2))
    }

    func testUnlinkedOnTheOtherSide() async throws {
        let a = try Node("A"), b = try Node("B")
        _ = try await pair(a, b)
        until("online") { a.node.state(of: b.id) == .online }
        // B forgets A without telling it (B's file was edited, or B unlinked while A was away).
        b.node.peers.remove(a.id)
        b.node.reconcile()
        a.node.stop()
        a.node.reconcile()
        until("A sees it is unlinked", timeout: 15) { a.node.state(of: b.id) == .unlinked }
        do { _ = try await a.node.call(b.id, "echo", timeout: 2); XCTFail() } catch {}
    }

    func testUnlinkTellsTheOtherMac() async throws {
        let a = try Node("A"), b = try Node("B")
        _ = try await pair(a, b)
        until("online") { a.node.state(of: b.id) == .online }
        await a.node.unlink(b.id)
        XCTAssertNil(a.node.peers.peer(b.id))
        until("B forgot A") { b.node.peers.peer(a.id) == nil }
    }

    func testGrantsAreEnforcedBySharer() async throws {
        let a = try Node("A"), b = try Node("B")
        _ = try await pair(a, b)
        until("online") { a.node.state(of: b.id) == .online }
        b.node.peers.update(a.id) { $0.allowFetch = false }
        _ = try await a.node.call(b.id, "echo", timeout: 5)
        do { _ = try await a.node.call(b.id, "fetch", timeout: 5); XCTFail("fetch is off") } catch LinkError.remote(let t) {
            XCTAssertTrue(t.contains("is off"), t)
        }
        b.node.peers.update(a.id) { $0.allowSearch = false }
        do { _ = try await a.node.call(b.id, "echo", timeout: 5); XCTFail("search is off") } catch LinkError.remote {}
        // A learns of the change when B pushes it.
        b.node.pushGrants(a.id)
        until("A sees B's grants") { a.node.peers.peer(b.id)?.theyAllowSearch == false }
    }

    func testOfflineFailsAtOnceAndTimeoutBounds() async throws {
        let a = try Node("A"), b = try Node("B")
        _ = try await pair(a, b)
        until("online") { a.node.state(of: b.id) == .online }
        // A request B never answers ends at the caller's deadline, and the link stays usable.
        let t0 = Date()
        do { _ = try await a.node.call(b.id, "sleep", timeout: 0.5); XCTFail() } catch LinkError.timeout {}
        XCTAssertLessThan(Date().timeIntervalSince(t0), 1.5)
        until("B's handler was cancelled when A gave up") { b.service.lock.withLock { b.service.cancelled } == 1 }
        _ = try await a.node.call(b.id, "echo", timeout: 5)
        // B gone: the next call fails at once instead of waiting.
        b.node.stop()
        until("A sees B offline", timeout: 15) { a.node.state(of: b.id) != .online }
        let t1 = Date()
        do { _ = try await a.node.call(b.id, "echo", timeout: 5); XCTFail() } catch LinkError.refused {}
        XCTAssertLessThan(Date().timeIntervalSince(t1), 0.05)
    }

    func testCancelReachesTheOtherEnd() async throws {
        let a = try Node("A"), b = try Node("B")
        _ = try await pair(a, b)
        until("online") { a.node.state(of: b.id) == .online }
        let id = b.id
        let t = Task { try await a.node.call(id, "sleep", timeout: 30) }
        until("B is serving it") { b.service.lock.withLock { b.service.calls.contains { $0.0 == "sleep" } } }
        t.cancel()
        do { _ = try await t.value; XCTFail() } catch LinkError.cancelled {}
        until("B's handler was cancelled") { b.service.lock.withLock { b.service.cancelled } == 1 }
    }

    func testManyConcurrentCallsAndBigBlobs() async throws {
        let a = try Node("A"), b = try Node("B")
        _ = try await pair(a, b)
        until("online") { a.node.state(of: b.id) == .online }
        let id = b.id, node = a.node
        let big = Data((0 ..< (8 << 20)).map { UInt8(truncatingIfNeeded: $0 &* 31) })
        try await withThrowingTaskGroup(of: Void.self) { g in
            for i in 0 ..< 40 {
                g.addTask {
                    let blob = i % 10 == 0 ? big : Data("\(i)".utf8)
                    let r = try await node.call(id, "echo", LinkMessage(body: Data("\(i)".utf8), blob: blob), timeout: 20)
                    XCTAssertEqual(r.body, Data("\(i)".utf8))
                    XCTAssertEqual(r.blob, blob)
                }
            }
            try await g.waitForAll()
        }
    }

    /// A byte flipped on the wire after the handshake: TLS's record authentication fails, the
    /// connection dies, and the call reports it - nothing altered gets through.
    func testFlippedByteKillsTheConnection() async throws {
        let a = try Node("A"), b = try Node("B")
        _ = try await pair(a, b)
        b.node.setPairingWindow(false)
        // Route A's link to B through a TCP proxy that can flip a byte.
        let proxy = try FlipProxy(to: b.node.port!)
        a.node.peers.update(b.id) { $0.host = "127.0.0.1"; $0.port = proxy.port }
        a.node.stop(); a.node.reconcile()
        until("online through the proxy", timeout: 15) { a.node.state(of: b.id) == .online }
        XCTAssertGreaterThan(proxy.bytes.get(), 0, "traffic is going through the proxy")
        _ = try await a.node.call(b.id, "echo", LinkMessage(body: Data("before".utf8)), timeout: 5)
        proxy.armed.set(true)
        do {
            _ = try await a.node.call(b.id, "echo", LinkMessage(body: Data(repeating: 7, count: 4096)), timeout: 5)
            XCTFail("a tampered record must not be delivered")
        } catch {
            XCTAssertTrue([LinkError.closed, .timeout].contains(error as? LinkError), "\(error)")
        }
        XCTAssertTrue(proxy.flipped.get(), "the proxy did flip a byte")
        proxy.stop()
    }

    /// A frozen Mac (SIGSTOP, a hung app) keeps its TCP connection up - the kernel answers for it -
    /// so only an unanswered ping notices. Simulated by a proxy that swallows everything.
    func testFrozenPeerGoesOffline() async throws {
        let saved = (LinkNode.pingInterval, LinkNode.pingTimeout)
        LinkNode.pingInterval = 0.3; LinkNode.pingTimeout = 0.5
        defer { (LinkNode.pingInterval, LinkNode.pingTimeout) = saved }
        let a = try Node("A"), b = try Node("B")
        _ = try await pair(a, b)
        b.node.setPairingWindow(false)
        let proxy = try FlipProxy(to: b.node.port!)
        a.node.peers.update(b.id) { $0.host = "127.0.0.1"; $0.port = proxy.port }
        a.node.stop(); a.node.reconcile()
        until("online through the proxy", timeout: 15) { a.node.state(of: b.id) == .online }
        proxy.blackhole.set(true)
        let t0 = Date()
        until("A notices B froze", timeout: 5) { a.node.state(of: b.id) != .online }
        XCTAssertLessThan(Date().timeIntervalSince(t0), 2.0, "within a ping interval and its timeout")
        proxy.stop()
    }

    /// A raw client channel to `port` with a fresh identity: what a stranger on the network has.
    private func stranger(_ port: UInt16) async throws -> (LinkChannel, LinkIdentity) {
        let me = try LinkIdentity(key: P256.Signing.PrivateKey())
        let q = DispatchQueue(label: "stranger")
        let conn = NWConnection(host: "127.0.0.1", port: NWEndpoint.Port(rawValue: port)!,
                                using: LinkTLS.parameters(identity: me, queue: q) { _ in true })
        let ready = Locked(false)
        conn.stateUpdateHandler = { if case .ready = $0 { ready.set(true) } }
        conn.start(queue: q)
        until("stranger connected") { ready.get() }
        let ch = LinkChannel(connection: conn, peerKey: LinkTLS.peerKey(conn)!, queue: q) { _, _ in throw LinkError.closed }
        ch.start()
        return (ch, me)
    }

    func testRevealMustMatchCommitment() async throws {
        let b = try Node("B")
        b.node.setPairingWindow(true)
        until("B listens") { b.node.port != nil }
        let (ch, _) = try await stranger(b.node.port!)
        defer { ch.close() }
        let n1 = LinkPairing.nonce(), n2 = LinkPairing.nonce()
        _ = try await ch.call("pair.begin", .json(PairBegin(name: "S", model: "", commit: LinkPairing.commitment(n1), share: true)), timeout: 5)
        do {
            _ = try await ch.call("pair.reveal", .json(PairReveal(nonce: n2)), timeout: 5)
            XCTFail("a reveal that is not the committed nonce must be refused")
        } catch LinkError.remote(let t) {
            XCTAssertTrue(t.contains("Pairing failed"), t)
        }
        XCTAssertTrue(b.requests.get().isEmpty, "no code was asked for")
    }

    func testStrangerGetsNothing() async throws {
        let b = try Node("B")
        b.node.setPairingWindow(true)
        until("B listens") { b.node.port != nil }
        let (ch, _) = try await stranger(b.node.port!)
        defer { ch.close() }
        for m in ["echo", "fetch", "hello", "unlink", "grants"] {
            do { _ = try await ch.call(m, timeout: 5); XCTFail("\(m) answered a stranger") } catch LinkError.remote(let t) {
                XCTAssertEqual(t, LinkNode.notLinked, m)
            }
        }
        XCTAssertTrue(b.service.calls.isEmpty, "the service never ran for a stranger")
    }

    /// Something else answering at a linked Mac's address (a new machine on its IP, an attacker)
    /// does not get A's requests: the pinned key does not match, so A never comes online.
    func testImpostorAtTheAddressIsRefused() async throws {
        let a = try Node("A"), b = try Node("B"), c = try Node("C")
        _ = try await pair(a, b)
        b.node.stop()
        // The impostor knows A's public key (it is public) and welcomes it, so if A did not check
        // the key at the other end, A would come online to C and send it A's searches.
        c.node.peers.put(LinkPeer(id: a.id, publicKey: a.node.identity.publicKey, name: "A", model: "",
                                  allowSearch: true, allowFetch: true, theyAllowSearch: true, theyAllowFetch: true))
        c.node.setPairingWindow(true)
        until("C listens") { c.node.port != nil }
        a.node.peers.update(b.id) { $0.host = "127.0.0.1"; $0.port = c.node.port! }
        a.node.stop(); a.node.reconcile()
        usleep(1_500_000)
        XCTAssertNotEqual(a.node.state(of: b.id), .online)
        do { _ = try await a.node.call(b.id, "echo", timeout: 2); XCTFail() } catch {}
        XCTAssertTrue(c.service.calls.isEmpty)
    }

    /// A link stays up past the time a connection gets to come up. The connect timer used to cancel
    /// the connection whether or not it had connected, so every link died 10 s in.
    func testLinkOutlivesConnectTimeout() async throws {
        let saved = LinkNode.connectTimeout
        LinkNode.connectTimeout = 0.5
        defer { LinkNode.connectTimeout = saved }
        let a = try Node("A"), b = try Node("B")
        _ = try await pair(a, b)
        a.node.stop(); a.node.reconcile()   // reconnect under the short timeout
        until("online") { a.node.state(of: b.id) == .online }
        try await Task.sleep(for: .seconds(1.5))
        XCTAssertEqual(a.node.state(of: b.id), .online, "still up three timeouts later")
        _ = try await a.node.call(b.id, "echo", timeout: 5)
    }

    /// A forgot B while B was away, so B still trusts A: linking again must work, and replace the link.
    func testRelinkWhenTheOtherMacStillTrustsThisOne() async throws {
        let a = try Node("A"), b = try Node("B")
        _ = try await pair(a, b)
        until("online") { a.node.state(of: b.id) == .online }
        a.node.peers.remove(b.id)
        a.node.reconcile()
        XCTAssertNotNil(b.node.peers.peer(a.id), "B still has A")
        let again = try await pair(a, b, share: false)
        XCTAssertEqual(again.id, b.id)
        XCTAssertEqual(b.node.peers.peer(a.id)?.theyAllowSearch, false, "the new link's grants replaced the old")
        until("online again") { a.node.state(of: b.id) == .online }
    }

    func testAddressParsing() {
        func s(_ e: NWEndpoint?) -> String? { e.map { "\($0)" } }
        XCTAssertEqual(s(LinkNode.endpoint(forAddress: "mac.local")), "mac.local:51298")
        XCTAssertEqual(s(LinkNode.endpoint(forAddress: " 100.64.1.2:6000 ")), "100.64.1.2:6000")
        XCTAssertEqual(s(LinkNode.endpoint(forAddress: "[fe80::1]:7000")), "fe80::1.7000")
        XCTAssertNotNil(LinkNode.endpoint(forAddress: "fe80::1"))
        XCTAssertNil(LinkNode.endpoint(forAddress: "host:notaport"))
        XCTAssertNil(LinkNode.endpoint(forAddress: ""))
    }

    func testRemotePath() {
        let p = RemotePath.make(peer: "abc", path: "/Users/x/Doc é.pdf")
        XCTAssertTrue(RemotePath.isRemote(p))
        XCTAssertEqual(RemotePath.parse(p)?.peer, "abc")
        XCTAssertEqual(RemotePath.parse(p)?.path, "/Users/x/Doc é.pdf")
        XCTAssertEqual((p as NSString).lastPathComponent, "Doc é.pdf")
        XCTAssertEqual((p as NSString).pathExtension, "pdf")
        XCTAssertNil(RemotePath.parse("/Users/x"))
        XCTAssertNil(RemotePath.parse("omni-link:///x"))
    }

    func testCodeShape() {
        let c = LinkPairing.code(exporter: Data([1]), clientKey: Data([2]), serverKey: Data([3]),
                                 clientNonce: Data([4]), serverNonce: Data([5]))
        XCTAssertEqual(c.count, 6)
        XCTAssertTrue(LinkPairing.matches(LinkPairing.display(c), c))
        XCTAssertTrue(LinkPairing.matches(String(c.prefix(3)) + "-" + c.suffix(3), c))
        XCTAssertFalse(LinkPairing.matches(String(c.dropLast()), c))
        // Every input moves it (length-prefixed, so shifting a byte between fields does too).
        let moved = LinkPairing.code(exporter: Data([1, 2]), clientKey: Data(), serverKey: Data([3]),
                                     clientNonce: Data([4]), serverNonce: Data([5]))
        XCTAssertNotEqual(c, moved)
    }
}

/// TCP proxy to 127.0.0.1:port that, once armed, flips one byte of the next client-to-server chunk.
final class FlipProxy: @unchecked Sendable {
    let listener: NWListener
    let armed = Locked(false), flipped = Locked(false), bytes = Locked(0), blackhole = Locked(false)
    private let q = DispatchQueue(label: "flip")
    var port: UInt16 { listener.port!.rawValue }
    init(to target: UInt16) throws {
        listener = try NWListener(using: .tcp, on: .any)
        listener.newConnectionHandler = { [q, armed, flipped, bytes, blackhole] down in
            let up = NWConnection(host: "127.0.0.1", port: NWEndpoint.Port(rawValue: target)!, using: .tcp)
            @Sendable func pump(_ from: NWConnection, _ to: NWConnection, flip: Bool) {
                from.receive(minimumIncompleteLength: 1, maximumLength: 65536) { data, _, done, err in
                    if var data, !data.isEmpty, !blackhole.get() {
                        bytes.set(bytes.get() + data.count)
                        if flip, armed.get(), !flipped.get() {
                            data[data.startIndex + data.count / 2] ^= 0x40
                            flipped.set(true)
                        }
                        to.send(content: data, completion: .idempotent)
                    }
                    if done || err != nil { to.cancel(); from.cancel(); return }
                    pump(from, to, flip: flip)
                }
            }
            up.start(queue: q); down.start(queue: q)
            pump(down, up, flip: true); pump(up, down, flip: false)
        }
        let ready = Locked(false)
        listener.stateUpdateHandler = { if case .ready = $0 { ready.set(true) } }
        listener.start(queue: q)
        for _ in 0 ..< 250 where !ready.get() { usleep(20_000) }
    }
    func stop() { listener.cancel() }
}
