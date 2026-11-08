import CryptoKit
import Network
import XCTest
@testable import OmniLink

/// The link's state on disk and its implicit machines: what happens when a file cannot be read, a
/// save fails, a client walks away part-way, a call starts already cancelled.
final class LinkStateTests: XCTestCase, @unchecked Sendable {
    private func until(_ what: String, timeout: Double = 10, _ cond: () -> Bool) {
        let end = Date().addingTimeInterval(timeout)
        while !cond() && Date() < end { usleep(20_000) }
        XCTAssertTrue(cond(), what)
    }

    private func tempDir(_ name: String = "state") throws -> URL {
        let d = FileManager.default.temporaryDirectory.appendingPathComponent("link-\(name)-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        addTeardownBlock {
            try? FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: d.path)
            try? FileManager.default.removeItem(at: d)
        }
        return d
    }

    private func samplePeer(_ id: String = "p1") -> LinkPeer {
        LinkPeer(id: id, publicKey: Data([1, 2, 3]), name: id, model: "m", allowSearch: true, allowFetch: true,
                 theyAllowSearch: true, theyAllowFetch: true, linkedAt: Date(timeIntervalSince1970: 1000))
    }

    private func pair(_ a: Node, _ b: Node, share: Bool = true) async throws -> LinkPeer {
        b.node.setPairingWindow(true)
        until("B listens") { b.node.port != nil }
        let shown = Locked<String?>(nil)
        let typer = Task.detached {
            for _ in 0 ..< 500 {
                if let req = b.requests.get().last, let code = shown.get() {
                    _ = req.submit(code)
                    return
                }
                try? await Task.sleep(for: .milliseconds(20))
            }
        }
        defer { typer.cancel() }
        return try await a.node.pair(b.endpoint, alsoShare: share) { code, _ in shown.set(code) }
    }

    /// A raw client channel with a fresh identity: what a stranger on the network has.
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

    private func begin(_ ch: LinkChannel, _ nonce: Data) async throws {
        _ = try await ch.call("pair.begin", .json(PairBegin(name: "S", model: "", commit: LinkPairing.commitment(nonce), share: true)), timeout: 5)
    }

    // MARK: - The identity file

    /// A file that is there and cannot be read is not a file that is absent: the key stays, untouched,
    /// and nothing is made in its place (every linked Mac pins it).
    func testAnUnreadableIdentityIsNotReplaced() throws {
        let dir = try tempDir()
        let url = dir.appendingPathComponent("identity")
        let first = try LinkIdentity.loadOrCreate(at: url)
        let bytes = try Data(contentsOf: url)
        try FileManager.default.setAttributes([.posixPermissions: 0o000], ofItemAtPath: url.path)
        defer { try? FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path) }
        XCTAssertThrowsError(try LinkIdentity.loadOrCreate(at: url), "no new key while the old one cannot be read")
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
        XCTAssertEqual(try Data(contentsOf: url), bytes, "the file is as it was")
        XCTAssertEqual(try LinkIdentity.loadOrCreate(at: url).id, first.id, "and the key comes back with it")
    }

    /// Readable and not a key: what was found is kept aside, and only then a new key is made.
    func testADamagedIdentityIsKeptAside() throws {
        let dir = try tempDir()
        let url = dir.appendingPathComponent("identity")
        let junk = Data("not json".utf8)
        try junk.write(to: url)
        let made = try LinkIdentity.loadOrCreate(at: url)
        let names = try FileManager.default.contentsOfDirectory(atPath: dir.path)
        let aside = names.filter { $0.hasPrefix("identity.unreadable") }
        XCTAssertEqual(aside.count, 1, "kept aside: \(names)")
        XCTAssertEqual(try Data(contentsOf: dir.appendingPathComponent(XCTUnwrap(aside.first))), junk)
        XCTAssertEqual(try LinkIdentity.loadOrCreate(at: url).id, made.id, "the new key is the key from now on")
    }

    func testAMissingIdentityIsMadeAndKept() throws {
        let dir = try tempDir()
        let url = dir.appendingPathComponent("sub/identity")
        let made = try LinkIdentity.loadOrCreate(at: url)
        XCTAssertEqual(try LinkIdentity.loadOrCreate(at: url).id, made.id)
        let mode = try FileManager.default.attributesOfItem(atPath: url.path)[.posixPermissions] as? NSNumber
        XCTAssertEqual(mode?.intValue, 0o600)
        XCTAssertFalse(FileManager.default.fileExists(atPath: url.path + ".tmp"), "no temp file left")
    }

    // MARK: - The peers file

    /// An undecodable file reads as no Macs, and the first pairing used to write over every link.
    func testUndecodablePeersAreKeptAsideBeforeTheFirstWrite() throws {
        let dir = try tempDir()
        let url = dir.appendingPathComponent("peers.json")
        let junk = Data("[{\"id\": \"future\", \"truncated".utf8)
        try junk.write(to: url)
        let store = LinkPeerStore(url: url)
        XCTAssertTrue(store.all.isEmpty)
        try? store.put(samplePeer())
        XCTAssertEqual(LinkPeerStore(url: url).all.map(\.id), ["p1"], "the new link is saved")
        let aside = try FileManager.default.contentsOfDirectory(atPath: dir.path).filter { $0.hasPrefix("peers.json.unreadable") }
        XCTAssertEqual(aside.count, 1, "what was there is kept aside")
        XCTAssertEqual(try Data(contentsOf: dir.appendingPathComponent(XCTUnwrap(aside.first))), junk)
    }

    /// A file that cannot be read at this moment is never written over, and comes back when it can be.
    func testUnreadablePeersAreNeverOverwritten() throws {
        let dir = try tempDir()
        let url = dir.appendingPathComponent("peers.json")
        try? LinkPeerStore(url: url).put(samplePeer("old"))
        let bytes = try Data(contentsOf: url)
        try FileManager.default.setAttributes([.posixPermissions: 0o000], ofItemAtPath: url.path)
        let store = LinkPeerStore(url: url)
        XCTAssertTrue(store.all.isEmpty)
        XCTAssertThrowsError(try store.put(samplePeer("new")), "a pairing cannot be saved over a file nobody could read")
        store.remove("old")
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
        XCTAssertEqual(try Data(contentsOf: url), bytes, "the file is as it was")
        // Readable again: the next write starts from what the file holds.
        try? store.put(samplePeer("new"))
        XCTAssertEqual(Set(LinkPeerStore(url: url).all.map(\.id)), ["old", "new"])
    }

    /// A pairing that cannot be saved fails where the user can see it; a revoke that cannot be saved
    /// still holds for the session, says so, and reaches the disk once the disk takes it.
    func testAFailedSaveIsNotSilent() throws {
        let dir = try tempDir()
        let url = dir.appendingPathComponent("peers.json")
        let store = LinkPeerStore(url: url)
        try? store.put(samplePeer("kept"))
        try FileManager.default.setAttributes([.posixPermissions: 0o555], ofItemAtPath: dir.path)
        defer { try? FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: dir.path) }
        XCTAssertThrowsError(try store.put(samplePeer("new")), "the disk refused it")
        XCTAssertNil(store.peer("new"), "memory does not claim a link the disk refused")
        store.update("kept") { $0.allowSearch = false }
        XCTAssertEqual(store.peer("kept")?.allowSearch, false, "a revoke holds in memory whatever the disk says")
        XCTAssertEqual(LinkPeerStore(url: url).peer("kept")?.allowSearch, true, "the disk has not got it")
        XCTAssertNotNil(store.saveFailure, "and the store says so")
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: dir.path)
        store.update("kept") { $0.allowSearch = false }   // no change, and still the next chance to save
        XCTAssertNil(store.saveFailure)
        store.update("kept") { $0.allowFetch = false }
        let again = LinkPeerStore(url: url).peer("kept")
        XCTAssertEqual(again?.allowSearch, false, "the revoke reached the disk")
        XCTAssertEqual(again?.allowFetch, false)
    }

    /// What a save writes is whole and private; no temp file stays behind.
    func testASaveLeavesOneWholePrivateFile() throws {
        let dir = try tempDir()
        let url = dir.appendingPathComponent("peers.json")
        let store = LinkPeerStore(url: url)
        for i in 0 ..< 5 { try? store.put(samplePeer("p\(i)")) }
        XCTAssertEqual(LinkPeerStore(url: url).all.count, 5)
        let mode = try FileManager.default.attributesOfItem(atPath: url.path)[.posixPermissions] as? NSNumber
        XCTAssertEqual(mode?.intValue, 0o600)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: dir.path), ["peers.json"])
    }

    /// An index moved before moves carried the link folder: the next launch finds the key and the
    /// peers again, and a Mac that linked anew at the new place keeps what it has.
    func testALinkFolderLeftBehindIsAdopted() throws {
        let root = try tempDir()
        let old = root.appendingPathComponent("old"), new = root.appendingPathComponent("new")
        let key = try LinkIdentity.loadOrCreate(at: old.appendingPathComponent("identity"))
        try LinkPeerStore(url: old.appendingPathComponent("peers.json")).put(samplePeer("kept"))
        XCTAssertTrue(LinkFile.adopt(from: old, to: new))
        XCTAssertEqual(try LinkIdentity.loadOrCreate(at: new.appendingPathComponent("identity")).id, key.id)
        XCTAssertEqual(LinkPeerStore(url: new.appendingPathComponent("peers.json")).all.map(\.id), ["kept"])
        XCTAssertTrue(FileManager.default.fileExists(atPath: old.appendingPathComponent("identity").path), "the old copy stays")
        // Second time, and with a key of its own: nothing happens.
        try? LinkPeerStore(url: old.appendingPathComponent("peers.json")).put(samplePeer("later"))
        XCTAssertFalse(LinkFile.adopt(from: old, to: new))
        XCTAssertEqual(LinkPeerStore(url: new.appendingPathComponent("peers.json")).all.map(\.id), ["kept"])
        // Nothing to adopt, and the same folder.
        XCTAssertFalse(LinkFile.adopt(from: root.appendingPathComponent("none"), to: root.appendingPathComponent("other")))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("other").path))
        XCTAssertFalse(LinkFile.adopt(from: old, to: old))
    }

    /// With OMNI_LINK_FULL_DIR, a folder on a small volume: a save that does not fit fails, says so,
    /// and leaves the file it would have replaced. Not in the ordinary run, which has no such volume.
    func testAFullDiskLeavesTheOldFile() throws {
        guard let dir = ProcessInfo.processInfo.environment["OMNI_LINK_FULL_DIR"] else { throw XCTSkip("no small volume") }
        let fm = FileManager.default
        try fm.createDirectory(atPath: dir, withIntermediateDirectories: true)
        let url = URL(fileURLWithPath: dir).appendingPathComponent("peers.json")
        let store = LinkPeerStore(url: url)
        try store.put(samplePeer("kept"))
        let bytes = try Data(contentsOf: url)
        // To the brim.
        let fill = URL(fileURLWithPath: dir).appendingPathComponent("fill")
        fm.createFile(atPath: fill.path, contents: nil)
        let fh = try FileHandle(forWritingTo: fill)
        let block = Data(repeating: 7, count: 1 << 20)
        while (try? fh.write(contentsOf: block)) != nil {}
        let small = Data(repeating: 7, count: 512)
        while (try? fh.write(contentsOf: small)) != nil {}
        try? fh.close()
        defer { try? fm.removeItem(at: fill) }
        var big = samplePeer("new"); big.name = String(repeating: "x", count: 4 << 20)
        XCTAssertThrowsError(try store.put(big))
        XCTAssertNil(store.peer("new"))
        XCTAssertEqual(try Data(contentsOf: url), bytes, "the old file is whole")
        XCTAssertFalse(fm.fileExists(atPath: url.path + ".tmp"), "no part of the new one is left")
    }

    // MARK: - Pairing, the answering side

    /// A malformed reveal must not hold the slot: the next Mac that asks gets to pair.
    func testAMalformedRevealFreesTheSlot() async throws {
        let a = try Node("A"), b = try Node("B")
        b.node.setPairingWindow(true)
        until("B listens") { b.node.port != nil }
        let (ch, _) = try await stranger(b.node.port!)
        defer { ch.close() }
        try await begin(ch, LinkPairing.nonce())
        do { _ = try await ch.call("pair.reveal", LinkMessage(body: Data("not json".utf8)), timeout: 5); XCTFail() } catch {}
        let peer = try await pair(a, b)
        XCTAssertEqual(peer.id, b.id, "the slot was left held")
    }

    /// Frames that overlap, as only a hostile client would send them: each runs in its own task, and
    /// the walk through pairing is read and written on the link's queue, so one begin pops one request
    /// and the slot is free again once the client goes.
    func testOverlappingFramesPopOneRequestAtMost() async throws {
        let a = try Node("A"), b = try Node("B")
        b.node.setPairingWindow(true)
        until("B listens") { b.node.port != nil }
        let (ch, _) = try await stranger(b.node.port!)
        let n = LinkPairing.nonce()
        let beginMsg = try LinkMessage.json(PairBegin(name: "S", model: "", commit: LinkPairing.commitment(n), share: true))
        let revealMsg = try LinkMessage.json(PairReveal(nonce: n))
        await withTaskGroup(of: Void.self) { g in
            for i in 0 ..< 40 {
                let (method, msg) = i % 2 == 0 ? ("pair.begin", beginMsg) : ("pair.reveal", revealMsg)
                g.addTask { _ = try? await ch.call(method, msg, timeout: 1) }
            }
        }
        XCTAssertLessThanOrEqual(b.requests.get().count, 1)
        b.requests.get().forEach { $0.cancel() }
        ch.close()
        try await Task.sleep(for: .milliseconds(200))
        let peer = try await pair(a, b)
        XCTAssertEqual(peer.id, b.id, "the slot came back")
    }

    /// One client's close frees only its own slot.
    func testOneClientsCloseDoesNotFreeAnothersSlot() async throws {
        let b = try Node("B")
        b.node.setPairingWindow(true)
        until("B listens") { b.node.port != nil }
        // The first client begins and fails its reveal, which frees the slot; its close stays armed.
        let (first, _) = try await stranger(b.node.port!)
        try await begin(first, LinkPairing.nonce())
        do { _ = try await first.call("pair.reveal", .json(PairReveal(nonce: LinkPairing.nonce())), timeout: 5); XCTFail() } catch {}
        // The second holds it now.
        let (second, _) = try await stranger(b.node.port!)
        defer { second.close() }
        try await begin(second, LinkPairing.nonce())
        first.close()
        try await Task.sleep(for: .milliseconds(200))
        // So a third hears busy.
        let (third, _) = try await stranger(b.node.port!)
        defer { third.close() }
        do {
            try await begin(third, LinkPairing.nonce())
            XCTFail("the first client's close freed the second one's slot")
        } catch LinkError.remote(let t) {
            XCTAssertTrue(t.contains("busy"), t)
        }
    }

    /// A reveal is gated on the window as a begin is: nobody pops a sheet on a Mac that is not pairing.
    func testARevealAfterTheWindowClosedPopsNothing() async throws {
        let b = try Node("B")
        b.node.setPairingWindow(true)
        until("B listens") { b.node.port != nil }
        // A linked Mac keeps B listening with its window closed.
        try? b.node.peers.put(LinkPeer(id: "x", publicKey: Data([1]), name: "x", model: "", allowSearch: false,
                                       allowFetch: false, theyAllowSearch: false, theyAllowFetch: false))
        let (ch, _) = try await stranger(b.node.port!)
        defer { ch.close() }
        let n = LinkPairing.nonce()
        try await begin(ch, n)
        b.node.setPairingWindow(false)
        try await Task.sleep(for: .milliseconds(100))
        do {
            _ = try await ch.call("pair.reveal", .json(PairReveal(nonce: n)), timeout: 1)
            XCTFail("refused")
        } catch {}
        try await Task.sleep(for: .milliseconds(100))
        XCTAssertTrue(b.requests.get().isEmpty, "no sheet on a Mac whose window is closed")
    }

    /// A request that ended before anyone was listening still reports its end: the sheet that listens
    /// late would otherwise stay up with nothing able to close it.
    func testARequestThatEndedBeforeItWasShownSaysSo() {
        let req = IncomingPairing(name: "S", model: "", code: "123456")
        req.finish(.cancelled)
        let heard = Locked<IncomingPairing.Outcome?>(nil)
        req.onEnd = { heard.set($0) }
        XCTAssertEqual(heard.get(), .cancelled)
        XCTAssertEqual(req.submit("123456"), .over)
    }

    // MARK: - Pairing, the asking side

    /// Cancelled during connect, it stops at once and nothing reaches the other Mac.
    func testCancelDuringConnectStopsThePairing() async throws {
        let a = try Node("A"), b = try Node("B")
        b.node.setPairingWindow(true)
        until("B listens") { b.node.port != nil }
        // A proxy that takes the TCP connection and swallows everything: TLS never finishes.
        let proxy = try FlipProxy(to: b.node.port!)
        proxy.blackhole.set(true)
        defer { proxy.stop() }
        let node = a.node
        let ep = NWEndpoint.hostPort(host: "127.0.0.1", port: NWEndpoint.Port(rawValue: proxy.port)!)
        let t = Task { try await node.pair(ep, alsoShare: true) { _, _ in } }
        try await Task.sleep(for: .milliseconds(300))
        let t0 = Date()
        t.cancel()
        do { _ = try await t.value; XCTFail("cancelled") } catch {}
        XCTAssertLessThan(Date().timeIntervalSince(t0), 1, "cancel does not wait out the connect")
    }

    /// Cancelled before it started: no begin, no reveal, no sheet on the other Mac.
    func testAPairingCancelledAtOnceSendsNothing() async throws {
        let a = try Node("A"), b = try Node("B")
        b.node.setPairingWindow(true)
        until("B listens") { b.node.port != nil }
        let node = a.node, ep = b.endpoint
        let t = Task { () -> LinkPeer in
            withUnsafeCurrentTask { $0?.cancel() }
            return try await node.pair(ep, alsoShare: true) { _, _ in }
        }
        do { _ = try await t.value; XCTFail("cancelled") } catch {}
        try await Task.sleep(for: .milliseconds(300))
        XCTAssertTrue(b.requests.get().isEmpty, "the other Mac's sheet did not open")
    }

    // MARK: - The channel

    /// A call started on a task already cancelled is not sent: a superseded search or thumbnail must
    /// not run on the other Mac.
    func testACallOnACancelledTaskIsNotSent() async throws {
        let a = try Node("A"), b = try Node("B")
        _ = try await pair(a, b)
        until("online") { a.node.state(of: b.id) == .online }
        let id = b.id, node = a.node
        for _ in 0 ..< 50 {
            let t = Task { () -> LinkMessage in
                withUnsafeCurrentTask { $0?.cancel() }
                return try await node.call(id, "echo", LinkMessage(), timeout: 5)
            }
            do { _ = try await t.value; XCTFail("cancelled") } catch { XCTAssertEqual(error as? LinkError, .cancelled) }
        }
        try await Task.sleep(for: .milliseconds(200))
        XCTAssertEqual(b.service.lock.withLock { b.service.calls.filter { $0.0 == "echo" }.count }, 0, "nothing ran on the other Mac")
        _ = try await node.call(id, "echo", timeout: 5)
    }

    // MARK: - The connection

    /// The other Mac coming back and dialling this one revives a row that said Unlinked: it trusts
    /// this Mac again, or it would not have said hello.
    func testUnlinkedIsRevivedByTheOtherMacsHello() async throws {
        // No Bonjour: seeing the other Mac appear revives a row too, and would hide what is under test.
        let a = try Node("A", bonjour: false), b = try Node("B", bonjour: false)
        _ = try await pair(a, b)
        until("online") { a.node.state(of: b.id) == .online && b.node.state(of: a.id) == .online }
        let forgotten = try XCTUnwrap(b.node.peers.peer(a.id))
        b.node.peers.remove(a.id)
        b.node.reconcile()
        a.node.stop()
        a.node.reconcile()
        until("A sees it is unlinked", timeout: 15) { a.node.state(of: b.id) == .unlinked }
        // B's file comes back (a restore), and B dials A, at the port A listens on now: these nodes
        // take any free port, where a real Mac has a fixed one.
        var restored = forgotten
        restored.host = "127.0.0.1"; restored.port = a.node.port
        try? b.node.peers.put(restored)
        b.node.reconcile()
        until("A is online again, with no Reconnect", timeout: 15) { a.node.state(of: b.id) == .online }
    }

    /// A lost hello, or a Mac that cannot dial this one: the row asks again by itself.
    func testUnlinkedAsksAgainByItself() async throws {
        let saved = LinkNode.unlinkedRetry
        LinkNode.unlinkedRetry = 0.5
        defer { LinkNode.unlinkedRetry = saved }
        let a = try Node("A", bonjour: false), b = try Node("B", bonjour: false)
        _ = try await pair(a, b)
        until("online") { a.node.state(of: b.id) == .online }
        let forgotten = try XCTUnwrap(b.node.peers.peer(a.id))
        b.node.peers.remove(a.id)
        b.node.reconcile()
        a.node.stop()
        a.node.reconcile()
        until("A sees it is unlinked", timeout: 15) { a.node.state(of: b.id) == .unlinked }
        // B trusts A again and has no way to tell it: a quiet restore, with no dial.
        try? b.node.peers.put(forgotten)
        until("A is online again", timeout: 15) { a.node.state(of: b.id) == .online }
    }

    /// A begin and then silence does not hold the one pairing slot for as long as it stays connected.
    func testABeginThatGoesSilentFreesTheSlot() async throws {
        let saved = LinkNode.pairingBeginTimeout
        LinkNode.pairingBeginTimeout = 0.5
        defer { LinkNode.pairingBeginTimeout = saved }
        let a = try Node("A"), b = try Node("B")
        b.node.setPairingWindow(true)
        until("B listens") { b.node.port != nil }
        let (ch, _) = try await stranger(b.node.port!)
        defer { ch.close() }
        try await begin(ch, LinkPairing.nonce())
        try await Task.sleep(for: .milliseconds(800))
        let peer = try await pair(a, b)
        XCTAssertEqual(peer.id, b.id, "the slot was left held")
    }

    /// Two Macs that ask each other at once: this Mac's own pairing keeps a request out, rather than
    /// each hiding behind a request the code the other needs.
    func testAMacPairingOutHearsNoRequestIn() async throws {
        let a = try Node("A"), b = try Node("B"), c = try Node("C")
        b.node.setPairingWindow(true); a.node.setPairingWindow(true)
        until("listening") { b.node.port != nil && a.node.port != nil }
        let shown = Locked<String?>(nil)
        let node = a.node, ep = b.endpoint
        let going = Task { try await node.pair(ep, alsoShare: true) { code, _ in shown.set(code) } }
        until("A shows a code") { shown.get() != nil }
        do {
            _ = try await c.node.pair(a.endpoint, alsoShare: true) { _, _ in }
            XCTFail("A is busy")
        } catch LinkError.remote(let t) {
            XCTAssertTrue(t.contains("busy"), t)
        }
        XCTAssertTrue(a.requests.get().isEmpty)
        going.cancel()
        _ = try? await going.value
        // And when it is over, A answers again.
        try await Task.sleep(for: .milliseconds(200))
        let (ch, _) = try await stranger(a.node.port!)
        defer { ch.close() }
        try await begin(ch, LinkPairing.nonce())
    }

    /// A stranger with no pairing in hand does not keep its connection.
    func testAStrangerWithNothingToSayIsClosed() async throws {
        let saved = LinkNode.strangerTimeout
        LinkNode.strangerTimeout = 0.5
        defer { LinkNode.strangerTimeout = saved }
        let b = try Node("B")
        b.node.setPairingWindow(true)
        until("B listens") { b.node.port != nil }
        let (ch, _) = try await stranger(b.node.port!)
        defer { ch.close() }
        _ = try? await ch.call("ping", timeout: 2)   // the handshake is done either way
        until("closed", timeout: 5) { ch.isClosed }
    }

    /// While it pairs it keeps its connection past that limit: a person is typing the code.
    func testAStrangerThatIsPairingKeepsItsConnection() async throws {
        let saved = LinkNode.strangerTimeout
        LinkNode.strangerTimeout = 0.2
        defer { LinkNode.strangerTimeout = saved }
        let a = try Node("A"), b = try Node("B")
        b.node.setPairingWindow(true)
        until("B listens") { b.node.port != nil }
        let shown = Locked<String?>(nil)
        let typer = Task.detached {
            while shown.get() == nil || b.requests.get().isEmpty { try? await Task.sleep(for: .milliseconds(20)) }
            try? await Task.sleep(for: .milliseconds(800))   // past the limit
            _ = b.requests.get().last?.submit(shown.get() ?? "")
        }
        defer { typer.cancel() }
        let peer = try await a.node.pair(b.endpoint, alsoShare: true) { code, _ in shown.set(code) }
        XCTAssertEqual(peer.id, b.id)
    }

    /// Past the cap the oldest gives way, so nobody can fill the Mac with connections.
    func testStrangersAreCapped() async throws {
        let saved = LinkNode.maxStrangers
        LinkNode.maxStrangers = 3
        defer { LinkNode.maxStrangers = saved }
        let b = try Node("B")
        b.node.setPairingWindow(true)
        until("B listens") { b.node.port != nil }
        var chans: [LinkChannel] = []
        for _ in 0 ..< 5 {
            let (ch, _) = try await stranger(b.node.port!)
            _ = try? await ch.call("ping", timeout: 2)
            chans.append(ch)
        }
        defer { chans.forEach { $0.close() } }
        until("the two oldest gave way", timeout: 5) { chans[0].isClosed && chans[1].isClosed }
        XCTAssertFalse(chans[4].isClosed)
    }

    /// A stopped-for-good node stays down, and nothing writes through it: its folder is no longer the
    /// link's. A late unlink or pair used to start it again, on the old folder's file.
    func testARetiredNodeStaysDownAndWritesNothing() async throws {
        let a = try Node("A"), b = try Node("B")
        _ = try await pair(a, b)
        until("online") { a.node.state(of: b.id) == .online }
        a.node.retire()
        XCTAssertNil(a.node.port)
        a.node.reconcile()
        try await Task.sleep(for: .milliseconds(300))
        XCTAssertNil(a.node.port, "reconcile did not start it")
        XCTAssertNotEqual(a.node.state(of: b.id), .online)
        await a.node.unlink(b.id)
        XCTAssertNotNil(LinkPeerStore(url: a.dir.appendingPathComponent("peers.json")).peer(b.id), "the unlink wrote nothing")
        XCTAssertNil(a.node.port, "and did not start it")
        b.node.setPairingWindow(true)
        do {
            _ = try await a.node.pair(b.endpoint, alsoShare: true) { _, _ in }
            XCTFail("a retired node does not pair")
        } catch {}
    }

    /// The service's generation reaches the other Mac in the ping's reply: an engine that arrived or
    /// a model that switched is seen with no flap of the link.
    func testTheServiceGenerationComesWithThePing() async throws {
        let saved = (LinkNode.pingInterval, LinkNode.pingTimeout)
        LinkNode.pingInterval = 0.2; LinkNode.pingTimeout = 2
        defer { (LinkNode.pingInterval, LinkNode.pingTimeout) = saved }
        let a = try Node("A"), b = try Node("B")
        _ = try await pair(a, b)
        until("online") { a.node.state(of: b.id) == .online }
        until("A sees B's first generation", timeout: 5) { a.node.serviceGeneration(of: b.id) == 0 }
        b.service.generation.set(7)
        until("A sees B's new generation", timeout: 5) { a.node.serviceGeneration(of: b.id) == 7 }
        XCTAssertEqual(a.node.state(of: b.id), .online, "the link did not flap")
    }

    /// Liveness also repairs what a lost push left stale: the ping's reply says what that Mac grants.
    func testAGrantsPushThatNeverArrivedIsRepairedByThePing() async throws {
        let saved = (LinkNode.pingInterval, LinkNode.pingTimeout)
        LinkNode.pingInterval = 0.2; LinkNode.pingTimeout = 2
        defer { (LinkNode.pingInterval, LinkNode.pingTimeout) = saved }
        let a = try Node("A"), b = try Node("B")
        _ = try await pair(a, b)
        until("online") { a.node.state(of: b.id) == .online }
        b.node.peers.update(a.id) { $0.allowSearch = false; $0.allowFetch = false }
        until("A sees B's grants with no push", timeout: 5) { a.node.peers.peer(b.id)?.theyAllowSearch == false }
    }

    /// With Local Network access denied the browser waits with a policy error and never fails, so the
    /// sheet spun for good. What it says instead, and that an ordinary state says nothing.
    func testADeniedLocalNetworkIsSaid() {
        let denied = LinkNode.discoveryProblem(.waiting(.dns(DNSServiceErrorType(kDNSServiceErr_PolicyDenied))))
        XCTAssertNotNil(denied)
        XCTAssertTrue(denied?.contains("Local Network") == true, denied ?? "")
        XCTAssertNotNil(LinkNode.discoveryProblem(.failed(.posix(.ENETDOWN))))
        XCTAssertNil(LinkNode.discoveryProblem(.ready))
        XCTAssertNil(LinkNode.discoveryProblem(.setup))
    }

}
