import CryptoKit
import Foundation
import OmniLink

// linktool: a linked Mac with no app, for testing a real Omni's answers over the link
// (docs/notes/link.md). It is a peer like any other - it pairs by being written into the
// other Mac's peers file while that Mac is quit - so every request goes through the app's own
// LinkNode checks and LinkResponder, exactly as another Mac's would.
//
//   linktool seed <toolDir> <appLinkDir> <host> <port> [grantFetch=1]
//       Make the tool's identity in <toolDir>, link it with the app whose link folder is
//       <appLinkDir> (quit that app first), and point the tool at <host>:<port>.
//   linktool call <toolDir> <method> '<json body>' [blobOut]
//       Call <method> on the app and print the reply (body JSON; blob size and SHA-256).

struct NoService: LinkService {
    func serve(_ method: String, _ msg: LinkMessage, peer: LinkPeer) async throws -> LinkMessage {
        throw LinkError.refused("linktool serves nothing")
    }
}

func fail(_ s: String) -> Never { FileHandle.standardError.write(Data((s + "\n").utf8)); exit(1) }

let args = CommandLine.arguments
guard args.count >= 3 else { fail("usage: linktool seed|call ...") }
let toolDir = URL(fileURLWithPath: args[2], isDirectory: true)

switch args[1] {
case "seed":
    guard args.count >= 6, let port = UInt16(args[5]) else { fail("linktool seed <toolDir> <appLinkDir> <host> <port> [grantFetch]") }
    let appDir = URL(fileURLWithPath: args[3], isDirectory: true)
    let grantFetch = args.count < 7 || args[6] != "0"
    let me = try LinkIdentity.loadOrCreate(at: toolDir.appendingPathComponent("identity"))
    // The app's public key, from its identity file (this is a test: the tool may read it).
    let app = try LinkIdentity.loadOrCreate(at: appDir.appendingPathComponent("identity"))
    LinkPeerStore(url: appDir.appendingPathComponent("peers.json")).put(LinkPeer(
        id: me.id, publicKey: me.publicKey, name: "linktool", model: "tool",
        allowSearch: true, allowFetch: grantFetch, theyAllowSearch: false, theyAllowFetch: false))
    LinkPeerStore(url: toolDir.appendingPathComponent("peers.json")).put(LinkPeer(
        id: app.id, publicKey: app.publicKey, name: "app", model: "", host: args[4], port: port,
        allowSearch: false, allowFetch: false, theyAllowSearch: true, theyAllowFetch: grantFetch))
    print("tool \(me.id) linked with app \(app.id)")

case "call":
    guard args.count >= 5 else { fail("linktool call <toolDir> <method> <json> [blobOut]") }
    let me = try LinkIdentity.loadOrCreate(at: toolDir.appendingPathComponent("identity"))
    let peers = LinkPeerStore(url: toolDir.appendingPathComponent("peers.json"))
    guard let app = peers.all.first else { fail("not seeded") }
    let node = LinkNode(identity: me, peers: peers, config: .init(name: "linktool", model: "tool", version: "0", port: 0, bonjour: false),
                        service: NoService())
    node.reconcile()
    let deadline = Date().addingTimeInterval(15)
    while node.state(of: app.id) != .online && Date() < deadline { usleep(50_000) }
    guard node.state(of: app.id) == .online else { fail("the app did not come online (\(node.state(of: app.id)))") }
    let sem = DispatchSemaphore(value: 0)
    let out = Locked<String>("")
    let method = args[3], body = Data(args[4].utf8), blobOut = args.count > 5 ? args[5] : nil
    // Detached: top-level code is on the main actor, which sem.wait() below is blocking.
    Task.detached {
        do {
            let t0 = Date()
            let r = try await node.call(app.id, method, LinkMessage(body: body), timeout: 60)
            let sha = SHA256.hash(data: r.blob).map { String(format: "%02x", $0) }.joined()
            if let blobOut { try? r.blob.write(to: URL(fileURLWithPath: blobOut)) }
            out.set(String(format: "ok %.1fms blob=%d sha256=%@\n", -t0.timeIntervalSinceNow * 1000, r.blob.count, sha)
                    + String(decoding: r.body, as: UTF8.self))
        } catch {
            out.set("error \(error)")
        }
        sem.signal()
    }
    sem.wait()
    print(out.get())
    node.stop()

default:
    fail("unknown command \(args[1])")
}

final class Locked<T>: @unchecked Sendable {
    private var v: T
    private let l = NSLock()
    init(_ v: T) { self.v = v }
    func get() -> T { l.lock(); defer { l.unlock() }; return v }
    func set(_ n: T) { l.lock(); v = n; l.unlock() }
}
