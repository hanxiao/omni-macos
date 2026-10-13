import Foundation

/// A Mac this one is linked with. What THIS Mac grants it is decided here and nowhere else; what it
/// grants this Mac is only reported (`theyAllowSearch`), and enforced on its side.
public struct LinkPeer: Codable, Sendable, Identifiable, Equatable {
    public var id: String
    public var publicKey: Data
    public var name: String
    /// Hardware model ("MacBookPro18,3"), for the sidebar icon.
    public var model: String
    /// Where it was last reached, for when Bonjour cannot see it (another network over a VPN).
    public var host: String?
    public var port: UInt16?
    /// May it search this Mac?
    public var allowSearch: Bool
    /// May it open (fetch) the files this Mac's index holds?
    public var allowFetch: Bool
    /// What it last said it grants this Mac.
    public var theyAllowSearch: Bool
    public var theyAllowFetch: Bool
    public var linkedAt: Date

    public init(id: String, publicKey: Data, name: String, model: String, host: String? = nil, port: UInt16? = nil,
                allowSearch: Bool, allowFetch: Bool, theyAllowSearch: Bool, theyAllowFetch: Bool, linkedAt: Date = Date()) {
        self.id = id; self.publicKey = publicKey; self.name = name; self.model = model
        self.host = host; self.port = port
        self.allowSearch = allowSearch; self.allowFetch = allowFetch
        self.theyAllowSearch = theyAllowSearch; self.theyAllowFetch = theyAllowFetch
        self.linkedAt = linkedAt
    }
}

/// The linked Macs, in one 0600 JSON file. Thread-safe; every write persists.
public final class LinkPeerStore: @unchecked Sendable {
    private let url: URL
    private let lock = NSLock()
    private var peers: [LinkPeer]

    public init(url: URL) {
        self.url = url
        if let d = try? Data(contentsOf: url), let p = try? JSONDecoder().decode([LinkPeer].self, from: d) {
            peers = p
        } else {
            peers = []
        }
    }

    public var all: [LinkPeer] { lock.withLock { peers } }
    public func peer(_ id: String) -> LinkPeer? { lock.withLock { peers.first { $0.id == id } } }
    public func peer(key: Data) -> LinkPeer? { lock.withLock { peers.first { $0.publicKey == key } } }

    /// Insert or replace by id.
    public func put(_ p: LinkPeer) {
        lock.withLock {
            if let i = peers.firstIndex(where: { $0.id == p.id }) { peers[i] = p } else { peers.append(p) }
            save()
        }
    }

    @discardableResult
    public func update(_ id: String, _ f: (inout LinkPeer) -> Void) -> LinkPeer? {
        lock.withLock {
            guard let i = peers.firstIndex(where: { $0.id == id }) else { return nil }
            let before = peers[i]
            f(&peers[i])
            if peers[i] != before { save() }
            return peers[i]
        }
    }

    public func remove(_ id: String) {
        lock.withLock {
            peers.removeAll { $0.id == id }
            save()
        }
    }

    private func save() {
        guard let d = try? JSONEncoder().encode(peers) else { return }
        let fm = FileManager.default
        try? fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let tmp = url.appendingPathExtension("tmp")
        fm.createFile(atPath: tmp.path, contents: d, attributes: [.posixPermissions: 0o600])
        _ = try? fm.replaceItemAt(url, withItemAt: tmp)
        if !fm.fileExists(atPath: url.path) { try? fm.moveItem(at: tmp, to: url) }
    }
}

/// A linked Mac's file, as a path in this Mac's result list: `omni-link://<peer id>/<path on it>`.
/// The file name stays the last component, so everything that reads a name or an extension off a
/// path keeps working; nothing may treat it as a file on this Mac.
public enum RemotePath {
    public static let scheme = "omni-link://"

    public static func isRemote(_ path: String) -> Bool { path.hasPrefix(scheme) }

    public static func make(peer: String, path: String) -> String { scheme + peer + path }

    /// (peer id, path on that Mac), or nil for anything else.
    public static func parse(_ s: String) -> (peer: String, path: String)? {
        guard s.hasPrefix(scheme) else { return nil }
        let rest = s.dropFirst(scheme.count)
        guard let slash = rest.firstIndex(of: "/") else { return nil }
        let peer = String(rest[..<slash])
        guard !peer.isEmpty else { return nil }
        return (peer, String(rest[slash...]))
    }

    public static func peer(of s: String) -> String? { parse(s)?.peer }
}
