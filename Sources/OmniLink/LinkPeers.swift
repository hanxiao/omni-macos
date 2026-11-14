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
///
/// WHAT CANNOT BE READ IS NOT "NO MACS". An undecodable or unreadable file used to give an empty list
/// that the next pairing saved over every link, and pairing is what anyone does when the sidebar has
/// just gone empty. A file that reads and does not decode is kept aside; one that will not read is
/// left alone, nothing is written while it stays so, and each write tries it again first.
///
/// AND A SAVE THAT FAILS IS NOT A SAVE. A new link that the disk refuses fails where it was asked
/// for (`put` throws, memory unchanged), since the other Mac would go on trusting a Mac that forgets
/// it at the next launch. A revoke or an unlink holds in memory whatever the disk says, because that
/// is the safe side for this session; `saveFailure` says it is not on disk, and every later write,
/// and `retrySave`, tries again.
public final class LinkPeerStore: @unchecked Sendable {
    private let url: URL
    private let lock = NSLock()
    private var peers: [LinkPeer]
    /// Why the file could not be read, while it cannot: nothing is written over it.
    private var unreadable: String?
    private var unsaved: String?

    public init(url: URL) {
        self.url = url
        peers = []
        load()
    }

    /// Reads the file into `peers`, or says why not. Called with the lock held, or from init.
    private func load() {
        switch LinkFile.read(url) {
        case .absent:
            unreadable = nil
        case .read(let d):
            if let p = try? JSONDecoder().decode([LinkPeer].self, from: d) {
                peers = p
                unreadable = nil
            } else {
                do { try LinkFile.keepAside(url); unreadable = nil }
                catch { unreadable = "\(error)" }
            }
        case .unreadable(let why):
            unreadable = why
        }
    }

    public var all: [LinkPeer] { lock.withLock { peers } }
    public func peer(_ id: String) -> LinkPeer? { lock.withLock { peers.first { $0.id == id } } }
    public func peer(key: Data) -> LinkPeer? { lock.withLock { peers.first { $0.publicKey == key } } }
    /// Why the file could not be read, while it cannot.
    public var readFailure: String? { lock.withLock { unreadable } }
    /// Why memory holds something the file does not, while it does.
    public var saveFailure: String? { lock.withLock { unsaved } }

    /// Insert or replace by id. Throws if the file cannot take it, and then nothing has changed.
    public func put(_ p: LinkPeer) throws {
        try lock.withLock {
            if unreadable != nil { load() }
            if let unreadable { throw LinkError.storage("The list of linked Macs cannot be read: \(unreadable)") }
            let before = peers
            if let i = peers.firstIndex(where: { $0.id == p.id }) { peers[i] = p } else { peers.append(p) }
            do { try save(); unsaved = nil } catch { peers = before; throw error }
        }
    }

    @discardableResult
    public func update(_ id: String, _ f: (inout LinkPeer) -> Void) -> LinkPeer? {
        lock.withLock {
            guard let i = peers.firstIndex(where: { $0.id == id }) else { return nil }
            let before = peers[i]
            f(&peers[i])
            // Also with nothing new while a save is owed: it is the next chance to pay it.
            if peers[i] != before || unsaved != nil { saveNoting() }
            return peers[i]
        }
    }

    public func remove(_ id: String) {
        lock.withLock {
            guard unreadable == nil else { return }
            peers.removeAll { $0.id == id }
            saveNoting()
        }
    }

    /// Try again what the file did not take. True when memory and the file now agree.
    @discardableResult
    public func retrySave() -> Bool {
        lock.withLock {
            if unsaved != nil { saveNoting() }
            return unsaved == nil
        }
    }

    private func saveNoting() {
        do { try save(); unsaved = nil } catch { unsaved = "\(error)" }
    }

    private func save() throws {
        try LinkFile.write(try JSONEncoder().encode(peers), to: url)
    }
}

/// A linked Mac's file, as a path in this Mac's result list: `omni-link://<peer id>/<path on it>`.
/// The file name stays the last component, so everything that reads a name or an extension off a
/// path keeps working; nothing may treat it as a file on this Mac.
public enum RemotePath {
    public static let scheme = "omni-link://"

    public static func isRemote(_ path: String) -> Bool { path.hasPrefix(scheme) }

    /// A path that is not a file's ("photos://...", "omni://screenshots/...") is joined with a
    /// slash: run straight onto the id, the id's end could not be found again ("abcphotos:").
    public static func make(peer: String, path: String) -> String {
        scheme + peer + (path.hasPrefix("/") ? path : "/" + path)
    }

    /// (peer id, path on that Mac), or nil for anything else.
    public static func parse(_ s: String) -> (peer: String, path: String)? {
        guard s.hasPrefix(scheme) else { return nil }
        let rest = s.dropFirst(scheme.count)
        guard let slash = rest.firstIndex(of: "/") else { return nil }
        let peer = String(rest[..<slash])
        guard !peer.isEmpty else { return nil }
        let path = rest[slash...]
        let tail = path.dropFirst()
        if let r = tail.range(of: "://"), !tail[..<r.lowerBound].isEmpty, tail[..<r.lowerBound].allSatisfy(\.isLetter) {
            return (peer, String(tail))
        }
        return (peer, String(path))
    }

    public static func peer(of s: String) -> String? { parse(s)?.peer }
}
