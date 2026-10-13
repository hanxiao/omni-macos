import CryptoKit
import Foundation
import Security

/// This Mac's long-term identity on the link: a P-256 key, and a self-signed certificate made from
/// it for TLS. Peers pin the PUBLIC KEY, never the certificate - the certificate is rebuilt at every
/// launch (ECDSA signatures are randomized, so its bytes differ each time) and only carries the key.
///
/// The key lives in a 0600 file in Omni's support folder rather than the Keychain: a Keychain item's
/// access list is tied to the code signature, so every differently signed build (Debug, Release,
/// the chaos copy) would raise a "wants to use confidential information" prompt. The index beside
/// the file holds the indexed content itself, so the file is no weaker than what it protects.
public struct LinkIdentity: Sendable {
    public let key: P256.Signing.PrivateKey
    /// The public key as an X9.63 point (65 bytes), the form Security hands back for a peer's
    /// certificate key - so pins compare like with like.
    public let publicKey: Data
    public let id: String
    /// DER certificate and the SecIdentity built from it, for Network.framework's TLS.
    public let certificate: Data
    nonisolated(unsafe) public let secIdentity: SecIdentity   // immutable CF object

    public init(key: P256.Signing.PrivateKey) throws {
        self.key = key
        publicKey = key.publicKey.x963Representation
        id = Self.id(of: publicKey)
        certificate = try SelfSignedCertificate.make(key: key, commonName: "Omni Link \(id.prefix(8))")
        guard let cert = SecCertificateCreateWithData(nil, certificate as CFData) else {
            throw LinkError.identity("the certificate did not parse")
        }
        let attrs: [CFString: Any] = [kSecAttrKeyType: kSecAttrKeyTypeECSECPrimeRandom,
                                      kSecAttrKeyClass: kSecAttrKeyClassPrivate,
                                      kSecAttrKeySizeInBits: 256]
        var err: Unmanaged<CFError>?
        guard let sk = SecKeyCreateWithData(key.x963Representation as CFData, attrs as CFDictionary, &err) else {
            throw LinkError.identity("the private key did not load: \(err?.takeRetainedValue().localizedDescription ?? "?")")
        }
        guard let ident = SecIdentityCreate(nil, cert, sk) else {
            throw LinkError.identity("the certificate does not match its key")
        }
        secIdentity = ident
    }

    /// A peer's ID from its X9.63 public key: the first 16 bytes of its SHA-256, in hex. Short enough
    /// to sit in a path, long enough that a collision is not a thing that happens.
    public static func id(of publicKey: Data) -> String {
        SHA256.hash(data: publicKey).prefix(16).map { String(format: "%02x", $0) }.joined()
    }

    /// The identity stored at `url`, created on first use.
    public static func loadOrCreate(at url: URL) throws -> LinkIdentity {
        let fm = FileManager.default
        if let data = try? Data(contentsOf: url),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let b64 = obj["key"] as? String, let raw = Data(base64Encoded: b64),
           let key = try? P256.Signing.PrivateKey(rawRepresentation: raw) {
            return try LinkIdentity(key: key)
        }
        let key = P256.Signing.PrivateKey()
        try fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let body = try JSONSerialization.data(withJSONObject: ["version": 1, "key": key.rawRepresentation.base64EncodedString()])
        // Written 0600 from the start: created, not chmod'ed after, so there is no moment it is readable.
        let tmp = url.appendingPathExtension("tmp")
        fm.createFile(atPath: tmp.path, contents: body, attributes: [.posixPermissions: 0o600])
        _ = try? fm.removeItem(at: url)
        try fm.moveItem(at: tmp, to: url)
        return try LinkIdentity(key: key)
    }
}

public enum LinkError: Error, CustomStringConvertible, Equatable {
    case identity(String)
    case closed
    case timeout
    case refused(String)
    case remote(String)
    case protocolViolation(String)
    case cancelled

    public var description: String {
        switch self {
        case .identity(let s): return "identity: \(s)"
        case .closed: return "the connection closed"
        case .timeout: return "no answer in time"
        case .refused(let s): return s
        case .remote(let s): return s
        case .protocolViolation(let s): return "protocol: \(s)"
        case .cancelled: return "cancelled"
        }
    }
}

/// The smallest X.509 certificate TLS will carry: v3, no extensions, self-signed with ECDSA P-256 /
/// SHA-256. Nothing verifies it as a certificate - both ends check only the key inside it against
/// their pin - so it needs to parse, not to chain.
enum SelfSignedCertificate {
    static func make(key: P256.Signing.PrivateKey, commonName: String) throws -> Data {
        let ecdsaSHA256: [UInt8] = DER.oid([0x2A, 0x86, 0x48, 0xCE, 0x3D, 0x04, 0x03, 0x02])
        let algorithm = DER.seq(ecdsaSHA256)
        let name = DER.seq(DER.set(DER.seq(DER.oid([0x55, 0x04, 0x03]), DER.utf8(commonName))))
        var serial = [UInt8](repeating: 0, count: 16)
        _ = SecRandomCopyBytes(kSecRandomDefault, serial.count, &serial)
        serial[0] &= 0x7F   // positive
        serial[0] |= 0x01   // and no leading zero byte to strip
        let validity = DER.seq(DER.utcTime("200101000000Z"), DER.generalizedTime("20991231235959Z"))
        let tbs = DER.seq(
            DER.tlv(0xA0, DER.integer([2])),          // [0] EXPLICIT version v3
            DER.integer(serial),
            algorithm,
            name,
            validity,
            name,
            [UInt8](key.publicKey.derRepresentation)  // SubjectPublicKeyInfo
        )
        let sig = try key.signature(for: Data(tbs)).derRepresentation
        return Data(DER.seq(tbs, algorithm, DER.bitString([UInt8](sig))))
    }
}

enum DER {
    static func length(_ n: Int) -> [UInt8] {
        if n < 0x80 { return [UInt8(n)] }
        var bytes: [UInt8] = []
        var v = n
        while v > 0 { bytes.insert(UInt8(v & 0xFF), at: 0); v >>= 8 }
        return [0x80 | UInt8(bytes.count)] + bytes
    }
    static func tlv(_ tag: UInt8, _ content: [UInt8]) -> [UInt8] { [tag] + length(content.count) + content }
    static func seq(_ parts: [UInt8]...) -> [UInt8] { tlv(0x30, parts.flatMap { $0 }) }
    static func set(_ parts: [UInt8]...) -> [UInt8] { tlv(0x31, parts.flatMap { $0 }) }
    static func integer(_ bytes: [UInt8]) -> [UInt8] {
        tlv(0x02, (bytes.first ?? 0) & 0x80 != 0 ? [0] + bytes : bytes)
    }
    static func oid(_ encoded: [UInt8]) -> [UInt8] { tlv(0x06, encoded) }
    static func utf8(_ s: String) -> [UInt8] { tlv(0x0C, Array(s.utf8)) }
    static func utcTime(_ s: String) -> [UInt8] { tlv(0x17, Array(s.utf8)) }
    static func generalizedTime(_ s: String) -> [UInt8] { tlv(0x18, Array(s.utf8)) }
    static func bitString(_ bytes: [UInt8]) -> [UInt8] { tlv(0x03, [0] + bytes) }
}
