import CryptoKit
import Foundation

/// The pairing code: a short authentication string with a commitment (docs/notes/link.md,
/// "Pairing"). Pure functions, so the construction is tested apart from any network.
public enum LinkPairing {
    public static let nonceBytes = 32

    public static func nonce() -> Data {
        var g = SystemRandomNumberGenerator()
        return Data((0 ..< nonceBytes).map { _ in UInt8.random(in: 0 ... 255, using: &g) })
    }

    /// What the client sends before it sees the server's nonce.
    public static func commitment(_ nonce: Data) -> Data {
        Data(SHA256.hash(data: Data("omni-link-commit".utf8) + nonce))
    }

    /// The six digits both ends show or check. Covers the TLS session (its exporter), both identity
    /// keys and both nonces, in a fixed order: client first.
    public static func code(exporter: Data, clientKey: Data, serverKey: Data,
                            clientNonce: Data, serverNonce: Data) -> String {
        var h = SHA256()
        for part in [Data("omni-link-sas".utf8), exporter, clientKey, serverKey, clientNonce, serverNonce] {
            h.update(data: withUnsafeBytes(of: UInt32(part.count).bigEndian) { Data($0) })
            h.update(data: part)
        }
        let d = Array(h.finalize())
        let n = UInt32(d[0]) << 24 | UInt32(d[1]) << 16 | UInt32(d[2]) << 8 | UInt32(d[3])
        return String(format: "%06u", n % 1_000_000)
    }

    /// What the user typed, reduced to digits, so "123 456" and "123-456" are the code.
    public static func normalize(_ typed: String) -> String {
        String(typed.unicodeScalars.filter { CharacterSet.decimalDigits.contains($0) && $0.isASCII }.map(Character.init))
    }

    /// Constant-time comparison: the code is not a secret, but there is no reason to leak a prefix.
    public static func matches(_ typed: String, _ code: String) -> Bool {
        let a = Array(normalize(typed).utf8), b = Array(code.utf8)
        guard a.count == b.count else { return false }
        var diff: UInt8 = 0
        for i in 0 ..< a.count { diff |= a[i] ^ b[i] }
        return diff == 0
    }

    /// "123 456", the way the code is shown.
    public static func display(_ code: String) -> String {
        guard code.count == 6 else { return code }
        return String(code.prefix(3)) + " " + String(code.suffix(3))
    }
}
