import Compression
import Foundation

/// gzip framing (RFC 1952) around Apple's raw DEFLATE.
///
/// Foundation compresses but does not frame, and `Compression`'s `COMPRESSION_ZLIB` emits a bare
/// DEFLATE stream. HTTP's `Content-Encoding: gzip` wants that stream inside a 10-byte header and an
/// 8-byte trailer carrying CRC-32 and the original length, so those are added here. `deflate` as an
/// HTTP encoding would need no framing but has never been reliably interpreted by clients, which is
/// why every server sends gzip instead.
public enum Gzip {

    /// Returns nil when the encoder declines - an incompressible body - and the caller should then
    /// send it uncompressed.
    public static func encode(_ data: Data) -> Data? {
        guard !data.isEmpty else { return nil }
        let cap = data.count + 64
        var deflated = Data(count: cap)
        let n: Int = deflated.withUnsafeMutableBytes { dst -> Int in
            guard let d = dst.bindMemory(to: UInt8.self).baseAddress else { return 0 }
            return data.withUnsafeBytes { src -> Int in
                guard let s = src.bindMemory(to: UInt8.self).baseAddress else { return 0 }
                return compression_encode_buffer(d, cap, s, data.count, nil, COMPRESSION_ZLIB)
            }
        }
        guard n > 0 else { return nil }
        // magic, DEFLATE, no flags, no mtime, no extra flags, unknown OS.
        var out = Data([0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0x00, 0xff])
        out.append(deflated.prefix(n))
        var crc = crc32(data).littleEndian
        var len = UInt32(truncatingIfNeeded: data.count).littleEndian
        withUnsafeBytes(of: &crc) { out.append(contentsOf: $0) }
        withUnsafeBytes(of: &len) { out.append(contentsOf: $0) }
        return out
    }

    /// Inverse of `encode`, for tests and for any caller that has to read one back.
    public static func decode(_ gz: Data) -> Data? {
        guard gz.count > 18, gz[gz.startIndex] == 0x1f, gz[gz.startIndex + 1] == 0x8b,
              gz[gz.startIndex + 2] == 0x08 else { return nil }
        let deflated = gz.subdata(in: (gz.startIndex + 10) ..< (gz.endIndex - 8))
        // loadUnaligned, not load: a Data slice's base pointer carries no alignment guarantee, and
        // `load(as:)` traps on a misaligned one. Caught by a test on a 4096-byte random body.
        let want = Int(UInt32(littleEndian: gz.suffix(4).withUnsafeBytes {
            $0.loadUnaligned(fromByteOffset: 0, as: UInt32.self)
        }))
        guard want >= 0, want < (1 << 30) else { return nil }
        var out = Data(count: Swift.max(want, 1))
        let n: Int = out.withUnsafeMutableBytes { dst -> Int in
            guard let d = dst.bindMemory(to: UInt8.self).baseAddress else { return 0 }
            return deflated.withUnsafeBytes { src -> Int in
                guard let s = src.bindMemory(to: UInt8.self).baseAddress else { return 0 }
                return compression_decode_buffer(d, Swift.max(want, 1), s, deflated.count, nil, COMPRESSION_ZLIB)
            }
        }
        guard n == want else { return nil }
        return out.prefix(n)
    }

    /// CRC-32 (IEEE), which the gzip trailer requires and Foundation does not expose.
    static let crcTable: [UInt32] = (0 ..< 256).map { i -> UInt32 in
        var c = UInt32(i)
        for _ in 0 ..< 8 { c = (c & 1) != 0 ? 0xEDB8_8320 ^ (c >> 1) : c >> 1 }
        return c
    }

    public static func crc32(_ data: Data) -> UInt32 {
        var c: UInt32 = 0xFFFF_FFFF
        data.withUnsafeBytes { raw in
            for b in raw.bindMemory(to: UInt8.self) {
                c = crcTable[Int((c ^ UInt32(b)) & 0xFF)] ^ (c >> 8)
            }
        }
        return c ^ 0xFFFF_FFFF
    }
}
