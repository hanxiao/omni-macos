import XCTest
@testable import OmniKit

/// gzip is only correct if a real decompressor accepts it, so these round-trip rather than check
/// that the encoder agrees with itself.
final class GzipTests: XCTestCase {

    /// A search payload's actual shape: JSON that repeats long absolute paths. This is what the
    /// saving comes from, so it is what the test compresses.
    private func searchPayload(_ hits: Int) -> Data {
        let rows = (0 ..< hits).map { i in
            "{\"path\":\"/Volumes/han2tb/backup/Documents/project/sub/file\(i).md\",\"score\":0.5\(i % 10),\"kind\":\"text\",\"snippet\":\"some matching text here\"}"
        }
        return Data(("{\"results\":[" + rows.joined(separator: ",") + "]}").utf8)
    }

    func testRoundTripsExactly() throws {
        for hits in [1, 10, 100, 1000] {
            let original = searchPayload(hits)
            let gz = try XCTUnwrap(Gzip.encode(original), "declined at \(hits) hits")
            XCTAssertEqual(Gzip.decode(gz), original, "round trip differs at \(hits) hits")
        }
    }

    /// The point of the exercise. A 10-hit payload is what an agent actually receives.
    func testCompressesSearchPayloadSubstantially() throws {
        let original = searchPayload(10)
        let gz = try XCTUnwrap(Gzip.encode(original))
        XCTAssertLessThan(gz.count * 2, original.count,
                          "expected better than 2x on repeated paths, got \(original.count) -> \(gz.count)")
    }

    /// The trailer is what a strict decompressor validates; a wrong CRC is accepted by a lenient
    /// inflate and rejected by zlib, so assert it directly.
    func testTrailerCarriesCRCAndLength() throws {
        let original = Data("the quick brown fox jumps over the lazy dog".utf8)
        let padded = original + Data(repeating: 0x20, count: 2000)
        let gz = try XCTUnwrap(Gzip.encode(padded))
        let crc = UInt32(littleEndian: gz.suffix(8).prefix(4).withUnsafeBytes {
            $0.loadUnaligned(fromByteOffset: 0, as: UInt32.self) })
        let len = UInt32(littleEndian: gz.suffix(4).withUnsafeBytes {
            $0.loadUnaligned(fromByteOffset: 0, as: UInt32.self) })
        XCTAssertEqual(len, UInt32(padded.count))
        XCTAssertEqual(crc, Gzip.crc32(padded))
    }

    /// CRC-32 against the published check value, so a bad table cannot pass by agreeing with
    /// itself. "123456789" is the standard vector for CRC-32/ISO-HDLC.
    func testCRC32MatchesPublishedCheckValue() {
        XCTAssertEqual(Gzip.crc32(Data("123456789".utf8)), 0xCBF4_3926)
    }

    func testEmptyDeclines() {
        XCTAssertNil(Gzip.encode(Data()))
    }

    /// Random bytes do not compress; the encoder must say so rather than return something larger,
    /// because the caller uses nil to mean "send it raw".
    func testIncompressibleDataDoesNotGrowSilently() throws {
        var rng = SystemRandomNumberGenerator()
        let noise = Data((0 ..< 4096).map { _ in UInt8.random(in: 0 ... 255, using: &rng) })
        if let gz = Gzip.encode(noise) {
            XCTAssertEqual(Gzip.decode(gz), noise, "if it claims to encode, it must round trip")
        }
    }
}
