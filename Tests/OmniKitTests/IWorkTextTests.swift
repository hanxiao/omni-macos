import Compression
import XCTest
@testable import OmniKit

/// The iWork reader, against archives built here byte by byte: Snappy literals and overlapping
/// back-copies, the ArchiveInfo walk, both zip methods, the package form, and input that is not an
/// archive at all. The reader was checked against real Pages and Keynote files with
/// `omni-verify extract`; these pin the formats so a refactor cannot quietly break one of them.
final class IWorkTextTests: XCTestCase {

    // MARK: - Builders

    private func varint(_ v: UInt64) -> [UInt8] {
        var v = v, out: [UInt8] = []
        repeat { var b = UInt8(v & 0x7F); v >>= 7; if v != 0 { b |= 0x80 }; out.append(b) } while v != 0
        return out
    }
    private func field(_ n: UInt64, varint v: UInt64) -> [UInt8] { varint(n << 3) + varint(v) }
    private func field(_ n: UInt64, bytes b: [UInt8]) -> [UInt8] { varint(n << 3 | 2) + varint(UInt64(b.count)) + b }

    /// A TSWP.StorageArchive: kind, then the text split across repeated field 3, with an unrelated
    /// field in between that the reader has to skip.
    private func storage(kind: UInt64, _ parts: [String]) -> [UInt8] {
        var m = field(1, varint: kind) + field(2, bytes: [0x08, 0x01])
        for p in parts { m += field(3, bytes: Array(p.utf8)) }
        return m + field(10, varint: 1)
    }

    /// One archive: ArchiveInfo listing the messages, then the messages.
    private func archive(_ messages: [(type: UInt64, payload: [UInt8])]) -> [UInt8] {
        var info = field(1, varint: 42)
        for m in messages {
            info += field(2, bytes: field(1, varint: m.type) + field(2, bytes: [1, 0, 5]) + field(3, varint: UInt64(m.payload.count)))
        }
        return varint(UInt64(info.count)) + info + messages.flatMap(\.payload)
    }

    /// A Snappy block of literals only, in chunks of at most 60 bytes, then IWA chunk framing.
    private func iwa(_ raw: [UInt8]) -> [UInt8] {
        var block = varint(UInt64(raw.count))
        var i = 0
        while i < raw.count {
            let n = min(60, raw.count - i)
            block.append(UInt8((n - 1) << 2))
            block += raw[i ..< i + n]; i += n
        }
        let len = block.count
        return [0, UInt8(len & 0xFF), UInt8(len >> 8 & 0xFF), UInt8(len >> 16 & 0xFF)] + block
    }

    private func le16(_ v: Int) -> [UInt8] { [UInt8(v & 0xFF), UInt8(v >> 8 & 0xFF)] }
    private func le32(_ v: Int) -> [UInt8] { le16(v & 0xFFFF) + le16(v >> 16 & 0xFFFF) }

    /// A zip with the given entries; `deflate` compresses each with raw DEFLATE (method 8).
    private func zip(_ files: [(String, [UInt8])], deflate: Bool) -> Data {
        var body: [UInt8] = [], central: [UInt8] = []
        for (name, data) in files {
            var payload = data, method = 0
            if deflate {
                var dst = [UInt8](repeating: 0, count: data.count + 64)
                let n = compression_encode_buffer(&dst, dst.count, data, data.count, nil, COMPRESSION_ZLIB)
                if n > 0 { payload = Array(dst.prefix(n)); method = 8 }
            }
            let nameBytes = Array(name.utf8), offset = body.count
            body += le32(0x0403_4B50) + le16(20) + le16(0) + le16(method) + le32(0) + le32(0)
                + le32(payload.count) + le32(data.count) + le16(nameBytes.count) + le16(0) + nameBytes + payload
            central += le32(0x0201_4B50) + le16(20) + le16(20) + le16(0) + le16(method) + le32(0) + le32(0)
                + le32(payload.count) + le32(data.count) + le16(nameBytes.count) + le16(0) + le16(0)
                + le16(0) + le16(0) + le32(0) + le32(offset) + nameBytes
        }
        let eocd = le32(0x0605_4B50) + le16(0) + le16(0) + le16(files.count) + le16(files.count)
            + le32(central.count) + le32(body.count) + le16(0)
        return Data(body + central + eocd)
    }

    private func tempFile(_ name: String, _ data: Data) throws -> URL {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("iwork-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent(name)
        try data.write(to: url)
        return url
    }

    // MARK: - Snappy

    func testSnappyBackCopyMayOverlapItsOwnOutput() {
        // "ab" then a copy of offset 2, length 6: "abababab". Tag type 1 carries length-4 in bits 2-4.
        let block: [UInt8] = [8, 1 << 2, 0x61, 0x62, UInt8((6 - 4) << 2 | 1), 2]
        var out: [UInt8] = []
        XCTAssertTrue(block.withUnsafeBufferPointer { Snappy.decodeBlock($0, into: &out, cap: 1 << 20) })
        XCTAssertEqual(String(decoding: out, as: UTF8.self), "abababab")
    }

    func testSnappyRefusesACopyBeforeTheStartAndALengthPastTheCap() {
        var out: [UInt8] = []
        let backTooFar: [UInt8] = [6, 1 << 2, 0x61, 0x62, UInt8((4 - 4) << 2 | 1), 9]
        XCTAssertFalse(backTooFar.withUnsafeBufferPointer { Snappy.decodeBlock($0, into: &out, cap: 1 << 20) })
        out = []
        let huge: [UInt8] = varint(1 << 40) + [0]
        XCTAssertFalse(huge.withUnsafeBufferPointer { Snappy.decodeBlock($0, into: &out, cap: 1 << 20) })
    }

    // MARK: - The archive walk

    func testStoragesComeBackWithTheirKindAndOtherMessagesAreSkipped() throws {
        let raw = archive([(2001, storage(kind: 0, ["Quarterly ", "plan\u{2029}Second paragraph"])),
                           (9999, [0x08, 0x05]),
                           (2005, storage(kind: 3, ["A text box"]))])
        let decoded = try XCTUnwrap(Snappy.iwaDecode(Data(iwa(raw)), cap: 1 << 20))
        XCTAssertEqual(Array(decoded), raw)
        let s = IWorkText.storages(decoded)
        XCTAssertEqual(s.map(\.kind), [0, 3])
        XCTAssertEqual(s.map(\.text), ["Quarterly plan\u{2029}Second paragraph", "A text box"])
    }

    func testTruncatedArchiveYieldsWhatWasCompleteAndNeverTraps() {
        let raw = archive([(2001, storage(kind: 0, ["whole"])), (2001, storage(kind: 0, ["cut off here"]))])
        for cut in stride(from: 0, to: raw.count, by: 3) {
            _ = IWorkText.storages(Data(raw.prefix(cut)))
        }
        XCTAssertEqual(IWorkText.storages(Data(raw)).map(\.text), ["whole", "cut off here"])
    }

    // MARK: - Whole documents

    private func document() -> [(String, [UInt8])] {
        [("Index/Document.iwa", iwa(archive([(2001, storage(kind: 3, ["Box"])),
                                             (2001, storage(kind: 0, ["Body\u{2029}text\u{FFFC}"]))]))),
         ("Index/Metadata.iwa", iwa(archive([(2005, storage(kind: 4, ["Speaker note"]))]))),
         ("preview.jpg", [0xFF, 0xD8, 0xFF]),
         ("Data/image.png", [UInt8](repeating: 7, count: 1000))]
    }

    func testStoredZipPagesFile() throws {
        let url = try tempFile("Plan.pages", zip(document(), deflate: false))
        XCTAssertEqual(IWorkText.extract(url), "Body\ntext\n\nBox\n\nSpeaker note")
        guard case .text(let s) = try FileExtractor.extract(url) else { return XCTFail("not text") }
        XCTAssertTrue(s.hasPrefix("Body"))
    }

    func testDeflatedZipKeynoteFile() throws {
        let url = try tempFile("Talk.key", zip(document(), deflate: true))
        XCTAssertEqual(IWorkText.extract(url), "Body\ntext\n\nBox\n\nSpeaker note")
    }

    func testPackageFormWithLooseArchives() throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("iwork-\(UUID().uuidString)/Old.pages")
        try FileManager.default.createDirectory(at: dir.appendingPathComponent("Index"), withIntermediateDirectories: true)
        for (name, data) in document() where name.hasPrefix("Index/") {
            try Data(data).write(to: dir.appendingPathComponent(name))
        }
        XCTAssertEqual(IWorkText.extract(dir), "Body\ntext\n\nBox\n\nSpeaker note")
    }

    func testNotAnArchiveIsNilNotACrash() throws {
        XCTAssertNil(IWorkText.extract(try tempFile("x.pages", Data("<?xml version=\"1.0\"?><sl:document/>".utf8))))
        XCTAssertNil(IWorkText.extract(try tempFile("y.pages", Data())))
        var junk = [UInt8](repeating: 0, count: 4096)
        for i in junk.indices { junk[i] = UInt8(truncatingIfNeeded: i &* 2_654_435_761 >> 7) }
        _ = IWorkText.extract(try tempFile("z.key", Data(junk)))
        guard case .empty = try FileExtractor.extract(try tempFile("w.pages", Data("nope".utf8))) else {
            return XCTFail("an unreadable Pages file should index as empty")
        }
    }

    func testLongTextIsCappedOnAScalarBoundary() throws {
        let body = String(repeating: "数据", count: 2000)
        let url = try tempFile("Long.pages", zip([("Index/Document.iwa", iwa(archive([(2001, storage(kind: 0, [body]))])))], deflate: false))
        let s = try XCTUnwrap(IWorkText.extract(url, maxBytes: 1001))
        XCTAssertLessThanOrEqual(s.utf8.count, 1001)
        XCTAssertGreaterThan(s.utf8.count, 990)
        XCTAssertTrue(s.allSatisfy { $0 == "数" || $0 == "据" }, "a cut scalar was repaired instead of dropped")
    }
}
