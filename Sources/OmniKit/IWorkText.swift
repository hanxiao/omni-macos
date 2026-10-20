import Compression
import Foundation

/// The text of a Pages or Keynote document, read straight from its archive.
///
/// `.pages` was listed as a supported extension and handed to `NSAttributedString`, which has no
/// iWork importer: "The file isn't in the correct format", every time, so every Pages document was
/// indexed with no text at all. Since iWork '13 a document is a zip of IWA files - Snappy-compressed
/// protobuf archives - and the words live in TSWP storage messages. Reading those takes a zip
/// directory, raw DEFLATE (Apple's Compression), a Snappy decoder and a protobuf walk; each is
/// small and written here rather than pulled in as a package.
///
/// Only the storages are read, so what comes back is the words of the body, text boxes, notes,
/// headers, footnotes and table-of-contents, in archive order with the body first. Numbers keeps
/// cell text in table tiles, not storages, so it is deliberately not claimed here.
enum IWorkText {

    /// TSWP.StorageArchive message types (2001 in documents, 2005 in shapes and slides).
    static let storageTypes: Set<UInt64> = [2001, 2005]
    /// One IWA file inflated past this is not a document we can index anyway; refuse it rather than
    /// let a hostile or corrupt archive claim gigabytes.
    static let maxInflated = 256 << 20

    /// nil when the file is not an iWork archive we can read (an iWork '09 XML document, a
    /// corrupt zip); "" when it is one with no words in it.
    static func extract(_ url: URL, maxBytes: Int = FileExtractor.maxTextBytes) -> String? {
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir) else { return nil }
        var archives: [(name: String, data: Data)] = []
        if isDir.boolValue {
            // The package form: Index/*.iwa loose, or (2013-era) Index.zip inside the package.
            let index = url.appendingPathComponent("Index")
            if let names = try? FileManager.default.contentsOfDirectory(atPath: index.path) {
                for n in names.sorted() where n.hasSuffix(".iwa") {
                    if let d = try? Data(contentsOf: index.appendingPathComponent(n), options: .alwaysMapped) {
                        archives.append((n, d))
                    }
                }
            } else if let z = try? Data(contentsOf: url.appendingPathComponent("Index.zip"), options: .alwaysMapped) {
                archives = Zip.entries(z) { $0.hasSuffix(".iwa") }
            }
        } else {
            guard let z = try? Data(contentsOf: url, options: .alwaysMapped) else { return nil }
            archives = Zip.entries(z) { $0.hasPrefix("Index/") && $0.hasSuffix(".iwa") }
        }
        guard !archives.isEmpty else { return nil }
        // The document's own archive first: in Pages that is where the body storage lives.
        archives.sort { a, b in
            let da = a.name.hasSuffix("Document.iwa"), db = b.name.hasSuffix("Document.iwa")
            return da != db ? da : a.name < b.name
        }
        var body: [String] = [], rest: [String] = []
        var total = 0
        for (_, data) in archives {
            guard total < maxBytes, let raw = Snappy.iwaDecode(data, cap: maxInflated) else { continue }
            for (kind, text) in storages(raw) where !text.isEmpty {
                if kind == 0 { body.append(text) } else { rest.append(text) }
                total += text.utf8.count
            }
        }
        let text = clean((body + rest).joined(separator: "\n\n"))
        guard text.utf8.count > maxBytes else { return text }
        // Back to the start of a scalar, so the cap is never exceeded and nothing is replaced.
        let u = Array(text.utf8)
        var cut = maxBytes
        while cut > 0, u[cut] & 0xC0 == 0x80 { cut -= 1 }
        return String(decoding: u[..<cut], as: UTF8.self)
    }

    /// Every TSWP storage in an inflated IWA stream, as (kind, text). The stream is a run of
    /// archives: a varint length, an ArchiveInfo naming each message's type and length, then the
    /// messages back to back.
    static func storages(_ raw: Data) -> [(kind: UInt64, text: String)] {
        var out: [(UInt64, String)] = []
        raw.withUnsafeBytes { (buf: UnsafeRawBufferPointer) in
            let p = buf.bindMemory(to: UInt8.self)
            var r = ProtoReader(p, 0, p.count)
            while !r.atEnd {
                guard let infoLen = r.varint(), infoLen <= UInt64(r.end - r.i), let info = r.sub(Int(infoLen)) else { return }
                var messages: [(type: UInt64, length: Int)] = []
                var ir = info
                while !ir.atEnd {
                    guard let (field, wire) = ir.tag() else { return }
                    if field == 2, wire == 2, let mi = ir.lengthDelimited() {
                        var mr = mi, type: UInt64 = 0, length = 0
                        while !mr.atEnd {
                            guard let (f, w) = mr.tag() else { break }
                            if f == 1, w == 0 { type = mr.varint() ?? 0 }
                            else if f == 3, w == 0 { length = Int(mr.varint() ?? 0) }
                            else if !mr.skip(w) { break }
                        }
                        messages.append((type, length))
                    } else if !ir.skip(wire) { return }
                }
                for m in messages {
                    guard let payload = r.sub(m.length) else { return }
                    guard storageTypes.contains(m.type) else { continue }
                    var sr = payload, kind: UInt64 = 3, parts: [String] = []   // TEXTBOX is the default
                    while !sr.atEnd {
                        guard let (f, w) = sr.tag() else { break }
                        if f == 1, w == 0 { kind = sr.varint() ?? kind }
                        else if f == 3, w == 2, let s = sr.lengthDelimited() { parts.append(s.string) }
                        else if !sr.skip(w) { break }
                    }
                    out.append((kind, parts.joined()))
                }
            }
        }
        return out
    }

    /// Storage text carries the layout in characters: U+2029 ends a paragraph, U+2028 breaks a line,
    /// U+FFFC holds an attachment's place, and a few controls mark sections, pages and footnotes.
    static func clean(_ s: String) -> String {
        var out = String.UnicodeScalarView()
        for u in s.unicodeScalars {
            switch u.value {
            case 0x2029, 0x2028, 0x0A, 0x0D, 0x0B, 0x0C, 0x04, 0x05: out.append("\n")
            case 0x09: out.append(u)
            case 0xFFFC: continue
            case 0 ..< 0x20: out.append(" ")
            default: out.append(u)
            }
        }
        return String(out).trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

/// A protobuf wire-format cursor over a byte range. Every read is bounds-checked and returns nil
/// rather than trapping: the input is a file from disk, not a message we wrote.
struct ProtoReader {
    let p: UnsafeBufferPointer<UInt8>
    var i: Int
    let end: Int
    init(_ p: UnsafeBufferPointer<UInt8>, _ start: Int, _ end: Int) { self.p = p; i = start; self.end = end }
    var atEnd: Bool { i >= end }

    mutating func varint() -> UInt64? {
        var v: UInt64 = 0, shift: UInt64 = 0
        while i < end {
            let b = p[i]; i += 1
            v |= UInt64(b & 0x7F) << shift
            if b & 0x80 == 0 { return v }
            shift += 7
            if shift > 63 { return nil }
        }
        return nil
    }
    mutating func tag() -> (UInt64, UInt64)? {
        guard let t = varint() else { return nil }
        return (t >> 3, t & 7)
    }
    /// The next `n` bytes as their own reader, advancing past them.
    mutating func sub(_ n: Int) -> ProtoReader? {
        guard n >= 0, n <= end - i else { return nil }
        defer { i += n }
        return ProtoReader(p, i, i + n)
    }
    mutating func lengthDelimited() -> ProtoReader? {
        guard let n = varint(), n <= UInt64(Int.max) else { return nil }
        return sub(Int(n))
    }
    mutating func skip(_ wire: UInt64) -> Bool {
        switch wire {
        case 0: return varint() != nil
        case 1: guard end - i >= 8 else { return false }; i += 8; return true
        case 2: return lengthDelimited() != nil
        case 5: guard end - i >= 4 else { return false }; i += 4; return true
        default: return false
        }
    }
    var string: String {
        String(decoding: UnsafeBufferPointer(rebasing: p[i ..< end]), as: UTF8.self)
    }
}

/// Snappy, as iWork frames it: chunks of a type byte (0 = compressed), a 3-byte little-endian
/// length, and a raw Snappy block - Google's framing format without its stream identifier or CRCs.
enum Snappy {
    static func iwaDecode(_ data: Data, cap: Int) -> Data? {
        data.withUnsafeBytes { (buf: UnsafeRawBufferPointer) -> Data? in
            let p = buf.bindMemory(to: UInt8.self)
            var out = [UInt8]()
            var i = 0
            while i + 4 <= p.count {
                let type = p[i]
                let n = Int(p[i + 1]) | Int(p[i + 2]) << 8 | Int(p[i + 3]) << 16
                i += 4
                guard n <= p.count - i else { return nil }
                if type == 0 {
                    guard decodeBlock(UnsafeBufferPointer(rebasing: p[i ..< i + n]), into: &out, cap: cap) else { return nil }
                } else if type == 1 {
                    out.append(contentsOf: p[i ..< i + n])   // stored chunk
                }
                i += n
            }
            return out.isEmpty ? nil : Data(out)
        }
    }

    /// One raw Snappy block: a varint of the uncompressed length, then literals and back-copies.
    static func decodeBlock(_ p: UnsafeBufferPointer<UInt8>, into out: inout [UInt8], cap: Int) -> Bool {
        var r = ProtoReader(p, 0, p.count)
        guard cap > out.count, let want = r.varint(), want <= UInt64(cap - out.count) else { return false }
        let base = out.count, stop = base + Int(want)
        out.reserveCapacity(stop)
        var i = r.i
        while i < p.count {
            let tag = p[i]; i += 1
            switch tag & 3 {
            case 0:
                var len = Int(tag >> 2)
                if len >= 60 {
                    let extra = len - 59
                    guard i + extra <= p.count else { return false }
                    len = 0
                    for k in 0 ..< extra { len |= Int(p[i + k]) << (8 * k) }
                    i += extra
                }
                len += 1
                guard len <= p.count - i, out.count + len <= stop else { return false }
                out.append(contentsOf: p[i ..< i + len]); i += len
            default:
                let len: Int, off: Int
                switch tag & 3 {
                case 1:
                    guard i < p.count else { return false }
                    len = Int((tag >> 2) & 7) + 4; off = Int(tag >> 5) << 8 | Int(p[i]); i += 1
                case 2:
                    guard i + 2 <= p.count else { return false }
                    len = Int(tag >> 2) + 1; off = Int(p[i]) | Int(p[i + 1]) << 8; i += 2
                default:
                    guard i + 4 <= p.count else { return false }
                    len = Int(tag >> 2) + 1
                    off = Int(p[i]) | Int(p[i + 1]) << 8 | Int(p[i + 2]) << 16 | Int(p[i + 3]) << 24; i += 4
                }
                guard off > 0, off <= out.count - base, out.count + len <= stop else { return false }
                // Byte by byte: a copy may overlap its own output (offset < length repeats a run).
                var from = out.count - off
                for _ in 0 ..< len { out.append(out[from]); from += 1 }
            }
        }
        return out.count == stop
    }
}

/// Just enough of a zip reader for a document archive: the central directory, stored and DEFLATE
/// entries, no zip64, no encryption. Entries are read through the caller's (mapped) Data, so the
/// images in a Pages file are never touched.
enum Zip {
    static func entries(_ z: Data, where wanted: (String) -> Bool) -> [(name: String, data: Data)] {
        z.withUnsafeBytes { (buf: UnsafeRawBufferPointer) -> [(String, Data)] in
            let p = buf.bindMemory(to: UInt8.self)
            func u16(_ o: Int) -> Int { Int(p[o]) | Int(p[o + 1]) << 8 }
            func u32(_ o: Int) -> Int { u16(o) | u16(o + 2) << 16 }
            // End of central directory: 22 bytes plus a comment of up to 64 KB, searched backwards.
            guard p.count >= 22 else { return [] }
            var eocd = -1
            var o = p.count - 22
            let floor = Swift.max(0, p.count - 22 - 65_535)
            while o >= floor { if u32(o) == 0x0605_4B50 { eocd = o; break }; o -= 1 }
            guard eocd >= 0 else { return [] }
            let count = u16(eocd + 10), cdOffset = u32(eocd + 16)
            var out: [(String, Data)] = []
            var c = cdOffset
            for _ in 0 ..< count {
                guard c + 46 <= p.count, u32(c) == 0x0201_4B50 else { break }
                let method = u16(c + 10), csize = u32(c + 20), usize = u32(c + 24)
                let nlen = u16(c + 28), xlen = u16(c + 30), klen = u16(c + 32), local = u32(c + 42)
                guard c + 46 + nlen <= p.count else { break }
                let name = String(decoding: UnsafeBufferPointer(rebasing: p[c + 46 ..< c + 46 + nlen]), as: UTF8.self)
                c += 46 + nlen + xlen + klen
                guard wanted(name), csize != 0xFFFF_FFFF, usize != 0xFFFF_FFFF,
                      usize <= IWorkText.maxInflated, local + 30 <= p.count, u32(local) == 0x0403_4B50 else { continue }
                let start = local + 30 + u16(local + 26) + u16(local + 28)
                guard csize > 0, start + csize <= p.count else { continue }
                let src = UnsafeBufferPointer(rebasing: p[start ..< start + csize])
                if method == 0 {
                    out.append((name, Data(src)))
                } else if method == 8, usize > 0 {
                    var dst = Data(count: usize)
                    let n = dst.withUnsafeMutableBytes { d -> Int in
                        compression_decode_buffer(d.bindMemory(to: UInt8.self).baseAddress!, usize,
                                                  src.baseAddress!, csize, nil, COMPRESSION_ZLIB)
                    }
                    if n == usize { out.append((name, dst)) }
                }
            }
            return out
        }
    }
}
