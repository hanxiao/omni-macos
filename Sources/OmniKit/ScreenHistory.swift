import Foundation
import CoreVideo
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

/// THE SCREEN HISTORY (docs/screenshots.md): every shot of the desktop, as video, and the few of
/// them worth finding again as picture files the index reads like any other screenshot.
///
/// Two folders. `history` holds the segments: one file per display per hour at most, HEVC frames
/// appended one record at a time (ScreenCodec), each record carrying its time, the app in front and
/// its window's title. Nothing there is crawled. `stills` is a crawl root like the clipboard's: a
/// shot the gate admits (ScreenGate) is also written there as a HEIC marked as a screen capture, so
/// the unchanged image pipeline embeds it and reads its text.
///
/// A segment is only ever appended to by the process that created it, and a new one starts at every
/// launch, so a record cut short by a kill is the last thing in its file: a reader stops at it and
/// nothing needs repairing. The frame table is read from the segments themselves; there is no second
/// file to fall out of step with them.
public final class ScreenHistory: @unchecked Sendable {
    public struct Shot: Sendable, Hashable, Identifiable {
        public let time: Double
        public let display: UInt32
        /// Also in the index, as a file under `stills`.
        public var indexed: Bool
        public let app: String
        public let title: String
        public let segment: String
        public let index: Int
        public var id: String { "\(segment)#\(index)" }
    }

    /// One stretch of work: shots with no gap longer than `ScreenHistory.sessionGap` between them.
    public struct Session: Sendable {
        public struct App: Sendable { public let name: String; public let shots: Int; public let titles: [String] }
        public let start: Double
        public let end: Double
        public let shots: Int
        /// Most seen first.
        public let apps: [App]
        /// The shot nearest the moment asked about.
        public let nearest: Shot
    }

    public struct Recorded: Sendable {
        public let shot: Shot
        /// The file written for the index, when the gate admitted the shot.
        public let still: URL?
        /// How much of the shot no recently indexed one shows, 0...1.
        public let novelty: Double
        public let bytes: Int
    }

    public static let segmentExtension = "omniscreen"
    public static let maxSegmentFrames = 720
    public static let maxSegmentSeconds = 3600.0
    public static let sessionGap = 300.0
    /// The furthest either side of a moment its session is followed.
    public static let sessionReach = 16 * 3600.0
    /// The index reads a picture no larger than this (IndexSettings.maxImageDimension), so the file
    /// written for it is no larger either; the full frame is in the history.
    public static let stillDimension = 1568
    public static let stillQuality = 0.8

    public let history: URL
    public let stills: URL

    private struct Frame {
        var shot: Shot
        let isKey: Bool
        let sets: [Data]
        let record: UInt64          // where the record starts
        let payload: UInt64
        let length: Int
        let key: Int                // the key frame this one decodes from
    }

    private struct Ref { let name: String; let start: Double; let display: UInt32 }

    private final class Writer {
        let name: String, handle: FileHandle, encoder: ScreenFrameEncoder
        let started: Double, day: Int
        var size: UInt64
        init(name: String, handle: FileHandle, encoder: ScreenFrameEncoder, started: Double, day: Int, size: UInt64) {
            self.name = name; self.handle = handle; self.encoder = encoder; self.started = started; self.day = day; self.size = size
        }
    }

    private let lock = NSLock()            // tables, listing
    private var tables: [String: [Frame]] = [:]
    private var listing: [Ref]?
    private let writing = NSLock()         // record, stop, prune, clear: one at a time
    private var writers: [UInt32: Writer] = [:]
    private var lastStored: [UInt32: ScreenSignature] = [:]
    private var gates: [UInt32: ScreenGate] = [:]
    private let decoding = NSLock()
    private var decoder = ScreenFrameDecoder()
    private var decoded: (segment: String, index: Int, pixels: CVPixelBuffer)?
    private var reading: (segment: String, handle: FileHandle)?

    private let segmentFrames: Int

    public init(history: URL, stills: URL, segmentFrames: Int = ScreenHistory.maxSegmentFrames) {
        self.history = history; self.stills = stills; self.segmentFrames = segmentFrames
    }

    // MARK: Recording

    /// Take one shot into the history. nil when the screen shows what the last stored shot of this
    /// display shows (nothing is written), or when it could not be encoded.
    ///
    /// `settled`: the user has just stopped. What the screen was left showing is indexed if any of it
    /// is new, even when that is the shot already stored.
    public func record(_ pixels: CVPixelBuffer, time: Double, display: UInt32, app: String, title: String, settled: Bool) -> Recorded? {
        writing.lock(); defer { writing.unlock() }
        guard let signature = ScreenSignature(pixels) else { return nil }
        if gates[display] == nil { gates[display] = gate(for: display) }
        if let last = lastStored[display], signature.changedShare(from: last) == 0 {
            guard settled, let writer = writers[display] else { return nil }
            let stored: Frame? = lock.withLock { tables[writer.name]?.last }
            guard let stored, !stored.shot.indexed else { return nil }
            let novelty = gates[display, default: ScreenGate()].novelty(signature)
            guard gates[display, default: ScreenGate()].admit(signature, settled: true) else { return nil }
            guard let still = writeStill(pixels, time: stored.shot.time, app: stored.shot.app, title: stored.shot.title) else { return nil }
            // The record is already on disk; only its flag changes.
            try? writer.handle.seek(toOffset: stored.record + 16)
            try? writer.handle.write(contentsOf: Data([(stored.isKey ? 1 : 0) | 2]))
            _ = try? writer.handle.seekToEnd()
            var shot = stored.shot
            shot.indexed = true
            lock.withLock { tables[writer.name]?[shot.index].shot = shot }
            return Recorded(shot: shot, still: still, novelty: novelty, bytes: 0)
        }
        let novelty = gates[display, default: ScreenGate()].novelty(signature)
        let indexed = gates[display, default: ScreenGate()].admit(signature, settled: settled)
        guard let writer = writer(for: display, pixels: pixels, time: time),
              let encoded = writer.encoder.encode(pixels) else { return nil }
        let cleanApp = Self.clean(app, limit: 40), cleanTitle = Self.clean(title, limit: 80)
        let still = indexed ? writeStill(pixels, time: time, app: cleanApp, title: cleanTitle) : nil
        let flagged = indexed && still != nil
        let (record, payloadOffset) = Self.record(time: time, isKey: encoded.isKey, indexed: flagged, app: cleanApp, title: cleanTitle,
                                                  sets: encoded.parameterSets, payload: encoded.data)
        do { try writer.handle.write(contentsOf: record) } catch { return nil }
        let at = writer.size
        writer.size += UInt64(record.count)
        lastStored[display] = signature
        let shot: Shot = lock.withLock {
            var table = tables[writer.name] ?? []
            let key = encoded.isKey ? table.count : (table.last?.key ?? 0)
            let shot = Shot(time: time, display: display, indexed: flagged, app: cleanApp, title: cleanTitle, segment: writer.name, index: table.count)
            table.append(Frame(shot: shot, isKey: encoded.isKey, sets: encoded.parameterSets, record: at,
                               payload: at + UInt64(payloadOffset), length: encoded.data.count, key: key))
            tables[writer.name] = table
            return shot
        }
        return Recorded(shot: shot, still: still, novelty: novelty, bytes: record.count)
    }

    /// Close the open segments. The next shot starts new ones.
    public func stop() {
        writing.lock(); defer { writing.unlock() }
        closeWriters()
    }

    private func closeWriters() {
        for w in writers.values { try? w.handle.close() }
        writers.removeAll(); lastStored.removeAll(); gates.removeAll()
    }

    /// How far back a starting recorder looks for what the index holds of a display.
    private static let gateSegments = 24

    /// The gate a display starts with: the shots of it the index already holds. A recorder that
    /// starts with an empty one (a launch, Screenshots turned off and on) indexes the screen it finds
    /// whatever is on it: on the first real session every start added one more picture of the same
    /// terminal (docs/screenshots.md, "A real session"). Read back from the segments like everything
    /// else here, so there is no second file to keep true: the last indexed shots whose files are
    /// still in the index's folder, decoded. Called under `writing`.
    private func gate(for display: UInt32) -> ScreenGate {
        var found: [Shot] = []
        walk: for ref in refs().reversed().lazy.filter({ $0.display == display }).prefix(Self.gateSegments) {
            for f in frames(of: ref.name).reversed() where f.shot.indexed && still(of: f.shot) != nil {
                found.append(f.shot)
                if found.count == ScreenGate.memory { break walk }
            }
        }
        var gate = ScreenGate()
        for shot in found.reversed() {
            if let pixels = pixels(shot), let signature = ScreenSignature(pixels) { gate.remember(signature) }
        }
        return gate
    }

    private func writer(for display: UInt32, pixels: CVPixelBuffer, time: Double) -> Writer? {
        let width = CVPixelBufferGetWidth(pixels), height = CVPixelBufferGetHeight(pixels)
        let day = Self.day(time)
        if let w = writers[display] {
            let count = lock.withLock { tables[w.name]?.count ?? 0 }
            if w.encoder.width == width, w.encoder.height == height, count < segmentFrames,
               time - w.started < Self.maxSegmentSeconds, time >= w.started, w.day == day { return w }
            try? w.handle.close()
            writers[display] = nil
        }
        guard let encoder = ScreenFrameEncoder(width: width, height: height) else { return nil }
        try? FileManager.default.createDirectory(at: history, withIntermediateDirectories: true)
        var start = Int(time)
        var name = "\(start)-\(display).\(Self.segmentExtension)"
        while FileManager.default.fileExists(atPath: history.appendingPathComponent(name).path) {
            start += 1; name = "\(start)-\(display).\(Self.segmentExtension)"
        }
        var header = Data("OMNISCR1".utf8)
        for v in [UInt32(width), UInt32(height), display, 0] { header.append(le: v) }
        let url = history.appendingPathComponent(name)
        guard FileManager.default.createFile(atPath: url.path, contents: header),
              let handle = try? FileHandle(forUpdating: url) else { return nil }
        _ = try? handle.seekToEnd()
        let w = Writer(name: name, handle: handle, encoder: encoder, started: Double(start), day: day, size: UInt64(header.count))
        writers[display] = w
        lock.withLock {
            tables[name] = []
            listing?.append(Ref(name: name, start: Double(start), display: display))
        }
        return w
    }

    // MARK: The segment file

    private static let headerLength = 24
    private static let marker: UInt32 = 0x544F_4853      // "SHOT"

    /// One record, and where its payload starts inside it.
    private static func record(time: Double, isKey: Bool, indexed: Bool, app: String, title: String, sets: [Data], payload: Data) -> (Data, Int) {
        let appBytes = Data(app.utf8), titleBytes = Data(title.utf8)
        var body = Data()
        body.append(le: time.bitPattern)
        body.append(UInt8((isKey ? 1 : 0) | (indexed ? 2 : 0)))
        body.append(UInt8(sets.count))
        body.append(le: UInt16(appBytes.count)); body.append(le: UInt16(titleBytes.count))
        body.append(le: UInt32(payload.count))
        body.append(appBytes); body.append(titleBytes)
        for s in sets { body.append(le: UInt16(s.count)); body.append(s) }
        let payloadOffset = 8 + body.count
        body.append(payload)
        var out = Data(capacity: body.count + 12)
        out.append(le: marker); out.append(le: UInt32(body.count))
        out.append(body)
        out.append(le: UInt32(body.count))
        return (out, payloadOffset)
    }

    /// Every whole record of a segment. A record cut short ends the walk.
    private static func scan(_ url: URL, name: String) -> [Frame] {
        guard let data = try? Data(contentsOf: url, options: .alwaysMapped), data.count >= headerLength,
              data.prefix(8) == Data("OMNISCR1".utf8) else { return [] }
        let display: UInt32 = data.le(at: 16)
        var frames: [Frame] = []
        var at = headerLength, key = 0
        while at + 8 <= data.count {
            let mark: UInt32 = data.le(at: at), length = Int(data.le(at: at + 4) as UInt32)
            guard mark == marker, length >= 18, at + 8 + length + 4 <= data.count,
                  Int(data.le(at: at + 8 + length) as UInt32) == length else { break }
            let b = at + 8
            let time = Double(bitPattern: data.le(at: b) as UInt64)
            let flags = data[data.startIndex + b + 8], setCount = Int(data[data.startIndex + b + 9])
            let appLength = Int(data.le(at: b + 10) as UInt16), titleLength = Int(data.le(at: b + 12) as UInt16)
            let payloadLength = Int(data.le(at: b + 14) as UInt32)
            var p = b + 18
            guard p + appLength + titleLength <= b + length else { break }
            let app = String(decoding: data.sub(p, appLength), as: UTF8.self); p += appLength
            let title = String(decoding: data.sub(p, titleLength), as: UTF8.self); p += titleLength
            var sets: [Data] = []
            var whole = true
            for _ in 0 ..< setCount {
                guard p + 2 <= b + length else { whole = false; break }
                let n = Int(data.le(at: p) as UInt16); p += 2
                guard p + n <= b + length else { whole = false; break }
                sets.append(Data(data.sub(p, n))); p += n
            }
            guard whole, p + payloadLength == b + length else { break }
            let isKey = flags & 1 != 0
            if isKey { key = frames.count }
            let shot = Shot(time: time, display: display, indexed: flags & 2 != 0, app: app, title: title, segment: name, index: frames.count)
            frames.append(Frame(shot: shot, isKey: isKey, sets: sets, record: UInt64(at), payload: UInt64(p), length: payloadLength, key: key))
            at = b + length + 4
        }
        return frames
    }

    // MARK: Reading

    private func refs() -> [Ref] {
        lock.withLock {
            if let listing { return listing }
            let names = (try? FileManager.default.contentsOfDirectory(atPath: history.path)) ?? []
            let found: [Ref] = names.compactMap { name in
                guard name.hasSuffix("." + Self.segmentExtension) else { return nil }
                let parts = name.dropLast(Self.segmentExtension.count + 1).split(separator: "-")
                guard parts.count == 2, let start = Double(parts[0]), let display = UInt32(parts[1]) else { return nil }
                return Ref(name: name, start: start, display: display)
            }.sorted { $0.start < $1.start }
            listing = found
            return found
        }
    }

    private func frames(of name: String) -> [Frame] {
        if let cached = lock.withLock({ tables[name] }) { return cached }
        let scanned = Self.scan(history.appendingPathComponent(name), name: name)
        return lock.withLock {
            if let raced = tables[name] { return raced }
            tables[name] = scanned
            return scanned
        }
    }

    /// The shots taken in a span of time, oldest first.
    public func shots(from: Double, to: Double) -> [Shot] {
        var out: [Shot] = []
        for ref in refs() where ref.start <= to && ref.start >= from - Self.maxSegmentSeconds - 1 {
            for f in frames(of: ref.name) where f.shot.time >= from && f.shot.time <= to { out.append(f.shot) }
        }
        return out.sorted { $0.time < $1.time }
    }

    /// The local days that have shots, oldest first, as the start of each day.
    public func days() -> [Date] {
        let calendar = Calendar.current
        var seen = Set<Date>()
        for ref in refs() { seen.insert(calendar.startOfDay(for: Date(timeIntervalSince1970: ref.start))) }
        return seen.sorted()
    }

    public var isEmpty: Bool { refs().isEmpty }

    /// A shot's pixels (BGRA, IOSurface-backed). Stepping forward decodes one frame; anywhere else
    /// decodes from the key frame before it, at most `ScreenCodec.keyInterval` frames.
    public func pixels(_ shot: Shot) -> CVPixelBuffer? {
        let table = frames(of: shot.segment)
        guard shot.index < table.count else { return nil }
        decoding.lock(); defer { decoding.unlock() }
        if let decoded, decoded.segment == shot.segment, decoded.index == shot.index { return decoded.pixels }
        var first = table[shot.index].key
        if let decoded, decoded.segment == shot.segment, decoded.index >= first, decoded.index < shot.index { first = decoded.index + 1 }
        if reading?.segment != shot.segment {
            try? reading?.handle.close()
            reading = nil
            guard let handle = try? FileHandle(forReadingFrom: history.appendingPathComponent(shot.segment)) else { return nil }
            reading = (shot.segment, handle)
        }
        guard let handle = reading?.handle else { return nil }
        var out: CVPixelBuffer?
        for i in first ... shot.index {
            let f = table[i]
            guard (try? handle.seek(toOffset: f.payload)) != nil, let data = try? handle.read(upToCount: f.length), data.count == f.length,
                  let image = decoder.decode(data, parameterSets: f.sets) else {
                decoded = nil
                decoder = ScreenFrameDecoder()
                return nil
            }
            out = image
        }
        guard let out else { return nil }
        decoded = (shot.segment, shot.index, out)
        return out
    }

    /// The stretch of work around a moment: nil when no shot is within `sessionGap` of it.
    ///
    /// Read two hours either side first and wider only while the stretch runs to the edge of what
    /// was read: a served search asks this for every screenshot it returns, and a day of segments
    /// read for each (each one scanned the first time) is work the answer rarely needs.
    public func session(around time: Double) -> Session? {
        var reach = 2 * 3600.0
        while true {
            let all = shots(from: time - reach, to: time + reach)
            guard let nearest = all.indices.min(by: { abs(all[$0].time - time) < abs(all[$1].time - time) }),
                  abs(all[nearest].time - time) <= Self.sessionGap else { return nil }
            var lo = nearest, hi = nearest
            while lo > 0, all[lo].time - all[lo - 1].time <= Self.sessionGap { lo -= 1 }
            while hi + 1 < all.count, all[hi + 1].time - all[hi].time <= Self.sessionGap { hi += 1 }
            // A stretch that ends within a gap of the edge may go on past it.
            let open = all[lo].time - (time - reach) <= Self.sessionGap || (time + reach) - all[hi].time <= Self.sessionGap
            if open, reach < Self.sessionReach { reach = min(reach * 2, Self.sessionReach); continue }
            var counts: [String: Int] = [:], titles: [String: [String]] = [:]
            for s in all[lo ... hi] where !s.app.isEmpty {
                counts[s.app, default: 0] += 1
                if !s.title.isEmpty, !(titles[s.app] ?? []).contains(s.title) { titles[s.app, default: []].append(s.title) }
            }
            let apps = counts.sorted { $0.value != $1.value ? $0.value > $1.value : $0.key < $1.key }
                .map { Session.App(name: $0.key, shots: $0.value, titles: titles[$0.key] ?? []) }
            return Session(start: all[lo].time, end: all[hi].time, shots: hi - lo + 1, apps: apps, nearest: all[nearest])
        }
    }

    /// The stretch around each of several moments, each stretch walked once however many of the
    /// moments fall in it: the thirty screenshots of one answer are mostly of one afternoon, and a
    /// walk a result measured 22.6 ms for thirty against 2.5 ms this way (docs/screenshots.md).
    public func sessions(around times: [Double]) -> [Session?] {
        var known: [Session] = []
        return times.map { time in
            // Inside a stretch's span the nearest shot is within half a gap, so it is that stretch.
            if let s = known.first(where: { $0.start <= time && time <= $0.end }) {
                let near = shots(from: time - Self.sessionGap, to: time + Self.sessionGap)
                guard let nearest = near.min(by: { abs($0.time - time) < abs($1.time - time) }) else { return nil }
                return Session(start: s.start, end: s.end, shots: s.shots, apps: s.apps, nearest: nearest)
            }
            let found = session(around: time)
            if let found { known.append(found) }
            return found
        }
    }

    // MARK: The files the index reads

    /// The folder and name (no extension) of a shot's file: "2026-11-10" and
    /// "2026-11-10 14.05.32 Xcode - Sidebar.swift".
    private static func stillName(time: Double, app: String, title: String) -> (folder: String, base: String) {
        let date = Date(timeIntervalSince1970: time)
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyy-MM-dd"
        let folder = f.string(from: date)
        f.dateFormat = "yyyy-MM-dd HH.mm.ss"
        var base = f.string(from: date)
        let subject = [app, title].filter { !$0.isEmpty }.joined(separator: " - ")
        if !subject.isEmpty { base += " " + subject }
        return (folder, base)
    }

    /// The file the index holds for a shot, while it is still there.
    public func still(of shot: Shot) -> URL? {
        guard shot.indexed else { return nil }
        let name = Self.stillName(time: shot.time, app: shot.app, title: shot.title)
        let url = stills.appendingPathComponent(name.folder, isDirectory: true).appendingPathComponent(name.base + ".heic")
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    /// When a file under `stills` was taken, read from its name. nil for any other file.
    public static func shotTime(ofStill url: URL) -> Double? {
        let name = url.lastPathComponent
        guard name.count >= 19 else { return nil }
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyy-MM-dd HH.mm.ss"
        return f.date(from: String(name.prefix(19)))?.timeIntervalSince1970
    }

    /// The picture file of a shot, dated at the shot.
    private func writeStill(_ pixels: CVPixelBuffer, time: Double, app: String, title: String) -> URL? {
        guard let image = Self.fitted(pixels, maxDimension: Self.stillDimension) else { return nil }
        let date = Date(timeIntervalSince1970: time)
        let name = Self.stillName(time: time, app: app, title: title)
        let folder = stills.appendingPathComponent(name.folder, isDirectory: true)
        let base = name.base
        do { try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true) } catch { return nil }
        var url = folder.appendingPathComponent(base + ".heic")
        var n = 2
        while FileManager.default.fileExists(atPath: url.path) { url = folder.appendingPathComponent("\(base) \(n).heic"); n += 1 }
        // Hidden until it is whole, marked and dated: the watcher never sees a half-written shot.
        let temp = folder.appendingPathComponent("." + url.lastPathComponent + ".tmp")
        guard let dest = CGImageDestinationCreateWithURL(temp as CFURL, UTType.heic.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, image, ScreenCodec.stillOptions(quality: Self.stillQuality))
        guard CGImageDestinationFinalize(dest) else { try? FileManager.default.removeItem(at: temp); return nil }
        if let flag = try? PropertyListSerialization.data(fromPropertyList: true, format: .binary, options: 0) {
            _ = flag.withUnsafeBytes { setxattr(temp.path, Self.screenCaptureAttribute, $0.baseAddress, flag.count, 0, 0) }
        }
        try? FileManager.default.setAttributes([.creationDate: date, .modificationDate: date], ofItemAtPath: temp.path)
        do { try FileManager.default.moveItem(at: temp, to: url) } catch { try? FileManager.default.removeItem(at: temp); return nil }
        return url
    }

    /// What marks a file as a screen capture to the indexer (Indexer.isScreenshot) and to Spotlight.
    public static let screenCaptureAttribute = "com.apple.metadata:kMDItemIsScreenCapture"

    private static func fitted(_ pixels: CVPixelBuffer, maxDimension: Int) -> CGImage? {
        guard let image = ScreenCodec.cgImage(pixels) else { return nil }
        let longest = max(image.width, image.height)
        guard longest > maxDimension else { return image }
        let k = Double(maxDimension) / Double(longest)
        let w = max(1, Int((Double(image.width) * k).rounded())), h = max(1, Int((Double(image.height) * k).rounded()))
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
              let cg = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0, space: space,
                                 bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue) else { return nil }
        cg.interpolationQuality = .high
        cg.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        return cg.makeImage()
    }

    /// A name's worth of an app name or window title: one line, no path separators, bounded.
    static func clean(_ s: String, limit: Int) -> String {
        var out = ""
        var space = false
        for u in s.unicodeScalars {
            if u == "/" || u == ":" || CharacterSet.controlCharacters.contains(u) || CharacterSet.newlines.contains(u) || u == " " {
                space = !out.isEmpty
            } else {
                if space { out.append(" "); space = false }
                out.unicodeScalars.append(u)
            }
        }
        if out.hasPrefix(".") { out.removeFirst() }
        return out.count > limit ? String(out.prefix(limit)).trimmingCharacters(in: .whitespaces) : out
    }

    private static func day(_ time: Double) -> Int {
        let c = Calendar.current.dateComponents([.year, .month, .day], from: Date(timeIntervalSince1970: time))
        return (c.year ?? 0) * 10_000 + (c.month ?? 0) * 100 + (c.day ?? 0)
    }

    // MARK: Retention

    /// Delete what is older than `cutoff`: whole segments, and the index's files one by one. Returns
    /// how many files went.
    @discardableResult
    public func prune(olderThan cutoff: Double) -> Int {
        writing.lock(); defer { writing.unlock() }
        let fm = FileManager.default
        var removed = 0
        let live = Set(writers.values.map(\.name))
        for ref in refs() where ref.start + Self.maxSegmentSeconds < cutoff && !live.contains(ref.name) {
            if (try? fm.removeItem(at: history.appendingPathComponent(ref.name))) != nil { removed += 1 }
            lock.withLock { tables[ref.name] = nil }
        }
        if removed > 0 {
            lock.withLock { listing = nil }
            decoding.withLock { try? reading?.handle.close(); reading = nil; decoded = nil }
        }
        for folder in (try? fm.contentsOfDirectory(at: stills, includingPropertiesForKeys: nil)) ?? [] {
            guard let files = try? fm.contentsOfDirectory(at: folder, includingPropertiesForKeys: [.contentModificationDateKey]) else { continue }
            var left = files.count
            for file in files {
                guard let modified = (try? file.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate,
                      modified.timeIntervalSince1970 < cutoff else { continue }
                if (try? fm.removeItem(at: file)) != nil { removed += 1; left -= 1 }
            }
            if left == 0 { try? fm.removeItem(at: folder) }
        }
        return removed
    }

    /// Delete everything: every segment and every file written for the index. The folders stay.
    public func clear() {
        writing.lock(); defer { writing.unlock() }
        closeWriters()
        decoding.withLock { try? reading?.handle.close(); reading = nil; decoded = nil }
        let fm = FileManager.default
        for dir in [history, stills] {
            for item in (try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? [] { try? fm.removeItem(at: item) }
        }
        lock.withLock { tables.removeAll(); listing = nil }
    }

    /// Bytes on disk: the history, and the files written for the index.
    public func diskUse() -> (history: Int, stills: Int) {
        func size(_ dir: URL) -> Int {
            guard let walk = FileManager.default.enumerator(at: dir, includingPropertiesForKeys: [.fileSizeKey]) else { return 0 }
            var total = 0
            for case let url as URL in walk { total += (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0 }
            return total
        }
        return (size(history), size(stills))
    }
}

private extension Data {
    mutating func append<T: FixedWidthInteger>(le value: T) {
        Swift.withUnsafeBytes(of: value.littleEndian) { append(contentsOf: $0) }
    }
    func le<T: FixedWidthInteger>(at offset: Int) -> T {
        var v: T = 0
        Swift.withUnsafeMutableBytes(of: &v) { $0.copyBytes(from: self[startIndex + offset ..< startIndex + offset + MemoryLayout<T>.size]) }
        return T(littleEndian: v)
    }
    func sub(_ offset: Int, _ count: Int) -> Data { self[startIndex + offset ..< startIndex + offset + count] }
}
