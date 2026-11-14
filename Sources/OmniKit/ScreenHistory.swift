import Foundation
import CoreVideo
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

/// THE SCREEN HISTORY (docs/screenshots.md): every shot of the desktop, as video, and nothing else.
/// The few shots worth finding again (ScreenGate) are flagged in the video, and the index holds
/// each under a path that names it: the frame is read back out of the video when the index embeds
/// it, and again whenever a result is shown, read or dragged out.
///
/// `history` holds the segments: one file per display per hour at most, HEVC frames appended one
/// record at a time (ScreenCodec), each record carrying its time, the app in front and its window's
/// title. `stills` is where an indexed shot's path points and where no file is: until 0.20.2 a
/// picture file was written there for each (`migrateStills` is what takes them away).
///
/// A SHOT'S PATH IS SPELLED FROM ITS SEGMENT ALONE: `<stills>/<day>/<day> <time> <app> - <title>.heic`
/// in the local time the segment was recorded in, which the segment's header keeps as an offset
/// from UTC. Spelled in the time zone of the Mac today, every path would change on a flight or at
/// the end of summer time, and the index would take each for a new file.
///
/// A segment is only ever appended to by the process that created it, and a new one starts at every
/// launch, so a record cut short by a kill is the last thing in its file: a reader stops at it and
/// nothing needs repairing. The frame table is read from the segments themselves. The one file
/// beside them, `shots.catalog`, is a list of each closed segment's indexed shots that a pass reads
/// in place of the whole video; it is checked against each segment's size and date and written
/// again from the segments whenever one differs.
public final class ScreenHistory: @unchecked Sendable {
    public struct Shot: Sendable, Hashable, Identifiable {
        public let time: Double
        public let display: UInt32
        /// Also in the index.
        public var indexed: Bool
        public let app: String
        public let title: String
        public let segment: String
        public let index: Int
        /// Which of the paths its second, app and title spell is this shot's: 1 for the plain one,
        /// 2...8 for "... 2.heic" (two displays showing one window in one second). 0 for a shot
        /// that has no path: one not indexed, or one whose picture is still a file (`migrateStills`).
        public var number: Int = 0
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
        /// The shot's path in the index, when the gate admitted it.
        public let path: String?
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
    /// THE INDEX READS THE DESKTOP AT A PIXEL A POINT, never more than the frame has: 1920 x 1080 for
    /// a 1920-point desktop on any display. The frame is scaled to that before it is embedded and
    /// before its text is read (ScreenText), and the text is what the size is for: of the words
    /// confirmed on 38 real screenshots, 1568 on the long side (0.20.0) gave the reader 87.4%, a
    /// pixel a point 92.9%; more pixels a point gained nothing (docs/screenshots.md, "The reader").
    /// A segment's header keeps this side; the full frame is what the history stores.
    public static func stillSide(width: Int, height: Int, scale: Double) -> Int {
        let long = max(width, height)
        return min(long, Int((Double(long) / max(scale, 1)).rounded()))
    }
    public static let stillQuality = 0.8

    public let history: URL
    public let stills: URL
    /// Where the paths of the indexed shots begin: `stills` as the index spells the files of a
    /// folder (links resolved), for a folder that is not there.
    public let root: String

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

    /// What a segment's header says of the paths of its shots. `side` 0: written before 0.20.3 and
    /// not migrated, or by a writer that had none to give; such a segment's shots have no path.
    private struct Info { let offset: Int; let side: Int; let width: Int; let height: Int }

    private final class Writer {
        let name: String, handle: FileHandle, encoder: ScreenFrameEncoder
        let started: Double, day: Int, info: Info
        var size: UInt64
        init(name: String, handle: FileHandle, encoder: ScreenFrameEncoder, started: Double, day: Int, info: Info, size: UInt64) {
            self.name = name; self.handle = handle; self.encoder = encoder; self.started = started; self.day = day
            self.info = info; self.size = size
        }
    }

    private let lock = NSLock()            // tables, listing
    private var tables: [String: [Frame]] = [:]
    /// Counts the tables taken back because their segment's file changed or went. What was read
    /// of a file under one count is not kept, nor a reader's place used, under another.
    private var tableChanges = 0
    private var infos: [String: Info] = [:]
    private var listing: [Ref]?
    private var live = Set<String>()       // the segments this process is writing
    private let cataloging = NSLock()      // shots.catalog: one reader-writer at a time
    private let writing = NSLock()         // record, stop, prune, clear: one at a time
    private var writers: [UInt32: Writer] = [:]
    private var lastStored: [UInt32: ScreenSignature] = [:]
    private var failure: String?
    private var gates: [UInt32: ScreenGate] = [:]
    /// A place in the video being read: a decoder, the frame it gave last and the segment it has
    /// open. One while one frame is asked for at a time; `readerCount` while several are (a page
    /// of results that are shots, each a jump to another key frame: docs/screenshots.md, "A shot
    /// as a file"), and one again when they have been given.
    private final class Reader {
        var decoder = ScreenFrameDecoder()
        var decoded: (segment: String, index: Int, pixels: CVPixelBuffer)?
        var reading: (segment: String, handle: FileHandle)?
        var changes = 0             // the count of table changes its place and its file are of
        var busy = false
        var used = 0
        func forget() { try? reading?.handle.close(); reading = nil; decoded = nil }
    }
    /// As many as the window decodes thumbnails at once. Forty shots spread over the days, four
    /// asked for at a time: 0.9 s through one reader, 0.2 s through four. A drag writes out as
    /// many shots at once and no more (FileDrag.shotFiles).
    public static let readerCount = 4
    private let decoding = NSCondition()   // readers, and each reader that is not busy
    private var readers: [Reader] = []
    private var readerUse = 0
    private var dropsWaiting = 0

    private let segmentFrames: Int

    public init(history: URL, stills: URL, segmentFrames: Int = ScreenHistory.maxSegmentFrames) {
        self.history = history; self.stills = stills; self.segmentFrames = segmentFrames
        let parent = stills.deletingLastPathComponent().path
        let real = realpath(parent, nil).map { p in defer { free(p) }; return String(cString: p) } ?? parent
        root = real + "/" + stills.lastPathComponent
    }

    // MARK: Recording

    /// Take one shot into the history. nil when the screen shows what the last stored shot of this
    /// display shows (nothing is written), or when it could not be encoded.
    ///
    /// `settled`: the user has just stopped. What the screen was left showing is indexed if any of it
    /// is new, even when that is the shot already stored.
    ///
    /// `scale`: the display's pixels a point (2 on a Retina display), which sizes what the index reads.
    public func record(_ pixels: CVPixelBuffer, time: Double, display: UInt32, app: String, title: String, settled: Bool,
                       scale: Double = 1) -> Recorded? {
        writing.lock(); defer { writing.unlock() }
        failure = nil
        // 0.20.2's picture files are taken in before a shot is recorded beside them, whoever is
        // first to ask: a shot indexed now could be given a name one of them has. If they cannot
        // be (tried again a minute on), shots are stored and none is indexed.
        if stillsPending, triedStills == 0 || ProcessInfo.processInfo.systemUptime - triedStills >= 60 {
            triedStills = ProcessInfo.processInfo.systemUptime
            takeInStills(step: nil)
        }
        guard let signature = ScreenSignature(pixels) else { failure = "The frame could not be read."; return nil }
        if gates[display] == nil { gates[display] = gate(for: display) }
        if let last = lastStored[display], signature.changedShare(from: last) == 0 {
            guard settled, let writer = writers[display] else { return nil }
            let stored: Frame? = lock.withLock { tables[writer.name]?.last }
            guard let stored, !stored.shot.indexed else { return nil }
            let novelty = gates[display, default: ScreenGate()].novelty(signature)
            // A shot with no path left to take is not one the index can hold: asked before the gate
            // is, which remembers what it admits.
            guard let number = freeNumber(time: stored.shot.time, info: writer.info, app: stored.shot.app, title: stored.shot.title),
                  gates[display, default: ScreenGate()].admit(signature, settled: true) else { return nil }
            // The record is already on disk; only its flag changes.
            do {
                try writer.handle.seek(toOffset: stored.record + 16)
                try writer.handle.write(contentsOf: Data([Self.flags(isKey: stored.isKey, indexed: true, number: number)]))
                _ = try writer.handle.seekToEnd()
            } catch {
                failure = error.localizedDescription
                drop(writer, display: display)
                return nil
            }
            var shot = stored.shot
            shot.indexed = true
            shot.number = number
            lock.withLock { tables[writer.name]?[shot.index].shot = shot }
            return Recorded(shot: shot, path: path(of: shot), novelty: novelty, bytes: 0)
        }
        let novelty = gates[display, default: ScreenGate()].novelty(signature)
        guard let writer = writer(for: display, pixels: pixels, time: time, scale: scale) else {
            failure = "The video could not be started."
            return nil
        }
        let cleanApp = Self.clean(app, limit: 40), cleanTitle = Self.clean(title, limit: 80)
        let number = freeNumber(time: time, info: writer.info, app: cleanApp, title: cleanTitle)
        let flagged = number != nil && gates[display, default: ScreenGate()].admit(signature, settled: settled)
        guard let encoded = writer.encoder.encode(pixels) else {
            failure = "The frame could not be encoded."
            drop(writer, display: display)
            return nil
        }
        let (record, payloadOffset) = Self.record(time: time, isKey: encoded.isKey, indexed: flagged, number: flagged ? number ?? 0 : 0,
                                                  app: cleanApp, title: cleanTitle, sets: encoded.parameterSets, payload: encoded.data)
        do { try writer.handle.write(contentsOf: record) } catch {
            failure = error.localizedDescription
            drop(writer, display: display)
            return nil
        }
        let at = writer.size
        writer.size += UInt64(record.count)
        lastStored[display] = signature
        let shot: Shot = lock.withLock {
            var table = tables[writer.name] ?? []
            let key = encoded.isKey ? table.count : (table.last?.key ?? 0)
            let shot = Shot(time: time, display: display, indexed: flagged, app: cleanApp, title: cleanTitle, segment: writer.name,
                            index: table.count, number: flagged ? number ?? 0 : 0)
            table.append(Frame(shot: shot, isKey: encoded.isKey, sets: encoded.parameterSets, record: at,
                               payload: at + UInt64(payloadOffset), length: encoded.data.count, key: key))
            tables[writer.name] = table
            return shot
        }
        return Recorded(shot: shot, path: path(of: shot), novelty: novelty, bytes: record.count)
    }

    /// The first of a name's numbers no indexed shot has: the plain name, unless another display
    /// showed the same window in the same second. nil when all eight are taken. Called under `writing`.
    private func freeNumber(time: Double, info: Info, app: String, title: String) -> Int? {
        guard info.side > 0, !stillsPending else { return nil }
        let second = Double(Self.second(time))
        let taken = Set(shots(from: second - 1, to: second + 1).compactMap { path(of: $0) })
        return (1 ... Self.stillNumbers).first {
            !taken.contains(root + "/" + Self.spelled(time: time, offset: info.offset, app: app, title: title, number: $0))
        }
    }

    /// Why the last `record` stored nothing, when it was not that the screen showed what the last
    /// stored shot shows. A `nil` from `record` is both, and a recorder that could not tell them
    /// apart looked the same whether the screen was still or nothing could be written.
    public var lastFailure: String? { writing.lock(); defer { writing.unlock() }; return failure }

    /// A segment whose encoder or file failed is closed: the next shot starts another, with a new
    /// encoder. Kept, it failed every shot until its hour ran out. Called under `writing`.
    private func drop(_ writer: Writer, display: UInt32) {
        try? writer.handle.close()
        lock.withLock { _ = live.remove(writer.name) }
        writers[display] = nil
        lastStored[display] = nil
    }

    /// Close the open segments. The next shot starts new ones.
    public func stop() {
        writing.lock(); defer { writing.unlock() }
        closeWriters()
    }

    private func closeWriters() {
        for w in writers.values { try? w.handle.close() }
        lock.withLock { live.removeAll() }
        writers.removeAll(); lastStored.removeAll(); gates.removeAll()
    }

    /// How far back a starting recorder looks for what the index holds of a display.
    private static let gateSegments = 24

    /// The gate a display starts with: the shots of it the index already holds. A recorder that
    /// starts with an empty one (a launch, Screenshots turned off and on) indexes the screen it finds
    /// whatever is on it: on the first real session every start added one more picture of the same
    /// terminal (docs/screenshots.md, "A real session"). Read back from the segments like everything
    /// else here, so there is no second file to keep true: the last indexed shots, decoded. Called
    /// under `writing`.
    private func gate(for display: UInt32) -> ScreenGate {
        var found: [Shot] = []
        walk: for ref in refs().reversed().lazy.filter({ $0.display == display }).prefix(Self.gateSegments) {
            for f in frames(of: ref.name).reversed() where f.shot.indexed {
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

    /// A SEGMENT IS ONE SIZE, ONE HOUR, ONE DAY, ONE TIME ZONE AND ONE SCALE: the last two are what
    /// its shots' paths are spelled from, so a change of either starts the next segment.
    private func writer(for display: UInt32, pixels: CVPixelBuffer, time: Double, scale: Double) -> Writer? {
        let width = CVPixelBufferGetWidth(pixels), height = CVPixelBufferGetHeight(pixels)
        let day = Self.day(time)
        let info = Info(offset: Self.offset(at: time),
                        side: min(Self.stillSide(width: width, height: height, scale: scale), Int(UInt16.max)), width: width, height: height)
        if let w = writers[display] {
            let count = lock.withLock { tables[w.name]?.count ?? 0 }
            if w.encoder.width == width, w.encoder.height == height, count < segmentFrames,
               time - w.started < Self.maxSegmentSeconds, time >= w.started, w.day == day,
               w.info.offset == info.offset, w.info.side == info.side { return w }
            try? w.handle.close()
            lock.withLock { _ = live.remove(w.name) }
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
        for v in [UInt32(width), UInt32(height), display, Self.packed(info)] { header.append(le: v) }
        let url = history.appendingPathComponent(name)
        guard FileManager.default.createFile(atPath: url.path, contents: header),
              let handle = try? FileHandle(forUpdating: url) else { return nil }
        _ = try? handle.seekToEnd()
        let w = Writer(name: name, handle: handle, encoder: encoder, started: Double(start), day: day, info: info, size: UInt64(header.count))
        writers[display] = w
        lock.withLock {
            tables[name] = []
            infos[name] = info
            live.insert(name)
            listing?.append(Ref(name: name, start: Double(start), display: display))
        }
        return w
    }

    // MARK: The segment file

    private static let headerLength = 24
    private static let marker: UInt32 = 0x544F_4853      // "SHOT"
    /// Where the header keeps `Info`: the four bytes 0.20.2 and before wrote as zero.
    private static let infoOffset: UInt64 = 20

    /// The header's last field: the offset from UTC in minutes (Int16) below the side (UInt16).
    private static func packed(_ info: Info) -> UInt32 {
        UInt32(UInt16(bitPattern: Int16(clamping: info.offset))) | UInt32(UInt16(clamping: info.side)) << 16
    }

    private static func info(packed v: UInt32, width: Int, height: Int) -> Info {
        Info(offset: Int(Int16(bitPattern: UInt16(v & 0xFFFF))), side: Int(v >> 16), width: width, height: height)
    }

    /// A record's flags. Bit 0: a key frame. Bit 1: in the index. Bits 2-4: the shot's number less
    /// one. Bit 5: the number is there, which is to say the shot's picture is its frame and not a
    /// file (a reader before 0.20.3 reads the first two bits and nothing of the rest).
    private static func flags(isKey: Bool, indexed: Bool, number: Int) -> UInt8 {
        var out: UInt8 = isKey ? 1 : 0
        if indexed { out |= 2 }
        if indexed, number > 0 { out |= 0x20 | UInt8((number - 1) & 7) << 2 }
        return out
    }

    /// 0 for a shot that is not in the index, and for one 0.20.2 flagged whose picture was a file:
    /// until `migrateStills` has given it a number it has no path, and nothing here takes it for indexed.
    private static func number(flags: UInt8) -> Int {
        flags & 2 != 0 && flags & 0x20 != 0 ? Int((flags >> 2) & 7) + 1 : 0
    }

    /// One record, and where its payload starts inside it.
    private static func record(time: Double, isKey: Bool, indexed: Bool, number: Int, app: String, title: String, sets: [Data],
                               payload: Data) -> (Data, Int) {
        let appBytes = Data(app.utf8), titleBytes = Data(title.utf8)
        var body = Data()
        body.append(le: time.bitPattern)
        body.append(flags(isKey: isKey, indexed: indexed, number: number))
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

    /// One record as it lies in its segment, without its frame.
    private struct Head {
        let at: Int                 // where the record starts
        let time: Double
        let flags: UInt8
        let names: [UInt8]          // the app's name, then the window's title
        let appLength: Int
        let sets: [Data]
        let payload: Int            // where the frame starts
        let length: Int             // and how long it is
        var app: String { String(decoding: names[..<appLength], as: UTF8.self) }
        var title: String { String(decoding: names[appLength...], as: UTF8.self) }
    }

    private struct Segment { let info: Info; let display: UInt32; let size: Int; let modified: Int }

    /// What one read at a record's head takes: the head, both names and the parameter sets of
    /// nearly every record, and the closing length of the record before it.
    private static let headRead = 1024

    /// A segment's header and every whole record after it, without the frames: one small read a
    /// record, so listing a segment does not read its video. A record cut short ends the walk.
    /// nil when the file is not a segment.
    private static func walk(_ url: URL, each: (Head) -> Void) -> Segment? {
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else { return nil }
        defer { close(fd) }
        var st = stat()
        guard fstat(fd, &st) == 0 else { return nil }
        let size = Int(st.st_size)
        func read(_ at: Int, _ count: Int) -> [UInt8]? {
            guard count > 0, at >= 0, at + count <= size else { return nil }
            var out = [UInt8](repeating: 0, count: count)
            let got = out.withUnsafeMutableBytes { pread(fd, $0.baseAddress, count, off_t(at)) }
            return got == count ? out : nil
        }
        func le<T: FixedWidthInteger>(_ bytes: [UInt8], _ at: Int) -> T {
            var v: T = 0
            for i in 0 ..< MemoryLayout<T>.size { v |= T(truncatingIfNeeded: bytes[at + i]) << T(8 * i) }
            return v
        }
        guard let header = read(0, headerLength), header.prefix(8).elementsEqual("OMNISCR1".utf8) else { return nil }
        let segment = Segment(info: info(packed: le(header, Int(infoOffset)), width: Int(le(header, 8) as UInt32), height: Int(le(header, 12) as UInt32)),
                              display: le(header, 16), size: size,
                              modified: Int(st.st_mtimespec.tv_sec) * 1_000_000_000 + Int(st.st_mtimespec.tv_nsec))
        var at = headerLength
        var block = read(at, min(headRead, size - at)) ?? []
        while block.count >= 26 {
            let mark: UInt32 = le(block, 0), length = Int(le(block, 4) as UInt32)
            guard mark == marker, length >= 18, at + 8 + length + 4 <= size else { break }
            let time = Double(bitPattern: le(block, 8) as UInt64)
            let flags = block[16], setCount = Int(block[17])
            let appLength = Int(le(block, 18) as UInt16), titleLength = Int(le(block, 20) as UInt16)
            let payloadLength = Int(le(block, 22) as UInt32)
            // What lies before the frame: the names, then the parameter sets.
            let before = length - 18 - payloadLength
            guard before >= appLength + titleLength else { break }
            if 26 + before > block.count {
                guard let more = read(at, 26 + before) else { break }
                block = more
            }
            var p = 26 + appLength + titleLength
            var sets: [Data] = []
            var whole = true
            for _ in 0 ..< setCount {
                guard p + 2 <= 26 + before else { whole = false; break }
                let n = Int(le(block, p) as UInt16); p += 2
                guard p + n <= 26 + before else { whole = false; break }
                sets.append(Data(block[p ..< p + n])); p += n
            }
            guard whole, p == 26 + before else { break }
            // The record's closing length, read with the head of the next record.
            let end = at + 8 + length
            guard let next = read(end, min(4 + headRead, size - end)), Int(le(next, 0) as UInt32) == length else { break }
            each(Head(at: at, time: time, flags: flags, names: Array(block[26 ..< 26 + appLength + titleLength]), appLength: appLength,
                      sets: sets, payload: at + 26 + before, length: payloadLength))
            at = end + 4
            block = Array(next.dropFirst(4))
        }
        return segment
    }

    /// Every whole record of a segment.
    private static func scan(_ url: URL, name: String) -> [Frame] {
        var heads: [Head] = []
        guard let segment = walk(url, each: { heads.append($0) }) else { return [] }
        var frames: [Frame] = []
        frames.reserveCapacity(heads.count)
        var key = 0
        for h in heads {
            let isKey = h.flags & 1 != 0
            if isKey { key = frames.count }
            let number = number(flags: h.flags)
            let shot = Shot(time: h.time, display: segment.display, indexed: number > 0, app: h.app, title: h.title, segment: name,
                            index: frames.count, number: number)
            frames.append(Frame(shot: shot, isKey: isKey, sets: h.sets, record: UInt64(h.at), payload: UInt64(h.payload), length: h.length, key: key))
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

    private func frames(of name: String) -> [Frame] { framesNow(of: name).table }

    /// A segment's table and the count of changes it was read under. A segment rewritten while it
    /// was being read is read again: the table kept is never one of a file that is gone.
    private func framesNow(of name: String) -> (table: [Frame], changes: Int) {
        while true {
            let (cached, seen) = lock.withLock { (tables[name], tableChanges) }
            if let cached { return (cached, seen) }
            let scanned = Self.scan(history.appendingPathComponent(name), name: name)
            let kept: [Frame]? = lock.withLock {
                guard tableChanges == seen else { return nil }
                if let raced = tables[name] { return raced }
                tables[name] = scanned
                return scanned
            }
            if let kept { return (kept, seen) }
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
        // Twice: a table read before its segment was rewritten does not say where the frames are.
        for _ in 0 ..< 2 {
            let (table, changes) = framesNow(of: shot.segment)
            // A shot from before a deletion in its segment may be another frame's place now.
            guard shot.index < table.count, table[shot.index].shot.time == shot.time else { return nil }
            var moved = false
            if let pixels = read(shot, table, changes: changes, moved: &moved) { return pixels }
            guard moved else { return nil }
            lock.withLock {
                guard !live.contains(shot.segment), tableChanges == changes else { return }
                tables[shot.segment] = nil
                tableChanges += 1
            }
        }
        return nil
    }

    /// The frame at `shot`, decoded from its key frame or from where a reader already is.
    /// `moved`: the file does not hold the table's records where the table has them.
    private func read(_ shot: Shot, _ table: [Frame], changes: Int, moved: inout Bool) -> CVPixelBuffer? {
        var first = table[shot.index].key
        let reader = takeReader(segment: shot.segment, from: first, to: shot.index, changes: changes)
        defer { giveBack(reader) }
        if reader.changes != changes {
            reader.forget()
            reader.decoder = ScreenFrameDecoder()
            reader.changes = changes
        }
        if let decoded = reader.decoded, decoded.segment == shot.segment, decoded.index == shot.index { return decoded.pixels }
        if let decoded = reader.decoded, decoded.segment == shot.segment, decoded.index >= first, decoded.index < shot.index { first = decoded.index + 1 }
        if reader.reading?.segment != shot.segment {
            reader.forget()
            guard let handle = try? FileHandle(forReadingFrom: history.appendingPathComponent(shot.segment)) else { return nil }
            reader.reading = (shot.segment, handle)
        }
        guard let handle = reader.reading?.handle else { return nil }
        var out: CVPixelBuffer?
        for i in first ... shot.index {
            let f = table[i]
            guard let data = Self.payload(of: f, in: handle) else {
                moved = true
                reader.forget()
                reader.decoder = ScreenFrameDecoder()
                return nil
            }
            guard let image = reader.decoder.decode(data, parameterSets: f.sets) else {
                reader.decoded = nil
                reader.decoder = ScreenFrameDecoder()
                return nil
            }
            out = image
        }
        guard let out else { return nil }
        reader.decoded = (shot.segment, shot.index, out)
        return out
    }

    /// A frame's bytes, when the file has its record where the table says it is: the record's
    /// mark, the shot's time and the frame's length and place are read back first.
    private static func payload(of f: Frame, in handle: FileHandle) -> Data? {
        let fd = handle.fileDescriptor
        func bytes(_ at: UInt64, _ count: Int) -> Data? {
            guard count > 0 else { return nil }
            var out = Data(count: count)
            let got = out.withUnsafeMutableBytes { pread(fd, $0.baseAddress, count, off_t(at)) }
            return got == count ? out : nil
        }
        guard let head = bytes(f.record, 26) else { return nil }
        let mark: UInt32 = head.le(at: 0), body: UInt32 = head.le(at: 4), time: UInt64 = head.le(at: 8), length: UInt32 = head.le(at: 22)
        guard mark == marker, time == f.shot.time.bitPattern, Int(length) == f.length,
              f.record + 8 + UInt64(body) == f.payload + UInt64(f.length) else { return nil }
        return bytes(f.payload, f.length)
    }

    /// A reader for the frames `from ... to` of a segment, the caller's alone until given back:
    /// one already on the way there (the one furthest along), else one nobody is using, else a
    /// new one while there are fewer than `readerCount`, else the wait for one.
    private func takeReader(segment: String, from: Int, to: Int, changes: Int) -> Reader {
        decoding.lock(); defer { decoding.unlock() }
        while true {
            // A deletion waiting for the reads under way goes before a read that has not begun.
            guard dropsWaiting == 0 else { decoding.wait(); continue }
            let idle = readers.filter { !$0.busy }
            func place(_ r: Reader) -> Int? {
                guard r.changes == changes, let d = r.decoded, d.segment == segment, d.index >= from, d.index <= to else { return nil }
                return d.index
            }
            var reader = idle.filter { place($0) != nil }.max { place($0)! < place($1)! } ?? idle.min { $0.used < $1.used }
            if reader == nil, readers.count < Self.readerCount {
                let made = Reader()
                readers.append(made)
                reader = made
            }
            if let reader {
                reader.busy = true
                readerUse += 1
                reader.used = readerUse
                return reader
            }
            decoding.wait()
        }
    }

    /// With nothing being read there is one reader, the one used last: a frame at the display's
    /// size and a decoder are not kept for each place a drawn page was read at.
    private func giveBack(_ reader: Reader) {
        decoding.lock(); defer { decoding.unlock() }
        reader.busy = false
        if !readers.contains(where: \.busy) {
            for other in readers where other !== reader { other.forget() }
            readers = [reader]
        }
        decoding.broadcast()
    }

    /// Before a segment's file changes or goes: waits out the reads under way and drops every
    /// reader, with the file it had open and the frame it held.
    private func dropReaders() {
        decoding.lock(); defer { decoding.unlock() }
        dropsWaiting += 1
        while readers.contains(where: \.busy) { decoding.wait() }
        for reader in readers { reader.forget() }
        readers = []
        dropsWaiting -= 1
        decoding.broadcast()
    }

    /// How many readers are kept: one at most with nothing being read.
    var readersKept: Int { decoding.withLock { readers.count } }

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

    // MARK: The paths the index holds

    /// The second a shot's name carries. A name is spelled from the time to the millisecond (0.20.2
    /// spelled it with a DateFormatter, which rounds there), so a shot taken in the last half
    /// millisecond of a second is named for the next.
    private static func second(_ time: Double) -> Int { Int(((time * 1000).rounded() / 1000).rounded(.down)) }

    /// A day and a time of day as a name spells them, for a count of seconds since 1970 on a clock
    /// with no time zone (days to a civil date: Howard Hinnant's algorithm).
    private static func wallClock(_ seconds: Int) -> (day: String, time: String) {
        let days = Int((Double(seconds) / 86400).rounded(.down)), rest = seconds - days * 86400
        let z = days + 719_468
        let era = (z >= 0 ? z : z - 146_096) / 146_097
        let doe = z - era * 146_097
        let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100)
        let mp = (5 * doy + 2) / 153
        let d = doy - (153 * mp + 2) / 5 + 1
        let m = mp < 10 ? mp + 3 : mp - 9
        let y = yoe + era * 400 + (m <= 2 ? 1 : 0)
        func two(_ v: Int) -> String { v < 10 ? "0\(v)" : "\(v)" }
        return ("\(y)-\(two(m))-\(two(d))", "\(two(rest / 3600)).\(two(rest / 60 % 60)).\(two(rest % 60))")
    }

    /// The seconds a name's first nineteen characters ("2026-11-10 14.05.32") spell, on that clock.
    private static func wallSeconds(_ name: String) -> Int? {
        let u = Array(name.utf8.prefix(19))
        guard u.count == 19, u[4] == 45, u[7] == 45, u[10] == 32, u[13] == 46, u[16] == 46 else { return nil }
        func number(_ range: Range<Int>) -> Int? {
            var v = 0
            for c in u[range] { guard c >= 48, c <= 57 else { return nil }; v = v * 10 + Int(c - 48) }
            return v
        }
        guard let y = number(0 ..< 4), let m = number(5 ..< 7), let d = number(8 ..< 10), let h = number(11 ..< 13),
              let min = number(14 ..< 16), let sec = number(17 ..< 19), (1 ... 12).contains(m), (1 ... 31).contains(d) else { return nil }
        let yy = m <= 2 ? y - 1 : y
        let era = (yy >= 0 ? yy : yy - 399) / 400
        let yoe = yy - era * 400
        let doy = (153 * (m > 2 ? m - 3 : m + 9) + 2) / 5 + d - 1
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy
        return (era * 146_097 + doe - 719_468) * 86400 + h * 3600 + min * 60 + sec
    }

    /// A shot's path below the root: "2026-11-10/2026-11-10 14.05.32 Xcode - Sidebar.swift.heic",
    /// numbered from the second on ("... 2.heic").
    ///
    /// SPELLED AS THE FILE SYSTEM SPELLS A NAME. 0.20.2 wrote a file here and the index holds the
    /// path the folder then listed, which is not the title's own bytes: a file made through
    /// Foundation has its name decomposed ("é" as "e" and an accent, Hangul as jamo; measured, 34
    /// bytes on disk for a 20-byte name). The path of a shot is that spelling, for the shots that
    /// had a file and for the ones that never will.
    private static func spelled(time: Double, offset: Int, app: String, title: String, number: Int) -> String {
        let wall = wallClock(second(time) + offset * 60)
        var base = wall.day + " " + wall.time
        let subject = [app, title].filter { !$0.isEmpty }.joined(separator: " - ")
        if !subject.isEmpty { base += " " + subject }
        if number > 1 { base += " \(number)" }
        return String(cString: (wall.day + "/" + base + ".heic" as NSString).fileSystemRepresentation)
    }

    /// The most shots that can share a second, an app and a title: one per display, numbered.
    private static let stillNumbers = 8

    private func info(of name: String) -> Info? {
        let (known, seen) = lock.withLock { (infos[name], tableChanges) }
        if let known { return known }
        guard let handle = try? FileHandle(forReadingFrom: history.appendingPathComponent(name)) else { return nil }
        defer { try? handle.close() }
        guard let header = try? handle.read(upToCount: Self.headerLength), header.count == Self.headerLength,
              header.prefix(8) == Data("OMNISCR1".utf8) else { return nil }
        let info = Self.info(packed: header.le(at: Int(Self.infoOffset)), width: Int(header.le(at: 8) as UInt32), height: Int(header.le(at: 12) as UInt32))
        lock.withLock { if tableChanges == seen { infos[name] = info } }
        return info
    }

    /// True for a path under the root: an indexed shot's, or what was one.
    public func isShotPath(_ path: String) -> Bool { path.hasPrefix(root + "/") }

    /// Where the index holds a shot. nil for a shot that is not in it.
    public func path(of shot: Shot) -> String? {
        guard shot.number > 0, let info = info(of: shot.segment), info.side > 0 else { return nil }
        return root + "/" + Self.spelled(time: shot.time, offset: info.offset, app: shot.app, title: shot.title, number: shot.number)
    }

    /// The shot a path names, while the history holds it.
    ///
    /// A name gives a second on the clock of whatever time zone its segment was recorded in, so the
    /// segments that could hold it are the ones begun within a day of that second on any clock;
    /// each says by its own offset which second it would be, and the shot is the one there whose
    /// path is this one.
    public func shot(atPath path: String) -> Shot? { frame(atPath: path)?.shot }

    /// The shot a path names, as a pass lists it (`indexedShots`).
    public func entry(atPath path: String) -> Entry? {
        frame(atPath: path).map { Entry(path: path, time: $0.shot.time, bytes: $0.length) }
    }

    private func frame(atPath path: String) -> Frame? {
        guard isShotPath(path), let wall = Self.wallSeconds((path as NSString).lastPathComponent) else { return nil }
        let reach = 14 * 3600.0 + Self.maxSegmentSeconds + 2
        for ref in refs() where ref.start >= Double(wall) - reach && ref.start <= Double(wall) + reach {
            guard let info = info(of: ref.name), info.side > 0 else { continue }
            let second = wall - info.offset * 60
            guard Double(second) >= ref.start - 1, Double(second) <= ref.start + Self.maxSegmentSeconds + 1 else { continue }
            for f in frames(of: ref.name) where f.shot.number > 0 && Self.second(f.shot.time) == second {
                if self.path(of: f.shot) == path { return f }
            }
        }
        return nil
    }

    // MARK: - A shot as a link

    /// What a served answer names a shot by (docs/screenshots.md, "A shot as a link"). The path
    /// the index holds has no file behind it and spaces in it, so an agent that opens it with its
    /// own reader finds nothing; a link says it is Omni's to read. Its second in UTC as fourteen
    /// digits (local time repeats when the clocks go back), and `/<display>` when another indexed
    /// shot shares that second.
    public static let linkScheme = "omni://screenshots"

    public func link(of shot: Shot) -> String? {
        guard shot.number > 0 else { return nil }
        let second = Self.second(shot.time)
        let wall = Self.wallClock(second)
        let digits = String((wall.day + wall.time).filter(\.isNumber))
        let shared = indexedFrames(atSecond: second).contains { $0.shot.id != shot.id }
        return Self.linkScheme + "/" + digits + (shared ? "/\(shot.display)" : "")
    }

    /// The link of the shot a path names, while the history holds it.
    public func link(forPath path: String) -> String? { shot(atPath: path).flatMap(link(of:)) }

    /// The shot a link names. nil when no indexed shot is at that second, or when more than one is
    /// and the link names no display.
    public func shot(atLink link: String) -> Shot? {
        guard link.hasPrefix(Self.linkScheme + "/") else { return nil }
        let parts = link.dropFirst(Self.linkScheme.count + 1).split(separator: "/", omittingEmptySubsequences: false)
        guard (1 ... 2).contains(parts.count), let second = Self.linkSeconds(parts[0]) else { return nil }
        var display: UInt32?
        if parts.count == 2 { guard let d = UInt32(parts[1]) else { return nil }; display = d }
        let found = indexedFrames(atSecond: second).filter { display == nil || $0.shot.display == display }
        return found.count == 1 ? found[0].shot : nil
    }

    /// The path the index holds for the shot a link names; the root for the bare scheme.
    public func path(forLink link: String) -> String? {
        if link == Self.linkScheme || link == Self.linkScheme + "/" { return root }
        return shot(atLink: link).flatMap(path(of:))
    }

    /// "20261112200123": the seconds since 1970 it spells in UTC.
    private static func linkSeconds(_ digits: Substring) -> Int? {
        let u = Array(digits.utf8)
        guard u.count == 14, u.allSatisfy({ $0 >= 48 && $0 <= 57 }) else { return nil }
        let d = String(digits)
        func at(_ i: Int, _ n: Int) -> String { String(d.dropFirst(i).prefix(n)) }
        return wallSeconds(at(0, 4) + "-" + at(4, 2) + "-" + at(6, 2) + " " + at(8, 2) + "." + at(10, 2) + "." + at(12, 2))
    }

    /// The indexed shots of one second, on every display: the segments that could hold it.
    private func indexedFrames(atSecond second: Int) -> [Frame] {
        let t = Double(second)
        return refs().filter { $0.start <= t + 1 && $0.start >= t - Self.maxSegmentSeconds - 1 }
            .flatMap { frames(of: $0.name) }
            .filter { $0.shot.number > 0 && Self.second($0.shot.time) == second }
    }

    /// The size the index reads a shot at: the desktop at a pixel a point (`stillSide`).
    public func size(of shot: Shot) -> (width: Int, height: Int)? {
        guard let info = info(of: shot.segment), info.side > 0, info.width > 0, info.height > 0 else { return nil }
        return Self.fit(width: info.width, height: info.height, maxDimension: info.side)
    }

    /// A shot's picture, no larger than `maxDimension` (nil: as the index reads it, `size(of:)`),
    /// with the size of the frame it was cut from.
    public func picture(of shot: Shot, maxDimension: Int? = nil) -> (image: CGImage, width: Int, height: Int)? {
        guard let side = maxDimension ?? info(of: shot.segment).map(\.side), side > 0,
              let pixels = pixels(shot), let image = Self.fitted(pixels, maxDimension: side) else { return nil }
        return (image, CVPixelBufferGetWidth(pixels), CVPixelBufferGetHeight(pixels))
    }

    /// An indexed shot as a picture file in `folder`, named as its path is and dated at the shot:
    /// what leaves the app when a result is dragged out, shared or opened in another app. A HEIC at
    /// a pixel a point, marked as a screen capture, as 0.20.2 kept one for every indexed shot.
    public func export(_ shot: Shot, to folder: URL) -> URL? {
        guard let path = path(of: shot) else { return nil }
        let url = folder.appendingPathComponent((path as NSString).lastPathComponent)
        if let made = try? url.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey]), (made.fileSize ?? 0) > 0,
           let date = made.contentModificationDate, abs(date.timeIntervalSince1970 - shot.time) <= 0.002 { return url }
        guard let image = picture(of: shot)?.image else { return nil }
        do { try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true) } catch { return nil }
        let temp = folder.appendingPathComponent("." + UUID().uuidString + ".tmp")
        guard let dest = CGImageDestinationCreateWithURL(temp as CFURL, UTType.heic.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, image, ScreenCodec.stillOptions(quality: Self.stillQuality))
        guard CGImageDestinationFinalize(dest) else { try? FileManager.default.removeItem(at: temp); return nil }
        if let flag = try? PropertyListSerialization.data(fromPropertyList: true, format: .binary, options: 0) {
            _ = flag.withUnsafeBytes { setxattr(temp.path, Self.screenCaptureAttribute, $0.baseAddress, flag.count, 0, 0) }
        }
        let date = Date(timeIntervalSince1970: shot.time)
        try? FileManager.default.setAttributes([.creationDate: date, .modificationDate: date], ofItemAtPath: temp.path)
        guard rename(temp.path, url.path) == 0 else { try? FileManager.default.removeItem(at: temp); return nil }
        return url
    }

    /// What marks a file as a screen capture to Spotlight and to the index, for a shot that has
    /// left the app as a file.
    public static let screenCaptureAttribute = "com.apple.metadata:kMDItemIsScreenCapture"

    private static func fit(width: Int, height: Int, maxDimension: Int) -> (width: Int, height: Int) {
        let longest = max(width, height)
        guard longest > maxDimension else { return (width, height) }
        let k = Double(maxDimension) / Double(longest)
        return (max(1, Int((Double(width) * k).rounded())), max(1, Int((Double(height) * k).rounded())))
    }

    private static func fitted(_ pixels: CVPixelBuffer, maxDimension: Int) -> CGImage? {
        guard let image = ScreenCodec.cgImage(pixels) else { return nil }
        let (w, h) = fit(width: image.width, height: image.height, maxDimension: maxDimension)
        guard w != image.width || h != image.height else { return image }
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

    /// Minutes from UTC at a moment, on the clock a name is spelled by: the app's time zone, which
    /// is the one 0.20.2's DateFormatter named its files in (and the Mac's, unless something set
    /// another for the process: `TimeZone.current` does not follow that, measured).
    private static func offset(at time: Double) -> Int {
        NSTimeZone.default.secondsFromGMT(for: Date(timeIntervalSince1970: time)) / 60
    }

    private static func day(_ time: Double) -> Int {
        let c = Calendar.current.dateComponents([.year, .month, .day], from: Date(timeIntervalSince1970: time))
        return (c.year ?? 0) * 10_000 + (c.month ?? 0) * 100 + (c.day ?? 0)
    }

    // MARK: The shots a pass reads

    /// An indexed shot as a pass over the index sees it: its path, when it was taken, and the bytes
    /// of its frame.
    public struct Entry: Sendable {
        public let path: String
        public let time: Double
        public let bytes: Int
    }

    /// A segment's indexed shots, and the size and date of the file they were read from.
    private struct Listed {
        let size: Int, modified: Int
        var shots: [(rel: String, time: Double, bytes: Int)]
    }

    private static func list(_ url: URL) -> Listed? {
        var found: [(time: Double, number: Int, app: String, title: String, bytes: Int)] = []
        guard let segment = walk(url, each: { h in
            let n = number(flags: h.flags)
            if n > 0 { found.append((h.time, n, h.app, h.title, h.length)) }
        }) else { return nil }
        guard segment.info.side > 0 else { return Listed(size: segment.size, modified: segment.modified, shots: []) }
        return Listed(size: segment.size, modified: segment.modified, shots: found.map {
            (spelled(time: $0.time, offset: segment.info.offset, app: $0.app, title: $0.title, number: $0.number), $0.time, $0.bytes)
        })
    }

    /// Every shot the index holds or is to hold, oldest segment first. nil while 0.20.2's picture
    /// files are still to be taken in (`migrateStills`): which shots are indexed is not yet known,
    /// and a pass told "none" would drop every one of them from the index.
    ///
    /// READ FROM THE CATALOG, NOT FROM THE VIDEO. A pass asks this each time it runs, and the answer
    /// lies a few bytes at a time through every segment of the history: read cold, that is a read
    /// in every record of it (docs/screenshots.md, "The list of indexed shots"). `shots.catalog`
    /// keeps each closed segment's answer beside its size and date; a segment whose file says
    /// otherwise, or that the catalog has not met, is listed again from the segment. The catalog
    /// is never the authority: lost or damaged, it is written again from the video.
    public func indexedShots() -> [Entry]? {
        guard !stillsPending else { return nil }
        cataloging.lock(); defer { cataloging.unlock() }
        let catalog = readCatalog()
        var kept: [String: Listed] = [:]
        var out: [Entry] = []
        var changed = false
        let writing = lock.withLock { live }
        for ref in refs() {
            if writing.contains(ref.name) {
                for f in frames(of: ref.name) where f.shot.indexed {
                    if let path = path(of: f.shot) { out.append(Entry(path: path, time: f.shot.time, bytes: f.length)) }
                }
                continue
            }
            let url = history.appendingPathComponent(ref.name)
            var st = stat()
            guard stat(url.path, &st) == 0 else { continue }
            var listed = catalog[ref.name]
            if listed == nil || listed?.size != Int(st.st_size)
                || listed?.modified != Int(st.st_mtimespec.tv_sec) * 1_000_000_000 + Int(st.st_mtimespec.tv_nsec) {
                listed = Self.list(url)
                changed = true
            }
            guard let listed else { continue }
            kept[ref.name] = listed
            for s in listed.shots { out.append(Entry(path: root + "/" + s.rel, time: s.time, bytes: s.bytes)) }
        }
        if changed || kept.count != catalog.count { writeCatalog(kept) }
        return out
    }

    private static let catalogName = "shots.catalog"
    private static let catalogMagic = Data("OMNICAT1".utf8)

    private static func checksum(_ data: Data) -> UInt64 {
        var h: UInt64 = 0xcbf2_9ce4_8422_2325
        for b in data { h = (h ^ UInt64(b)) &* 0x0000_0100_0000_01b3 }
        return h
    }

    private func readCatalog() -> [String: Listed] {
        guard let data = try? Data(contentsOf: history.appendingPathComponent(Self.catalogName)), data.count >= 20,
              data.prefix(8) == Self.catalogMagic,
              Self.checksum(data.prefix(data.count - 8)) == data.le(at: data.count - 8) as UInt64 else { return [:] }
        let end = data.count - 8
        var at = 8
        func take(_ n: Int) -> Int? { guard n >= 0, at + n <= end else { return nil }; defer { at += n }; return at }
        func text() -> String? {
            guard let l = take(2) else { return nil }
            let n = Int(data.le(at: l) as UInt16)
            guard let t = take(n) else { return nil }
            return String(decoding: data.sub(t, n), as: UTF8.self)
        }
        guard let c = take(4) else { return [:] }
        var out: [String: Listed] = [:]
        for _ in 0 ..< Int(data.le(at: c) as UInt32) {
            guard let name = text(), let fixed = take(20) else { return [:] }
            var listed = Listed(size: Int(data.le(at: fixed) as UInt64), modified: Int(data.le(at: fixed + 8) as Int64), shots: [])
            for _ in 0 ..< Int(data.le(at: fixed + 16) as UInt32) {
                guard let head = take(12), let rel = text() else { return [:] }
                listed.shots.append((rel, Double(bitPattern: data.le(at: head) as UInt64), Int(data.le(at: head + 8) as UInt32)))
            }
            out[name] = listed
        }
        return at == end ? out : [:]
    }

    private func writeCatalog(_ catalog: [String: Listed]) {
        var data = Self.catalogMagic
        func text(_ s: String) { let b = Data(s.utf8.prefix(Int(UInt16.max))); data.append(le: UInt16(b.count)); data.append(b) }
        data.append(le: UInt32(catalog.count))
        for (name, listed) in catalog.sorted(by: { $0.key < $1.key }) {
            text(name)
            data.append(le: UInt64(listed.size)); data.append(le: Int64(listed.modified)); data.append(le: UInt32(listed.shots.count))
            for s in listed.shots {
                data.append(le: s.time.bitPattern); data.append(le: UInt32(s.bytes))
                text(s.rel)
            }
        }
        data.append(le: Self.checksum(data))
        try? data.write(to: history.appendingPathComponent(Self.catalogName), options: .atomic)
    }

    // MARK: The files of 0.20.2

    /// True while the folder 0.20.2 kept a picture file in for every indexed shot is still there.
    public var stillsPending: Bool { FileManager.default.fileExists(atPath: stills.path) }

    public enum Migration: Sendable, Equatable {
        /// There was no folder: nothing to do.
        case none
        /// The folder is gone. `kept`: shots the index goes on holding. `dropped`: shots flagged
        /// for the index whose file was gone, now flagged as any other shot.
        case done(kept: Int, dropped: Int)
        /// Something could not be read or written; the folder is still there and the next call
        /// starts again.
        case failed(String)
    }

    /// Where the folder goes in one step before it is deleted, so that "the folder is there" means
    /// "not taken in yet" and never "half deleted".
    private var removedStills: URL { history.appendingPathComponent("stills.removed", isDirectory: true) }

    /// TAKE 0.20.2'S PICTURE FILES INTO THE VIDEO AND DELETE THEM. Until 0.20.2 an indexed shot was
    /// a flag in its record and a file under `stills`, found again by spelling the file's name in
    /// the time zone of the day; the index holds each file's path. From 0.20.3 the path is spelled
    /// from the segment (its offset from UTC, in the header) and the record (which of the names is
    /// its own), and no file is behind it.
    ///
    /// Each segment's header takes the offset and the side its files were written with, read from
    /// one of them: the file is dated at its shot to the millisecond and named in local time, and
    /// the difference is the offset. Each flagged shot then takes the number of the file dated at
    /// it, so its path is the one the index already holds and no row of the index changes. A shot
    /// whose file is there but cannot be told from another's (its date lost, or its name spelled
    /// on another clock) takes a path no file had, and the index reads it again from its frame. A
    /// shot whose file is gone (the retention took it, or somebody did) is flagged as any other.
    /// When every segment is done, and only then, the folder goes.
    ///
    /// KILLED ANYWHERE, IT STARTS AGAIN AND ENDS THE SAME. A header is written before the flags
    /// that are read by it; a flag is one byte, written in place; a shot already numbered is never
    /// looked at again; the files are all still there until every flag is on disk; and the folder
    /// leaves in one rename. Nothing here touches the index: a path that no shot answers to any
    /// more is dropped by the next pass, like any file that is gone.
    ///
    /// `step` is called after each thing written, with what it was: a test kills the process there.
    @discardableResult
    public func migrateStills(step: ((String) -> Void)? = nil) -> Migration {
        writing.lock(); defer { writing.unlock() }
        return takeInStills(step: step)
    }

    private var triedStills = 0.0      // under `writing`: when `record` last tried, by the uptime clock

    /// Under `writing`.
    @discardableResult
    private func takeInStills(step: ((String) -> Void)?) -> Migration {
        let fm = FileManager.default
        try? fm.removeItem(at: removedStills)
        guard stillsPending else { return .none }
        // Flags are written into segments here: none is left open for a writer whose table of it
        // would then be wrong.
        closeWriters()

        // The files: each one's path below the folder, and its date.
        var files: [String: Double] = [:]
        for day in (try? fm.contentsOfDirectory(atPath: stills.path)) ?? [] {
            for name in (try? fm.contentsOfDirectory(atPath: stills.appendingPathComponent(day).path)) ?? []
            where name.hasSuffix(".heic") && !name.hasPrefix(".") {
                var st = stat()
                let rel = day + "/" + name
                guard stat(stills.appendingPathComponent(rel).path, &st) == 0 else { continue }
                files[rel] = Double(st.st_mtimespec.tv_sec) + Double(st.st_mtimespec.tv_nsec) / 1e9
            }
        }
        let dated = files.sorted { $0.value < $1.value }
        /// The files dated at a shot: to what a file system keeps of a date, not the tens of
        /// milliseconds between two displays of one look.
        func filesDated(at time: Double) -> ArraySlice<(key: String, value: Double)> {
            let tolerance = 0.002
            var lo = 0, hi = dated.count
            while lo < hi { let mid = (lo + hi) / 2; if dated[mid].value < time - tolerance { lo = mid + 1 } else { hi = mid } }
            var end = lo
            while end < dated.count, dated[end].value <= time + tolerance { end += 1 }
            return dated[lo ..< end]
        }
        /// The long side of a file, if it is a picture of a frame of this size: the frame fitted
        /// to that side is the file's size exactly. A file of another display's shot is not.
        func side(of rel: String, frame: Info) -> Int? {
            guard let source = CGImageSourceCreateWithURL(stills.appendingPathComponent(rel) as CFURL, nil),
                  let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
                  let w = properties[kCGImagePropertyPixelWidth] as? Int, let h = properties[kCGImagePropertyPixelHeight] as? Int,
                  max(w, h) > 0, max(w, h) <= Int(UInt16.max),
                  Self.fit(width: frame.width, height: frame.height, maxDimension: max(w, h)) == (w, h) else { return nil }
            return max(w, h)
        }

        // The segments: the paths already taken, and the shots that have a flag and no number.
        struct Old { let at: Int; let flags: UInt8; let time: Double; let app: String; let title: String }
        var taken = Set<String>()
        var pending: [(name: String, start: Double, info: Info, shots: [Old])] = []
        for ref in refs() {
            var old: [Old] = [], numbered: [(time: Double, number: Int, app: String, title: String)] = []
            guard let segment = Self.walk(history.appendingPathComponent(ref.name), each: { h in
                let n = Self.number(flags: h.flags)
                if n > 0 { numbered.append((h.time, n, h.app, h.title)) }
                else if h.flags & 2 != 0 { old.append(Old(at: h.at, flags: h.flags, time: h.time, app: h.app, title: h.title)) }
            }) else { continue }
            if segment.info.side > 0 {
                for s in numbered { taken.insert(Self.spelled(time: s.time, offset: segment.info.offset, app: s.app, title: s.title, number: s.number)) }
            }
            if !old.isEmpty { pending.append((ref.name, ref.start, segment.info, old)) }
        }

        var kept = 0, dropped = 0
        /// One pass over the segments still to do. `exact`: only the shots whose own file is found
        /// by its date; the rest wait for the second pass, when every such file has its shot.
        func pass(exact: Bool) -> String? {
            for i in pending.indices {
                let segment = pending[i]
                guard !segment.shots.isEmpty else { continue }
                guard let handle = try? FileHandle(forUpdating: history.appendingPathComponent(segment.name)) else {
                    return "\(segment.name) could not be opened."
                }
                defer { try? handle.close() }
                var info = segment.info
                do {
                    if info.side == 0 {
                        // The offset and the side this segment's files were written with.
                        var found: (offset: Int, side: Int)?
                        search: for shot in segment.shots {
                            if exact {
                                for file in filesDated(at: shot.time) {
                                    guard let wall = Self.wallSeconds((file.key as NSString).lastPathComponent) else { continue }
                                    let apart = wall - Self.second(shot.time)
                                    guard apart % 60 == 0, abs(apart) <= 18 * 3600, let side = side(of: file.key, frame: info) else { continue }
                                    found = (apart / 60, side)
                                    break search
                                }
                            } else {
                                // Files whose dates something has changed (a restore, a copy): by
                                // the plain name, in this Mac's time zone at that hour. The file
                                // may be another display's shot of that second; it is then one of
                                // a frame of the same size, and all that is read from it is the
                                // side a shot of this segment is pictured at.
                                let offset = Self.offset(at: shot.time)
                                let rel = Self.spelled(time: shot.time, offset: offset, app: shot.app, title: shot.title, number: 1)
                                if files[rel] != nil, let side = side(of: rel, frame: info) { found = (offset, side); break search }
                            }
                        }
                        if let found {
                            info = Info(offset: found.offset, side: found.side, width: info.width, height: info.height)
                            var packed = Data()
                            packed.append(le: Self.packed(info))
                            try handle.seek(toOffset: Self.infoOffset)
                            try handle.write(contentsOf: packed)
                            try handle.synchronize()
                            pending[i].info = info
                            step?("header")
                        }
                    }
                    var left: [Old] = []
                    for shot in segment.shots {
                        let names = info.side > 0 ? (1 ... Self.stillNumbers).map {
                            Self.spelled(time: shot.time, offset: info.offset, app: shot.app, title: shot.title, number: $0)
                        } : []
                        var number = 0
                        if let own = names.firstIndex(where: { name in
                            !taken.contains(name) && (files[name].map { abs($0 - shot.time) <= 0.002 } ?? false)
                        }) {
                            number = own + 1
                        } else if exact {
                            left.append(shot)
                            continue
                        } else if !filesDated(at: shot.time).isEmpty || names.contains(where: { files[$0] != nil && !taken.contains($0) }) {
                            // Its file is there and cannot be told from another's: under a name
                            // this segment does not spell (the hour a time zone changed in), or
                            // with its date lost (a restore, a copy), when which of the names of
                            // its second was its own is no longer known. Nothing is guessed: the
                            // shot takes a path no file ever had, and the index reads it again
                            // from its frame.
                            number = (names.firstIndex { !taken.contains($0) && files[$0] == nil } ?? -1) + 1
                        }
                        try handle.seek(toOffset: UInt64(shot.at) + 16)
                        try handle.write(contentsOf: Data([number > 0 ? Self.flags(isKey: shot.flags & 1 != 0, indexed: true, number: number) : shot.flags & 1]))
                        if number > 0 { taken.insert(names[number - 1]); kept += 1 } else { dropped += 1 }
                        step?("shot")
                    }
                    try handle.synchronize()
                    pending[i].shots = left
                } catch {
                    return error.localizedDescription
                }
                lock.withLock { tables[segment.name] = nil; infos[segment.name] = nil; tableChanges += 1 }
            }
            return nil
        }
        if let failure = pass(exact: true) ?? pass(exact: false) { return .failed(failure) }
        dropReaders()
        gates.removeAll()

        // Every flag is on disk. The folder leaves in one step, and is then deleted.
        guard rename(stills.path, removedStills.path) == 0 else { return .failed("The folder could not be moved.") }
        step?("moved")
        try? fm.removeItem(at: removedStills)
        step?("deleted")
        return .done(kept: kept, dropped: dropped)
    }

    // MARK: Retention

    /// Delete what is older than `cutoff`: whole segments. Returns the paths of the indexed shots
    /// that went with them, for the index to drop, or nil when nothing was old enough. (Killed
    /// before the index drops them, the next pass does: `indexedShots` no longer lists them.)
    @discardableResult
    public func prune(olderThan cutoff: Double) -> [String]? {
        writing.lock(); defer { writing.unlock() }
        let fm = FileManager.default
        removeStaleRewrites()
        var gone: [String] = []
        var removed = false
        let open = Set(writers.values.map(\.name))
        for ref in refs() where ref.start + Self.maxSegmentSeconds < cutoff && !open.contains(ref.name) {
            let url = history.appendingPathComponent(ref.name)
            let paths = (Self.list(url)?.shots ?? []).map { root + "/" + $0.rel }
            guard (try? fm.removeItem(at: url)) != nil else { continue }
            gone += paths
            removed = true
            lock.withLock { tables[ref.name] = nil; infos[ref.name] = nil; tableChanges += 1 }
        }
        if removed {
            lock.withLock { listing = nil }
            dropReaders()
        }
        return removed ? gone : nil
    }

    public struct Deleted: Sendable {
        /// Shots that left the history.
        public let shots: Int
        /// The paths of the indexed ones among them: the index drops these.
        public let paths: [String]
        /// Shots outside the span that went with it: ones that could not be encoded again.
        public let lost: Int
    }

    /// Delete the shots taken in a span of time, on one display or on all of them: their frames
    /// leave the history, and the caller drops their paths from the index.
    ///
    /// A frame is stored as what changed since the one before it, so the frames after a deleted one,
    /// up to the next key frame (at most `ScreenCodec.keyInterval`), are decoded and encoded again
    /// from a new key frame; everything else is copied as it is. The segment is written beside
    /// itself and renamed over, so a kill leaves the old segment or the new one. Killed before the
    /// caller has dropped the paths, the next pass drops them: `indexedShots` no longer lists them.
    @discardableResult
    public func delete(from: Double, to: Double, display: UInt32? = nil) -> Deleted {
        writing.lock(); defer { writing.unlock() }
        let fm = FileManager.default
        removeStaleRewrites()
        func doomed(_ f: Frame) -> Bool { f.shot.time >= from && f.shot.time <= to }
        let touched = refs().filter { ref in
            ref.start <= to && ref.start >= from - Self.maxSegmentSeconds - 1 && (display == nil || ref.display == display)
                && frames(of: ref.name).contains(where: doomed)
        }

        var shots = 0, lost = 0, removedFile = false
        var gone: [String] = []
        for ref in touched {
            let table = frames(of: ref.name)
            let before = Set(table.compactMap { path(of: $0.shot) })
            let url = history.appendingPathComponent(ref.name)
            if let (d, w) = writers.first(where: { $0.value.name == ref.name }) {
                try? w.handle.close()
                lock.withLock { _ = live.remove(w.name) }
                writers[d] = nil; lastStored[d] = nil; gates[d] = nil
            }
            let rewritten = rewrite(url, table: table, without: doomed)
            if rewritten != nil { lock.withLock { tables[ref.name] = nil; tableChanges += 1 } }
            dropReaders()
            guard let rewritten else { continue }       // the segment is as it was: nothing of it is counted
            shots += table.count - rewritten.kept - rewritten.lost
            lost += rewritten.lost
            // What the index held of this segment and the segment no longer has: the deleted shots,
            // and any that could not be encoded again.
            if rewritten.kept == 0 {
                removedFile = true
                lock.withLock { infos[ref.name] = nil }
                gone += before
            } else {
                gone += before.subtracting((Self.list(url)?.shots ?? []).map { root + "/" + $0.rel })
            }
        }
        if removedFile { lock.withLock { listing = nil } }
        // What the index held of a display is what its gate remembers: read again from what is left.
        if !gone.isEmpty { gates.removeAll() }
        return Deleted(shots: shots, paths: gone.sorted(), lost: lost)
    }

    /// A segment without some of its frames, written beside it and renamed over it; the file is
    /// removed when no frame is left. nil when it could not be written, and the segment is then
    /// untouched. Called under `writing`.
    private func rewrite(_ url: URL, table: [Frame], without doomed: (Frame) -> Bool) -> (kept: Int, lost: Int)? {
        let fm = FileManager.default
        guard table.contains(where: { !doomed($0) }) else {
            return (try? fm.removeItem(at: url)) != nil ? (0, 0) : nil
        }
        guard let old = try? Data(contentsOf: url, options: .alwaysMapped), old.count >= Self.headerLength else { return nil }
        let temp = url.appendingPathExtension(Self.rewriteExtension)
        guard fm.createFile(atPath: temp.path, contents: old.prefix(Self.headerLength)),
              let out = try? FileHandle(forWritingTo: temp) else { return nil }
        _ = try? out.seekToEnd()
        var kept = 0, lost = 0
        var encoder: ScreenFrameEncoder?
        var broken = false          // a frame this one was encoded against is gone or encoded again
        var fromEncoder = false     // the frame before this one in the new file came from `encoder`
        do {
            for f in table {
                if doomed(f) { broken = true; continue }
                if f.isKey { broken = false }
                if !broken {
                    try out.write(contentsOf: old.sub(Int(f.record), Int(f.payload) + f.length + 4 - Int(f.record)))
                    kept += 1; fromEncoder = false
                    continue
                }
                guard let pixels = pixels(f.shot) else { lost += 1; continue }
                if encoder == nil { encoder = ScreenFrameEncoder(width: CVPixelBufferGetWidth(pixels), height: CVPixelBufferGetHeight(pixels)) }
                guard let encoded = encoder?.encode(pixels, forceKey: !fromEncoder), fromEncoder || encoded.isKey else {
                    // Without this frame the ones after it have nothing to decode from either.
                    lost += 1; fromEncoder = false
                    continue
                }
                let (record, _) = Self.record(time: f.shot.time, isKey: encoded.isKey, indexed: f.shot.indexed, number: f.shot.number,
                                              app: f.shot.app, title: f.shot.title, sets: encoded.parameterSets, payload: encoded.data)
                try out.write(contentsOf: record)
                kept += 1; fromEncoder = true
            }
            try out.synchronize()
            try out.close()
        } catch {
            try? out.close()
            try? fm.removeItem(at: temp)
            return nil
        }
        if kept == 0 {
            try? fm.removeItem(at: temp)
            return (try? fm.removeItem(at: url)) != nil ? (0, lost) : nil
        }
        guard rename(temp.path, url.path) == 0 else { try? fm.removeItem(at: temp); return nil }
        return (kept, lost)
    }

    private static let rewriteExtension = "rewrite"

    /// A rewrite a kill cut short: the segment it was replacing is whole, the copy is not.
    private func removeStaleRewrites() {
        let fm = FileManager.default
        for name in (try? fm.contentsOfDirectory(atPath: history.path)) ?? [] where name.hasSuffix("." + Self.rewriteExtension) {
            try? fm.removeItem(at: history.appendingPathComponent(name))
        }
    }

    /// Delete everything: every segment, the catalog, and 0.20.2's files if they were still there.
    /// The caller drops everything under `root` from the index.
    public func clear() {
        writing.lock(); defer { writing.unlock() }
        closeWriters()
        let fm = FileManager.default
        cataloging.withLock {
            for item in (try? fm.contentsOfDirectory(at: history, includingPropertiesForKeys: nil)) ?? [] { try? fm.removeItem(at: item) }
        }
        try? fm.removeItem(at: stills)
        lock.withLock { tables.removeAll(); infos.removeAll(); listing = nil; tableChanges += 1 }
        dropReaders()       // after the files went: none is held open, deleted and still on the disk
    }

    /// Bytes on disk.
    public func diskUse() -> Int {
        var total = 0
        for dir in [history, stills] {
            guard let walk = FileManager.default.enumerator(at: dir, includingPropertiesForKeys: [.fileSizeKey]) else { continue }
            for case let url as URL in walk { total += (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0 }
        }
        return total
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
