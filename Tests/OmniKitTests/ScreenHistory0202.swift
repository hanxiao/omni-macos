import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
@testable import OmniKit

/// WHAT 0.20.2 LEFT ON A MAC, made again for the tests of the update that takes it in: a picture
/// file for every indexed shot, and segments whose headers and flags say nothing of a path.
///
/// The naming and the writing are 0.20.2's own code (ScreenHistory.stillName, writeStill), kept
/// here word for word: they are the oracle the history's spelling of a path is held to, so nothing
/// in them is taken from the code under test.
enum ScreenHistory0202 {
    struct Failed: Error { let what: String }

    /// The day's folder and the file's name without its extension, from a shot's time and the app
    /// and title its record holds.
    static func name(time: Double, app: String, title: String) -> (folder: String, base: String) {
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

    /// Writes a shot's picture file as 0.20.2 did: the first of the name and its numbered twins
    /// that no file has, through a hidden temporary file, marked as a screen capture and dated at
    /// the shot. Returns the file.
    @discardableResult
    static func writeStill(_ image: CGImage, in stills: URL, time: Double, app: String, title: String) throws -> URL {
        let fm = FileManager.default
        let name = name(time: time, app: app, title: title)
        let folder = stills.appendingPathComponent(name.folder, isDirectory: true)
        try fm.createDirectory(at: folder, withIntermediateDirectories: true)
        var url = folder.appendingPathComponent(name.base + ".heic")
        var n = 2
        while fm.fileExists(atPath: url.path) {
            url = folder.appendingPathComponent("\(name.base) \(n).heic")
            n += 1
        }
        let temp = folder.appendingPathComponent("." + url.lastPathComponent + ".tmp")
        guard let dest = CGImageDestinationCreateWithURL(temp as CFURL, UTType.heic.identifier as CFString, 1, nil) else {
            throw Failed(what: "no destination")
        }
        CGImageDestinationAddImage(dest, image, ScreenCodec.stillOptions(quality: 0.8))
        guard CGImageDestinationFinalize(dest) else { throw Failed(what: "not written") }
        let flag = try PropertyListSerialization.data(fromPropertyList: true, format: .binary, options: 0)
        _ = flag.withUnsafeBytes { setxattr(temp.path, "com.apple.metadata:kMDItemIsScreenCapture", $0.baseAddress, flag.count, 0, 0) }
        let date = Date(timeIntervalSince1970: time)
        try fm.setAttributes([.creationDate: date, .modificationDate: date], ofItemAtPath: temp.path)
        try fm.moveItem(at: temp, to: url)
        return url
    }

    /// Makes the segments in `folder` what 0.20.2 wrote: the header's last word zero, and of a
    /// record's flags only "key frame" and "indexed". The list kept beside them goes too.
    static func strip(_ folder: URL) throws {
        let fm = FileManager.default
        for name in try fm.contentsOfDirectory(atPath: folder.path) where name.hasSuffix(".omniscreen") {
            let url = folder.appendingPathComponent(name)
            var data = try Data(contentsOf: url)
            guard data.count >= 24 else { continue }
            for i in 20 ..< 24 { data[i] = 0 }
            var at = 24
            while at + 26 <= data.count {
                let mark = data.subdata(in: at ..< at + 4).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }
                let length = Int(data.subdata(in: at + 4 ..< at + 8).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) })
                guard UInt32(littleEndian: mark) == 0x544F_4853, at + 8 + length + 4 <= data.count else { break }
                data[at + 16] &= 3
                at += 8 + length + 4
            }
            let dates = try fm.attributesOfItem(atPath: url.path)
            try data.write(to: url)
            if let modified = dates[.modificationDate] { try fm.setAttributes([.modificationDate: modified], ofItemAtPath: url.path) }
        }
        try? fm.removeItem(at: folder.appendingPathComponent("shots.catalog"))
    }

    /// Takes a history recorded by this build back to what 0.20.2 would have left of the same
    /// recording: a picture file for each indexed shot, written in the order they were taken, and
    /// stripped segments. The history is stopped; open it again to read it.
    /// Returns each file's path below the folder, in the order written, beside its shot.
    @discardableResult
    static func downgrade(_ history: ScreenHistory, skipping: (ScreenHistory.Shot) -> Bool = { _ in false }) throws -> [(rel: String, shot: ScreenHistory.Shot)] {
        let shots = history.shots(from: 0, to: 4e9)
            .filter(\.indexed).sorted { ($0.time, $0.display) < ($1.time, $1.display) }
        var pictures: [(ScreenHistory.Shot, CGImage)] = []
        for shot in shots {
            guard let image = history.picture(of: shot)?.image else { throw Failed(what: "no picture at \(shot.time)") }
            pictures.append((shot, image))
        }
        history.stop()
        var written: [(String, ScreenHistory.Shot)] = []
        // The folder is there from the first indexed shot on, files or none.
        try FileManager.default.createDirectory(at: history.stills, withIntermediateDirectories: true)
        for (shot, image) in pictures where !skipping(shot) {
            let url = try writeStill(image, in: history.stills, time: shot.time, app: shot.app, title: shot.title)
            written.append((url.deletingLastPathComponent().lastPathComponent + "/" + url.lastPathComponent, shot))
        }
        try strip(history.history)
        return written
    }

    /// Every picture file below `stills`, as "<day>/<name>", in the bytes the folder lists it
    /// under: read with readdir, which is what a crawl put in the index.
    static func listed(_ stills: URL) -> [String] {
        func names(_ path: String) -> [String] {
            guard let d = opendir(path) else { return [] }
            defer { closedir(d) }
            var out: [String] = []
            while let e = readdir(d) {
                let name = withUnsafePointer(to: &e.pointee.d_name) {
                    $0.withMemoryRebound(to: CChar.self, capacity: Int(e.pointee.d_namlen) + 1) { String(cString: $0) }
                }
                if !name.hasPrefix(".") { out.append(name) }
            }
            return out
        }
        var out: [String] = []
        for day in names(stills.path) {
            for file in names(stills.path + "/" + day) where file.hasSuffix(".heic") { out.append(day + "/" + file) }
        }
        return out.sorted { Array($0.utf8).lexicographicallyPrecedes(Array($1.utf8)) }
    }
}
