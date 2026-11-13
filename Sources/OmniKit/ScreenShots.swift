import Foundation
import CoreGraphics
import os

/// THE SCREEN HISTORY AS AN INGESTION CHANNEL.
///
/// An indexed shot has a path in the index and no file behind it (`ScreenHistory.path(of:)`): its
/// picture is a frame of the video, taken out when it is asked for. Everything the indexer asks a
/// file system about a file - is it there, how big, give me its picture - it asks the history about
/// a shot, here. A history is registered once by whoever owns it (the app at launch, a test for its
/// own scratch folder); the indexer finds it by the path, as it finds a Photos asset by its scheme.
public enum ScreenShots {
    private static let state = OSAllocatedUnfairLock(initialState: [(prefix: String, history: ScreenHistory)]())

    /// Registers a history: paths under its root are shots from now on. Once per root.
    public static func add(_ history: ScreenHistory) {
        let prefix = history.root + "/"
        state.withLock { if !$0.contains(where: { $0.prefix == prefix }) { $0.append((prefix, history)) } }
    }

    public static func remove(_ history: ScreenHistory) {
        state.withLock { $0.removeAll { $0.history === history } }
    }

    /// The history whose shots live under `path`.
    public static func history(holding path: String) -> ScreenHistory? {
        state.withLock { $0.first { path.hasPrefix($0.prefix) }?.history }
    }

    /// The history whose root this is, by the path the index holds or the folder 0.20.2 kept there.
    public static func history(root path: String) -> ScreenHistory? {
        state.withLock { $0.first { $0.history.root == path || $0.history.stills.path == path }?.history }
    }

    /// Is this the folder the shots' paths are under, or a day's folder in it (a shot's path is
    /// "<root>/<day>/<name>")? Neither is on disk, and a search scoped to one is a search of its
    /// shots all the same.
    public static func isFolder(_ path: String) -> Bool {
        state.withLock {
            $0.contains { path == $0.history.root || path.hasPrefix($0.prefix) && !path.dropFirst($0.prefix.count).contains("/") }
        }
    }

    /// Is this a shot's path? A lock and a prefix compare: called per row.
    public static func isShotPath(_ path: String) -> Bool {
        state.withLock { $0.contains { path.hasPrefix($0.prefix) } }
    }

    /// What the history says of a shot's path. `unknown` while 0.20.2's picture files are still to
    /// be taken in (`ScreenHistory.stillsPending`): until then a path that finds no shot has not
    /// lost one, and nothing may be removed on its word.
    enum Presence { case held(ScreenHistory.Entry), gone, unknown }

    static func presence(_ path: String) -> Presence {
        guard let history = history(holding: path), !history.stillsPending else { return .unknown }
        return history.entry(atPath: path).map(Presence.held) ?? .gone
    }

    /// Has the history let go of the shot this path names?
    public static func isGone(_ path: String) -> Bool {
        if case .gone = presence(path) { return true }
        return false
    }

    /// A shot as the pipeline carries a file: `modified` its time, `size` the bytes of its frame.
    ///
    /// A SHOT NEVER CHANGES, so one the index already holds is the version it holds: `known`'s date
    /// and size are taken. 0.20.2 indexed each from a picture file, and its row carries that file's
    /// date and size, which no frame has; compared with the frame's, every shot of an updated index
    /// would read as changed and go through the model again.
    static func file(_ entry: ScreenHistory.Entry, known: StoredFile?) -> CrawledFile {
        CrawledFile(path: entry.path, modified: known?.modified ?? entry.time, size: known?.size ?? entry.bytes)
    }

    /// The text of a shot, read as 0.20.2 read its picture file: the frame at a pixel a point, then
    /// at the reader's size, in pieces (measured against the file's reading on seven desktops:
    /// docs/screenshots.md, "The frame out of the video against the still"). nil: no such shot.
    static func text(atPath path: String) -> String? {
        guard let history = history(holding: path), let shot = history.shot(atPath: path),
              let picture = history.picture(of: shot)?.image,
              let read = ScreenText.fitted(picture, long: ScreenText.readSide(width: picture.width, height: picture.height, dpi: nil))
        else { return nil }
        return ScreenText.recognize(pieces: read)
    }
}

/// Shots ride the pipeline as pictures. The path ends in ".heic" because 0.20.2's files did, and
/// the index's rows were kept as they were; nothing here reads a file.
struct ScreenContentSource: ContentSource {
    let history: ScreenHistory

    static func claims(_ file: CrawledFile) -> Bool { ScreenShots.isShotPath(file.path) }

    /// The size the index holds a shot at: the desktop at a pixel a point. nil for a shot the
    /// history no longer holds - not a deletion (ContentSource.probe); the pass's sweep and the
    /// calls that removed it drop its rows.
    func probe(_ file: CrawledFile, settings: IndexSettings) -> SourceProbe? {
        guard let shot = history.shot(atPath: file.path), let size = history.size(of: shot) else { return nil }
        return SourceProbe(kind: .image, width: size.width, height: size.height, measured: true)
    }

    /// Keyed on the shot, as a Photos asset is on the asset: the frame is not a run of bytes to
    /// hash, and two shots are never the same picture (the recorder stores a screen once).
    func contentKey(_ file: CrawledFile, kind: FileKind, dim: Int,
                    chunkOverlap: Int, settings: IndexSettings) -> String? {
        guard kind == .image, let shot = history.shot(atPath: file.path) else { return nil }
        return "2|shot|m\(dim)|t\(shot.time.bitPattern)|n\(shot.display)|d\(settings.maxImageDimension)"
            + Indexer.screenTextMark(file, settings: settings)
    }

    func content(_ file: CrawledFile, kind: FileKind, settings: IndexSettings) -> ExtractedContent {
        // Not read is not empty: only a file with nothing in it leaves the index (ExtractedContent).
        guard kind == .image, let shot = history.shot(atPath: file.path), let size = history.size(of: shot) else { return .unreadable }
        let side = max(size.width, size.height)
        let cap = settings.maxImageDimension > 0 ? min(side, settings.maxImageDimension) : side
        guard let picture = history.picture(of: shot, maxDimension: cap) else { return .unreadable }
        return .images([picture.image])
    }

    func video(_ file: CrawledFile, probe: SourceProbe, settings: IndexSettings) -> SourceDecode? { nil }
    func audio(_ file: CrawledFile, probe: SourceProbe, settings: IndexSettings) -> SourceDecode? { nil }
}
