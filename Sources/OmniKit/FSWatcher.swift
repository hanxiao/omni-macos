import Foundation
import CoreServices

/// Watches a set of folders with FSEvents and reports changed file paths (coalesced), with the ones
/// flagged as renamed: both halves of a move inside the watched folders carry that flag and arrive
/// in one callback, while a plain delete never does (measured: `rm -rf` reports removed, a move out
/// of the tree reports only the old path, renamed).
/// Persisting `lastEventId` lets a relaunch replay changes missed while the app was closed.
public final class FSWatcher: @unchecked Sendable {
    private var stream: FSEventStreamRef?
    private let dispatchQueue = DispatchQueue(label: "omni.fswatch")
    private let paths: [String]
    private let sinceWhen: FSEventStreamEventId
    /// @Sendable: it runs on the watcher's queue. Without it a closure written inside a @MainActor
    /// type is inferred main-actor isolated, and Swift 6 traps on the first call from this queue.
    private let onChange: @Sendable (_ paths: [String], _ flags: Flags) -> Void

    /// What the flags of one callback say beyond the paths.
    public struct Flags: Sendable {
        /// Flagged renamed: both halves of a move inside the watched folders, one callback.
        public var renamed = Set<String>()
        /// Events not delivered one by one - see init.
        public var rescan = Set<String>()
        /// Directories whose every event in the callback was metadata only (permissions, owner,
        /// times, Finder info): never created or renamed, so nothing arrived in them that the
        /// files' own events do not report. Measured: chmod reports `D o`, touch `D i o`, while
        /// mkdir, a move in and a copy in all carry Created or Renamed. Crawling one of these
        /// re-read its whole subtree for nothing - a root's own metadata walked every file.
        public var metadataOnlyDirs = Set<String>()
    }

    /// `rescan`: paths whose events were not delivered one by one - FSEvents coalesced or dropped
    /// them (MustScanSubDirs, UserDropped, KernelDropped) or the watched root itself changed
    /// (RootChanged: deleted, moved, a volume unmounted or mounted again). Whatever was deleted
    /// under such a path produced no event of its own, so only a walk of it can find out.
    public init(paths: [String], since: UInt64? = nil,
                onChange: @escaping @Sendable (_ paths: [String], _ flags: Flags) -> Void) {
        self.paths = paths
        self.sinceWhen = since.map { FSEventStreamEventId($0) } ?? FSEventStreamEventId(kFSEventStreamEventIdSinceNow)
        self.onChange = onChange
    }

    public func start() {
        guard stream == nil, !paths.isEmpty else { return }
        // The stream must RETAIN the watcher: FSEventStreamInvalidate does not wait for a callback
        // already executing on the dispatch queue, so an unretained `info` could be used after the
        // owner drops the watcher mid-burst (add/remove folder while files are changing). With
        // retain/release callbacks the stream holds +1 until it is invalidated AND the last in-flight
        // callback returns; stop() must therefore always be called explicitly (restartWatcher does),
        // or the stream would keep the watcher alive.
        var context = FSEventStreamContext(
            version: 0,
            info: Unmanaged.passUnretained(self).toOpaque(),
            retain: { ptr in
                guard let ptr else { return nil }
                _ = Unmanaged<FSWatcher>.fromOpaque(ptr).retain()
                return UnsafeRawPointer(ptr)
            },
            release: { ptr in
                guard let ptr else { return }
                Unmanaged<FSWatcher>.fromOpaque(ptr).release()
            },
            copyDescription: nil)
        let flags = UInt32(kFSEventStreamCreateFlagFileEvents | kFSEventStreamCreateFlagNoDefer | kFSEventStreamCreateFlagUseCFTypes | kFSEventStreamCreateFlagWatchRoot)
        guard let s = FSEventStreamCreate(
            kCFAllocatorDefault, fsEventsCallback, &context,
            paths as CFArray, sinceWhen, 1.5 /* latency: coalesce bursts */, flags)
        else { return }
        FSEventStreamSetDispatchQueue(s, dispatchQueue)
        FSEventStreamStart(s)
        stream = s
    }

    public func stop() {
        guard let s = stream else { return }
        FSEventStreamStop(s)
        FSEventStreamInvalidate(s)
        FSEventStreamRelease(s)
        stream = nil
    }

    /// The latest event id seen; persist this to resume across launches.
    public func latestEventId() -> UInt64 {
        UInt64(stream.map { FSEventStreamGetLatestEventId($0) } ?? sinceWhen)
    }

    deinit { stop() }

    fileprivate func handle(_ paths: [String], flags: Flags) { onChange(paths, flags) }
}

private func fsEventsCallback(
    stream: ConstFSEventStreamRef,
    info: UnsafeMutableRawPointer?,
    numEvents: Int,
    eventPaths: UnsafeMutableRawPointer,
    eventFlags: UnsafePointer<FSEventStreamEventFlags>,
    eventIds: UnsafePointer<FSEventStreamEventId>
) {
    guard let info else { return }
    let watcher = Unmanaged<FSWatcher>.fromOpaque(info).takeUnretainedValue()
    let paths = (unsafeBitCast(eventPaths, to: NSArray.self) as? [String]) ?? []
    var out = FSWatcher.Flags()
    let walk = UInt32(kFSEventStreamEventFlagMustScanSubDirs | kFSEventStreamEventFlagUserDropped
                      | kFSEventStreamEventFlagKernelDropped | kFSEventStreamEventFlagRootChanged)
    let arrived = UInt32(kFSEventStreamEventFlagItemCreated | kFSEventStreamEventFlagItemRenamed
                         | kFSEventStreamEventFlagItemRemoved) | walk
    var dirFlags: [String: UInt32] = [:]   // a path can appear more than once in a callback
    for i in 0 ..< min(numEvents, paths.count) {
        let f = eventFlags[i]
        if f & UInt32(kFSEventStreamEventFlagItemRenamed) != 0 { out.renamed.insert(paths[i]) }
        if f & walk != 0 { out.rescan.insert(paths[i]) }
        if f & UInt32(kFSEventStreamEventFlagItemIsDir) != 0 { dirFlags[paths[i], default: 0] |= f }
    }
    for (p, f) in dirFlags where f & arrived == 0 { out.metadataOnlyDirs.insert(p) }
    watcher.handle(paths, flags: out)
}
