import Foundation
import CoreServices

/// Watches a set of folders with FSEvents and reports changed file paths (coalesced).
/// Persisting `lastEventId` lets a relaunch replay changes missed while the app was closed.
public final class FSWatcher: @unchecked Sendable {
    private var stream: FSEventStreamRef?
    private let dispatchQueue = DispatchQueue(label: "omni.fswatch")
    private let paths: [String]
    private let sinceWhen: FSEventStreamEventId
    private let onChange: ([String]) -> Void

    public init(paths: [String], since: UInt64? = nil, onChange: @escaping ([String]) -> Void) {
        self.paths = paths
        self.sinceWhen = since.map { FSEventStreamEventId($0) } ?? FSEventStreamEventId(kFSEventStreamEventIdSinceNow)
        self.onChange = onChange
    }

    public func start() {
        guard stream == nil, !paths.isEmpty else { return }
        var context = FSEventStreamContext(
            version: 0,
            info: Unmanaged.passUnretained(self).toOpaque(),
            retain: nil, release: nil, copyDescription: nil)
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

    fileprivate func handle(_ paths: [String]) { onChange(paths) }
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
    watcher.handle(paths)
}
