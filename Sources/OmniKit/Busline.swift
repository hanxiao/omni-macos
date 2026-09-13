import Foundation

/// What a queue is actually doing, as numbers.
///
/// Every lane in this process is a bare `DispatchQueue`, which is invisible: it has no depth, no
/// backlog, and no record of work that ran after it stopped mattering. Crowding therefore only ever
/// surfaced as a FEELING - "switching folders gets slower the more you click" - and three separate
/// regressions shipped because nothing could assert otherwise.
///
/// Worse, the obvious assertion does not work. A saturation test that measures DURATION passes
/// against deliberately broken code, because a synthetic fixture makes the slow operation
/// microseconds and there is nothing to queue behind: verified three times over, with the lane
/// split, the supersede logic, and the browse listing. The number that survives a small fixture is
/// `wasted` - work that ran after becoming pointless - because it counts events, not time.
///
/// Cost is one `os_unfair_lock` per operation, which is why this is safe to leave on in shipping
/// builds. `peakDepth` and `wasted` are the two a test should assert on.
public final class Busline: @unchecked Sendable {
    public struct Reading: Sendable, Equatable {
        public var enqueued: Int
        public var completed: Int
        public var depth: Int
        public var peakDepth: Int
        /// Work that ran to completion after it had already been superseded. The number that turns
        /// "this feels slow when I click fast" into a failing assertion.
        public var wasted: Int
    }

    public let name: String
    private var lock = os_unfair_lock_s()
    private var enqueued = 0
    private var completed = 0
    private var depth = 0
    private var peakDepth = 0
    private var wasted = 0

    public init(name: String) { self.name = name }

    /// Around a unit of work on the lane. `enter` is the ENQUEUE, not the start, which is the whole
    /// point: depth is what is waiting, and a lane whose depth climbs under load is crowded no
    /// matter how fast each individual item is.
    @inline(__always) public func enter() {
        os_unfair_lock_lock(&lock)
        enqueued += 1
        depth += 1
        if depth > peakDepth { peakDepth = depth }
        os_unfair_lock_unlock(&lock)
    }

    @inline(__always) public func leave(wasted didWaste: Bool = false) {
        os_unfair_lock_lock(&lock)
        completed += 1
        depth -= 1
        if depth < 0 { depth = 0 }
        if didWaste { wasted += 1 }
        os_unfair_lock_unlock(&lock)
    }

    /// Record that a unit of work finished having become pointless, without it being a whole
    /// enter/leave pair - for lanes that discover the supersede after admitting the work.
    @inline(__always) public func noteWasted() {
        os_unfair_lock_lock(&lock)
        wasted += 1
        os_unfair_lock_unlock(&lock)
    }

    public var reading: Reading {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        return Reading(enqueued: enqueued, completed: completed, depth: depth,
                       peakDepth: peakDepth, wasted: wasted)
    }

    /// Zero the high-water marks so a test can measure one episode rather than the process. Depth
    /// is NOT reset - it describes work currently in flight, which a reset would falsify.
    public func resetPeaks() {
        os_unfair_lock_lock(&lock)
        peakDepth = depth
        wasted = 0
        enqueued = 0
        completed = 0
        os_unfair_lock_unlock(&lock)
    }
}
