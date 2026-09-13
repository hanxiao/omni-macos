import Foundation

/// Is a user-facing request on the GPU right now?
///
/// The app has THREE things that submit MLX work and only one of them is arbitrated: the engine's
/// priority gate orders search against indexing, `VectorStore`'s scan takes no gate at all, and the
/// OCR model runs in a lane of its own. That is tolerable because MLX submits through one Metal
/// command queue per process, so the lanes serialise anyway - what the gate really controls is who
/// submits next and HOW BIG each submission is.
///
/// This is the missing piece between the lanes: a process-wide count of interactive requests in
/// flight, raised around the whole of a search (the embed AND the scan) and read by the OCR decode
/// loop. It is deliberately NOT the engine's `interactiveQueryActive`, which stays true for two
/// seconds after a query so the indexer can keep its batches small - a two-second window is right
/// for choosing a batch size and completely wrong for blocking a decode step.
///
/// Measured before this existed: a search over HTTP during an OCR run ran p50 49.3 ms against 14.9
/// idle, and the FIRST request - the one landing while a page prefills - took 1656 ms against 239.
public enum GPUInteractive {
    private static let cond = NSCondition()
    /// `nonisolated(unsafe)` because the safety is the condition variable, not the compiler: every
    /// read and write below happens under `cond`. The alternative spellings do not fit - an actor
    /// cannot be consulted from the decode loop without an await, and `Mutex` would still need the
    /// condition for the wait.
    private nonisolated(unsafe) static var inFlight = 0

    public static func enter() {
        cond.lock(); inFlight += 1; cond.unlock()
    }

    public static func leave() {
        cond.lock()
        inFlight -= 1
        if inFlight <= 0 { inFlight = 0; cond.broadcast() }
        cond.unlock()
    }

    public static func around<T>(_ work: () throws -> T) rethrows -> T {
        enter(); defer { leave() }
        return try work()
    }

    public static var isBusy: Bool {
        cond.lock(); defer { cond.unlock() }
        return inFlight > 0
    }

    /// Block the caller while interactive work holds the GPU, up to `timeout`.
    ///
    /// ALWAYS BOUNDED. This is called from the OCR decode loop, which must keep making progress
    /// whatever happens on the other lane - a request that never lowers the count (a bug, a
    /// cancelled task) would otherwise stop a transcription for good. The cap costs at most one
    /// extra decode step of latency per stall.
    public static func yieldWhileBusy(timeout: TimeInterval = 0.25) {
        cond.lock()
        if inFlight > 0 {
            let deadline = Date().addingTimeInterval(timeout)
            while inFlight > 0, Date() < deadline { cond.wait(until: deadline) }
        }
        cond.unlock()
    }
}
