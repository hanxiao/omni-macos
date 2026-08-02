import Foundation

// The A/B levers, in one place.
//
// Every lever below is a process-wide static. That is what makes an in-app paper run possible at
// all (setenv after first touch is either a silent no-op or a permanent change to the live app,
// and spawning omni-verify would load a second copy of the model), and it is also what makes it
// dangerous: a concurrent search would run against the USER's store with a benchmark's lever
// settings. So mutation is confined to this file and fenced by three rules, all enforced here
// rather than by convention:
//
//   1. Owner thread. The controller captures the thread that created it and traps on a mutation
//      from anywhere else. The suite is strictly serial on one detached thread, so any mutation
//      from a queue, a completion handler or the main actor is a bug and shows up as a crash in
//      testing rather than as a quietly wrong table.
//   2. No nesting. An arm scope cannot open inside another arm scope; the restore order of two
//      overlapping scopes is exactly the mistake that leaves a lever flipped.
//   3. Always restored. Every scope restores in a `defer`, and `restore()` re-captures the live
//      statics and asserts they match what was captured before the run.
//
// Callers are additionally responsible for the "no work in flight" half of the contract: live
// indexing paused and awaited, the FS watcher stopped, serving refused, and MLX work evaluated
// before an arm scope ends. The suite does the first three; a case body that leaves an unevaluated
// MLXArray behind at the end of an arm scope has measured the wrong arm.

/// The base representation of the scan matrix. Distinct from `Int?` on purpose: "leave the ship
/// policy alone" and "force the ship policy's own auto choice" are different instructions, and the
/// auto choice is cap-class dependent, so it differs between an 8 GB and a 16 GB machine.
public enum PaperQuantBase: Sendable, Codable, Equatable {
    case auto
    case bits(Int)
}

/// A partial assignment: nil means "leave whatever is pinned". Codable so an arm can be stamped in
/// the export as exactly the set of levers it moved.
public struct PaperLeverSet: Sendable, Codable, Equatable {
    public var adaptiveBatch: Bool?
    public var tailRows: Bool?
    public var chunkCache: Bool?
    public var contentDedup: Bool?
    public var cantWinGate: Bool?
    public var gemvSlice: Bool?
    public var vacuumSmallCache: Bool?
    public var idleFold: Bool?
    public var proactiveFold: Bool?
    public var lexical: Bool?
    public var quantBase: PaperQuantBase?

    public init(adaptiveBatch: Bool? = nil, tailRows: Bool? = nil, chunkCache: Bool? = nil,
                contentDedup: Bool? = nil, cantWinGate: Bool? = nil, gemvSlice: Bool? = nil,
                vacuumSmallCache: Bool? = nil, idleFold: Bool? = nil, proactiveFold: Bool? = nil,
                lexical: Bool? = nil, quantBase: PaperQuantBase? = nil) {
        self.adaptiveBatch = adaptiveBatch; self.tailRows = tailRows; self.chunkCache = chunkCache
        self.contentDedup = contentDedup; self.cantWinGate = cantWinGate; self.gemvSlice = gemvSlice
        self.vacuumSmallCache = vacuumSmallCache; self.idleFold = idleFold
        self.proactiveFold = proactiveFold; self.lexical = lexical; self.quantBase = quantBase
    }

    /// The suite-wide pins from the plan's settings table. Held for the whole run and restored with
    /// everything else: dedup on (the corpus is generated with repeated paragraphs on purpose),
    /// gemv slicing on (a correctness lever, never an arm), the filename channel off (it is a
    /// corpus statistic and would perturb every search timing), and the quant base left on the ship
    /// policy until a case forces it.
    public static let suiteWide = PaperLeverSet(contentDedup: true, gemvSlice: true,
                                                idleFold: true, proactiveFold: true,
                                                lexical: false, quantBase: .auto)

    /// The levers this set actually moves, as `name=value`, for stamping an arm in the export.
    public var stamped: [String: String] {
        var out: [String: String] = [:]
        func put(_ k: String, _ v: Bool?) { if let v { out[k] = v ? "on" : "off" } }
        put("adaptive_batch", adaptiveBatch); put("tail_rows", tailRows)
        put("chunk_cache", chunkCache); put("content_dedup", contentDedup)
        put("cantwin_gate", cantWinGate); put("gemv_slice", gemvSlice)
        put("vacuum_small_cache", vacuumSmallCache); put("idle_fold", idleFold)
        put("proactive_fold", proactiveFold); put("lexical", lexical)
        if let q = quantBase { out["quant_base"] = { if case .bits(let b) = q { "\(b)" } else { "auto" } }() }
        return out
    }
}

/// A full snapshot of the live statics. Captured before the run, restored after it, and compared
/// against the live values at the end so a missed restore is loud instead of silent.
public struct PaperLevers: Sendable, Equatable {
    public var adaptiveBatch: Bool
    public var tailRows: Bool
    public var chunkCache: Bool
    public var contentDedup: Bool
    public var cantWinGate: Bool
    public var gemvSlice: Bool
    public var vacuumSmallCache: Bool
    public var idleFold: Bool
    public var proactiveFold: Bool
    public var lexical: Bool
    public var quantBase: PaperQuantBase

    public static func capture() -> PaperLevers {
        PaperLevers(
            adaptiveBatch: OmniEngine.adaptiveBatch,
            tailRows: Qwen3Backbone.tailRowsEnabled,
            chunkCache: Indexer.chunkCache,
            contentDedup: Indexer.contentDedup,
            cantWinGate: VectorStore.cantWinGate,
            gemvSlice: VectorStore.gemvSlice,
            vacuumSmallCache: VectorStore.vacuumSmallCache,
            idleFold: VectorStore.idleFold,
            proactiveFold: VectorStore.proactiveFold,
            lexical: LexicalIndex.enabled,
            quantBase: VectorStore.quantBaseOverride.map { PaperQuantBase.bits($0) } ?? .auto
        )
    }

    public func apply() {
        OmniEngine.adaptiveBatch = adaptiveBatch
        Qwen3Backbone.tailRowsEnabled = tailRows
        Indexer.chunkCache = chunkCache
        Indexer.contentDedup = contentDedup
        VectorStore.cantWinGate = cantWinGate
        VectorStore.gemvSlice = gemvSlice
        VectorStore.vacuumSmallCache = vacuumSmallCache
        VectorStore.idleFold = idleFold
        VectorStore.proactiveFold = proactiveFold
        LexicalIndex.enabled = lexical
        VectorStore.quantBaseOverride = { if case .bits(let b) = quantBase { b } else { nil } }()
    }

    public func applying(_ set: PaperLeverSet) -> PaperLevers {
        var out = self
        if let v = set.adaptiveBatch { out.adaptiveBatch = v }
        if let v = set.tailRows { out.tailRows = v }
        if let v = set.chunkCache { out.chunkCache = v }
        if let v = set.contentDedup { out.contentDedup = v }
        if let v = set.cantWinGate { out.cantWinGate = v }
        if let v = set.gemvSlice { out.gemvSlice = v }
        if let v = set.vacuumSmallCache { out.vacuumSmallCache = v }
        if let v = set.idleFold { out.idleFold = v }
        if let v = set.proactiveFold { out.proactiveFold = v }
        if let v = set.lexical { out.lexical = v }
        if let v = set.quantBase { out.quantBase = v }
        return out
    }

    /// Everything, as `name=value`, for `pin.*` in the export.
    public var stamped: [String: String] {
        PaperLeverSet(adaptiveBatch: adaptiveBatch, tailRows: tailRows, chunkCache: chunkCache,
                      contentDedup: contentDedup, cantWinGate: cantWinGate, gemvSlice: gemvSlice,
                      vacuumSmallCache: vacuumSmallCache, idleFold: idleFold,
                      proactiveFold: proactiveFold, lexical: lexical, quantBase: quantBase).stamped
    }
}

/// Owns every mutation of the levers for the duration of a run.
public final class PaperLeverController: @unchecked Sendable {
    /// Settle gap after a lever moves, before the arm's work starts. The suite is serial so nothing
    /// should be in flight, but MLX evaluates lazily and the store folds on its own timers; a short
    /// fence costs a quarter second per arm and removes a whole class of "the previous arm's work
    /// finished under the new lever" doubt.
    public let settleSeconds: TimeInterval

    private let owner: Thread
    private let original: PaperLevers
    private var pinned: PaperLevers
    private var activeArmName: String?

    public init(settleSeconds: TimeInterval = 0.25) {
        self.settleSeconds = settleSeconds
        self.owner = Thread.current
        self.original = PaperLevers.capture()
        self.pinned = original
    }

    public var levers: PaperLevers { pinned }
    public var activeArm: String? { activeArmName }

    /// Suite-wide pin. Legal only between cases, with no arm scope open.
    public func pin(_ set: PaperLeverSet) {
        assertOwner()
        precondition(activeArmName == nil, "paper levers pinned while arm '\(activeArmName ?? "")' was open")
        pinned = pinned.applying(set)
        pinned.apply()
    }

    /// Enter an arm: apply `set` on top of the pinned values, settle, run, restore the pinned values
    /// in a `defer` that fires on throw and on cancel alike.
    public func withArm<T>(_ name: String, _ set: PaperLeverSet, _ body: () throws -> T) rethrows -> T {
        assertOwner()
        precondition(activeArmName == nil,
                     "paper arm '\(name)' opened inside arm '\(activeArmName ?? "")': arm scopes do not nest")
        activeArmName = name
        pinned.applying(set).apply()
        defer {
            pinned.apply()
            activeArmName = nil
        }
        if settleSeconds > 0 { Thread.sleep(forTimeInterval: settleSeconds) }
        return try body()
    }

    /// Put every lever back exactly as it was before the run. Called from the suite's `defer`, so it
    /// runs on success, on a throw and on a cancel.
    public func restore() {
        assertOwner()
        original.apply()
        activeArmName = nil
        pinned = original
        // A missed lever here means the live app keeps a benchmark's setting until relaunch, which
        // is the worst outcome this module can produce. Fail loudly in debug rather than ship it.
        assert(PaperLevers.capture() == original, "paper levers did not restore to their pre-run values")
    }

    private func assertOwner() {
        precondition(Thread.current === owner,
                     "paper levers mutated off the suite thread: the levers are process-wide and the "
                     + "suite is strictly serial, so a mutation from another thread races live work")
    }
}
