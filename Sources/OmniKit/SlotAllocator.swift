import Foundation

/// WHO OWNS WHICH ROW OF THE VECTOR FILE.
///
/// A chunk's id IS its slot in `.vecs`, so allocating an id and allocating 1536 bytes of vector are
/// the same act. v4 derived a slot from a row's rank in rowid order counted through a hole list, and
/// the note beside that code says what is wrong with it: from the first hole onward the
/// correspondence is recoverable only while the two structures move in lockstep, and
/// "unobservably false" afterwards. An explicit id cannot drift.
///
/// WHY THIS DOES NOT BRING BACK THE COMPACTION PROBLEM. v4 rejected a slot column because
/// "compaction renumbers everything. Per row that is a 4.5M-row UPDATE, measured at 33.8s, which
/// cannot run inside close()." That is an argument against RECLAIMING holes by renumbering, and a
/// free list reclaims them without renumbering anything: a released slot is handed to the next new
/// chunk. The file grows only when live chunks exceed the high-water mark, and the 96,256 holes v4
/// accumulated against 4.53M live rows stop accumulating.
///
/// THE QUARANTINE IS THE PART THAT IS EASY TO GET WRONG. A slot released in one transaction must
/// not be handed out in that same transaction. A search in flight may already hold that id in a
/// candidate list, and if a different chunk's vector has been written into the slot underneath it,
/// the search scores the wrong content and reports it under the wrong file. Nothing crashes. So
/// releases land in `quarantine` and only `commit()` - called at a stamp, after the readers that
/// could be holding stale ids are done - moves them where `allocate` can see them.
public struct SlotAllocator: Sendable {

    /// Slots below `highWater` that no chunk owns and that are safe to hand out now.
    private(set) var available: [Int] = []
    /// Released during the current transaction. Not allocatable until `commit()`.
    private(set) var quarantine: [Int] = []
    /// One past the highest slot ever allocated. The vector file is this many rows long.
    private(set) var highWater: Int = 0

    public init() {}

    /// Restore from what SQLite holds: the free list and the high-water mark.
    public init(available: [Int], highWater: Int) {
        self.available = available.filter { $0 >= 0 && $0 < highWater }
        self.highWater = Swift.max(0, highWater)
    }

    /// The next slot to write a vector into. Prefers a hole, extends the file only when there is
    /// none. Lowest first, so the file stays as dense at the front as it can and a truncation after
    /// a large delete has a chance of being worth doing.
    public mutating func allocate() -> Int {
        if let i = available.indices.min(by: { available[$0] < available[$1] }) {
            return available.remove(at: i)
        }
        let id = highWater
        highWater += 1
        return id
    }

    /// Give a slot back. It becomes allocatable at the next `commit()`, never before.
    public mutating func release(_ id: Int) {
        guard id >= 0, id < highWater else { return }
        quarantine.append(id)
    }

    /// End of transaction: quarantined slots become allocatable.
    public mutating func commit() {
        guard !quarantine.isEmpty else { return }
        available.append(contentsOf: quarantine)
        quarantine.removeAll(keepingCapacity: true)
    }

    /// Transaction rolled back: the releases never happened, so the slots are still owned.
    public mutating func rollback() {
        quarantine.removeAll(keepingCapacity: true)
    }

    /// Slots that are neither live nor free. A leak is invisible - the vector file simply never
    /// shrinks - so it is checked rather than assumed.
    public func leaked(liveIDs: Set<Int>) -> [Int] {
        let free = Set(available).union(quarantine)
        return (0 ..< highWater).filter { !liveIDs.contains($0) && !free.contains($0) }
    }

    /// Rebuild the free list from the chunk table alone, which is the authority. The free list is a
    /// cache of "ids below the high-water mark that no chunk row owns", so it is always recoverable
    /// and never the thing that has to survive a crash - the same property that lets v4's
    /// `vec_holes` be rebuilt from SQLite.
    public static func reconciled(liveIDs: Set<Int>, highWater: Int) -> SlotAllocator {
        let hw = Swift.max(0, highWater)
        return SlotAllocator(available: (0 ..< hw).filter { !liveIDs.contains($0) }, highWater: hw)
    }

    /// Live slots plus free slots must exactly cover the file. Anything else means a slot is owned
    /// twice or by nobody.
    public func covers(liveIDs: Set<Int>) -> Bool {
        let free = Set(available).union(quarantine)
        guard free.isDisjoint(with: liveIDs) else { return false }
        return free.count + liveIDs.count == highWater
    }
}
