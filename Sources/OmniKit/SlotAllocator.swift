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

    /// Slots below `highWater` that no chunk owns and that are safe to hand out now, held as a
    /// BINARY MIN-HEAP rather than a plain array.
    ///
    /// The obvious form - an array plus a linear `min` in `allocate` - is O(n) per allocation and
    /// therefore quadratic over a reindex. At the 96,256 holes a real index carries after days of
    /// churn that is not a slow path, it is a hang, and it would only ever appear at the scale
    /// where the free list starts to matter. The heap makes `allocate` O(log n) and `commit` O(k
    /// log n) in what was actually released, with no per-mutation pass over the whole list.
    private var heap: [Int] = []
    /// The free slots, in no particular order. Reporting and tests; the order is the heap's.
    public var available: [Int] { heap }
    /// Released during the current transaction. Not allocatable until `commit()`.
    private(set) var quarantine: [Int] = []
    /// Free slots that a sharer has taken since they were freed. `allocate` drops them instead of
    /// handing them out; see `claim`.
    private var claimed: Set<Int> = []
    /// One past the highest slot ever allocated. The vector file is this many rows long.
    private(set) var highWater: Int = 0

    public init() {}

    /// Restore from what SQLite holds: the free list and the high-water mark.
    public init(available: [Int], highWater: Int) {
        self.highWater = Swift.max(0, highWater)
        heap = available.filter { $0 >= 0 && $0 < self.highWater }
        heapify()
    }

    /// The next slot to write a vector into. Prefers a hole, extends the file only when there is
    /// none. Lowest first, so the file stays as dense at the front as it can and a truncation after
    /// a large delete has a chance of being worth doing.
    public mutating func allocate() -> Int {
        // Skip anything a sharer took directly (see `claim`). Lazy deletion, because removing an
        // arbitrary value from a binary heap is O(n) and this runs per written chunk.
        while let id = popMin() {
            if claimed.remove(id) != nil { continue }
            return id
        }
        let id = highWater
        highWater += 1
        return id
    }

    /// A SLOT TAKEN WITHOUT ASKING. A chunk whose content already exists is seated on that
    /// content's existing position rather than allocated one - it is the same vector, so there is
    /// nothing to write and nothing to allocate. But if that position was ALSO sitting in the free
    /// list, because the content's last occurrence had been deleted earlier, the allocator would
    /// hand the same position to a different content later and two contents would share one
    /// vector: one of them scoring, and being returned, as the other.
    ///
    /// So a sharer says so. Cheap and idempotent: claiming something that was never free is a
    /// no-op, and the entry is dropped when `allocate` next walks past it.
    public mutating func claim(_ id: Int) {
        guard id >= 0 else { return }
        if id >= highWater { highWater = id + 1; return }   // never was in the free set
        if let i = quarantine.firstIndex(of: id) { quarantine.remove(at: i) }
        claimed.insert(id)
    }

    /// Give a slot back. It becomes allocatable at the next `commit()`, never before.
    public mutating func release(_ id: Int) {
        guard id >= 0, id < highWater else { return }
        quarantine.append(id)
    }

    /// End of transaction: quarantined slots become allocatable.
    public mutating func commit() {
        guard !quarantine.isEmpty else { return }
        for id in quarantine where !claimed.contains(id) { push(id) }
        quarantine.removeAll(keepingCapacity: true)
    }

    // MARK: - Heap

    private mutating func heapify() {
        guard heap.count > 1 else { return }
        for i in stride(from: heap.count / 2 - 1, through: 0, by: -1) { siftDown(i) }
    }

    private mutating func push(_ id: Int) {
        heap.append(id)
        var i = heap.count - 1
        while i > 0 {
            let p = (i - 1) / 2
            guard heap[p] > heap[i] else { break }
            heap.swapAt(p, i); i = p
        }
    }

    private mutating func popMin() -> Int? {
        guard let first = heap.first else { return nil }
        if heap.count == 1 { heap.removeLast(); return first }
        heap[0] = heap.removeLast()
        siftDown(0)
        return first
    }

    private mutating func siftDown(_ from: Int) {
        var i = from
        let n = heap.count
        while true {
            let l = 2 * i + 1, r = l + 1
            var m = i
            if l < n, heap[l] < heap[m] { m = l }
            if r < n, heap[r] < heap[m] { m = r }
            if m == i { return }
            heap.swapAt(i, m); i = m
        }
    }

    /// Transaction rolled back: the releases never happened, so the slots are still owned.
    public mutating func rollback() {
        quarantine.removeAll(keepingCapacity: true)
    }

    /// Hand every free slot back, keeping the high-water mark. Used where the positions moved under
    /// the allocator - a compaction renumbers them, so the cached list describes a numbering that
    /// no longer exists and has to be rebuilt from the table rather than adjusted.
    /// The file grew, and nothing was freed by it. Raising the ceiling is the whole update: the new
    /// positions belong to the rows that appended them, so the free set is unchanged.
    ///
    /// Without this, "is the allocator current" was `highWater == positions`, which is false after
    /// EVERY append - so under churn each allocation rebuilt the whole free set from every row.
    /// Measured on a 4,000-file churn: 587 operations against 914 with the free list off, a third
    /// of the throughput spent rediscovering a set that had not changed.
    public mutating func raiseHighWater(to n: Int) {
        guard n > highWater else { return }
        highWater = n
    }

    public mutating func forgetFreeList() {
        heap.removeAll(keepingCapacity: true)
        quarantine.removeAll(keepingCapacity: true)
        claimed.removeAll(keepingCapacity: true)
    }

    /// Slots that are neither live nor free. A leak is invisible - the vector file simply never
    /// shrinks - so it is checked rather than assumed.
    public func leaked(liveIDs: Set<Int>) -> [Int] {
        let free = Set(heap).union(quarantine)
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
        let free = Set(heap).union(quarantine)
        guard free.isDisjoint(with: liveIDs) else { return false }
        return free.count + liveIDs.count == highWater
    }
}
