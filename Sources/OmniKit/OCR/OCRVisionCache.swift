import Foundation
import MLX

/// Visual features, keyed by the pixels that produced them.
///
/// The vision tower is 280 ms of a page's ~653 ms prefill, and the resample ahead of it another
/// 55 ms. Neither depends on the PROMPT: the prompt reaches the model as token ids, and the
/// pixels never appear in them. So the two halves of `preparePage` split cleanly, and a page the
/// workspace has already looked at can be transcribed again - with an edited prompt, after a stop,
/// or because the same file was dropped twice - for the cost of the language model alone.
///
/// This is the same observation vllm-mlx makes (content-hash the image, reuse the embedding, 28x
/// on repeats). It is worth nothing on a first pass over distinct pages, which is why it is a
/// cache and not a pipeline stage.
///
/// The key is 128 bits over the whole pixel buffer, not 64. A collision here would hand one page's
/// visual features to another page's transcription and there is nothing downstream that would
/// notice, so the cost of a second hash is worth paying to make that not happen.
final class OCRVisionCache: @unchecked Sendable {

    struct Entry {
        let visual: MLXArray
        let grid: (w: Int, h: Int)
        let bytes: Int
    }

    private struct Key: Hashable {
        let a: UInt64
        let b: UInt64
        let width: Int
        let height: Int
    }

    private var store: [Key: Entry] = [:]
    private var order: [Key] = []           // least recently used first
    private var bytes = 0
    private let limit: Int
    private let lock = NSLock()

    private(set) var hits = 0
    private(set) var misses = 0

    /// 256 MB holds about 58 pages at the 4.4 MB a page of this scan costs. The bound matters
    /// more than the number: a run over a long document would otherwise keep every page it has
    /// ever seen, and `OCRBatchPlan` charges this to the reserve so the batch width leaves room
    /// for it rather than discovering it.
    init(limitBytes: Int = OCRBatchPlan.visualCacheBytes) {
        self.limit = limitBytes
    }

    // MARK: - key

    private static func hashes(_ image: OCRImage) -> (UInt64, UInt64) {
        // Two FNV-1a passes with different offset bases, over the same bytes in one walk. 4 MB of
        // pixels at 8 bytes a step is a fraction of a millisecond against the 335 ms it guards.
        var a: UInt64 = 0xcbf2_9ce4_8422_2325
        var b: UInt64 = 0x9e37_79b9_7f4a_7c15
        image.rgb.withUnsafeBytes { raw in
            let words = raw.bindMemory(to: UInt64.self)
            for w in words {
                a = (a ^ w) &* 0x1000_0000_01b3
                b = (b ^ w) &* 0x8894_7609_1197_1e2b
            }
            // the tail the word loop could not reach
            let done = words.count * 8
            for i in done ..< raw.count {
                a = (a ^ UInt64(raw[i])) &* 0x1000_0000_01b3
                b = (b ^ UInt64(raw[i])) &* 0x8894_7609_1197_1e2b
            }
        }
        return (a, b)
    }

    private static func key(_ image: OCRImage) -> Key {
        let (a, b) = hashes(image)
        return Key(a: a, b: b, width: image.width, height: image.height)
    }

    // MARK: - access

    func lookup(_ image: OCRImage) -> Entry? {
        let k = Self.key(image)
        lock.lock()
        defer { lock.unlock() }
        guard let entry = store[k] else { misses += 1; return nil }
        hits += 1
        if let at = order.firstIndex(of: k) { order.remove(at: at); order.append(k) }
        return entry
    }

    func insert(_ image: OCRImage, visual: MLXArray, grid: (w: Int, h: Int)) {
        let size = visual.size * visual.dtype.size
        let k = Self.key(image)
        lock.lock()
        defer { lock.unlock() }
        if let existing = store[k] {
            bytes -= existing.bytes
            if let at = order.firstIndex(of: k) { order.remove(at: at) }
        }
        store[k] = Entry(visual: visual, grid: grid, bytes: size)
        order.append(k)
        bytes += size
        while bytes > limit, let oldest = order.first {
            order.removeFirst()
            if let gone = store.removeValue(forKey: oldest) { bytes -= gone.bytes }
        }
    }

    func clear() {
        lock.lock()
        defer { lock.unlock() }
        store.removeAll()
        order.removeAll()
        bytes = 0
    }

    var report: String {
        lock.lock()
        defer { lock.unlock() }
        return "vision cache: \(hits) hits, \(misses) misses, \(store.count) pages, "
            + String(format: "%.0f MB", Double(bytes) / 1_048_576)
    }
}
