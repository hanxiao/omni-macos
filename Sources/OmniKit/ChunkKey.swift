import Foundation
import CryptoKit

/// THE IDENTITY OF ONE CHUNK'S CONTENT.
///
/// Two chunks share a key exactly when they would produce the same vector, so a key is the content
/// plus everything that decides what the embedder does with it. Get that wrong in either direction
/// and the failure is silent: too loose and a stale vector is served for text that changed, too
/// strict and 3.5M duplicate vectors are embedded again for no reason.
///
/// GENERATION IS PART OF THE KEY. The leading field names the cutter, so chunks cut by the fixed
/// grid and chunks cut by `ContentChunker` occupy disjoint key spaces. That is what lets one index
/// hold both while it migrates: a generation-1 chunk can never be mistaken for a generation-2 chunk
/// of the same bytes, because the same bytes under two cutters are genuinely different chunks with
/// different neighbours. It is also what makes the migration lazy - a file re-cut under generation 2
/// drops its old pointers, the orphaned chunks fall to `refs = 0`, and their slots return to the
/// free list with no migration pass for the vectors at all.
public enum ChunkKey {

    /// 128 bits, the same width and the same reasoning as `StoreSchema.contentKeyDigest`: a
    /// collision over a few million items is around 1e-27, and a collision is not silent corruption
    /// anyway, because the chunk the key resolves to is re-checked before its vector is reused.
    private static func digest(_ prefix: String, _ payload: Data) -> String {
        var h = SHA256()
        h.update(data: Data(prefix.utf8))
        h.update(data: payload)
        return h.finalize().prefix(16).map { String(format: "%02x", $0) }.joined()
    }

    // MARK: - Generation 1: the fixed grid

    /// THE v4 FORMAT, REPRODUCED EXACTLY. Do not "tidy" this string.
    ///
    /// The whole migration rests on it. Every one of the 9,130,536 text chunks in an existing index
    /// carries a key computed by this formula, and the migration reuses each of those vectors by
    /// looking the key up rather than embedding anything. One byte different here and every lookup
    /// misses, the migration silently re-embeds the entire corpus, and the thing that was supposed
    /// to cost no GPU costs days of it. `ChunkKeyTests` writes the formula out a second time and
    /// compares, so a change has to be made in two places on purpose.
    public static func grid(_ text: String, maxChars: Int, overlap: Int, dim: Int) -> String {
        digest("1|c\(maxChars)|o\(overlap)|m\(dim)|", Data(text.utf8))
    }

    // MARK: - Generation 2: content-defined

    /// A text chunk cut by `ContentChunker`. The cutter's fingerprint names every parameter that
    /// moves a boundary, so changing the target size re-cuts rather than silently mixing chunks cut
    /// under different rules into one key space.
    public static func text(_ text: String, dim: Int) -> String {
        digest("2|\(ContentChunker.fingerprint)|m\(dim)|", Data(text.utf8))
    }

    // MARK: - Media

    /// A media chunk: an image, one rendered page of a scan, one 240 s audio segment, one 240 s
    /// video segment. `payload` is the bytes the tower actually sees - decoded pixels, the mel, the
    /// sampled frames - not the file on disk, because two files that decode to the same pixels
    /// should share a vector and two files with identical bytes are already caught one level up by
    /// the per-file content key.
    ///
    /// HONEST SCOPE: for a plain image a chunk IS the whole file, so this mostly restates what the
    /// file-level dedup already knows. It earns its place on repeated PDF pages, on the same clip
    /// embedded in two videos, and on having one code path instead of two.
    ///
    /// `preprocess` must name every setting that changes those bytes (decode size, frame count,
    /// segment length), for the same reason the text keys carry the cutter.
    public static func media(kind: FileKind, payload: Data, preprocess: String, dim: Int) -> String {
        digest("2|\(kind.rawValue)|\(preprocess)|m\(dim)|", payload)
    }

    /// A media chunk keyed by the VECTOR IT STORES.
    ///
    /// v4 gives image, scan, video and audio no content key at all, so they never share - a page
    /// that appears in two PDFs is stored twice. The obvious key is the decoded payload, and it is
    /// the better one because it can be computed BEFORE the tower runs and so skips the GPU as
    /// well. It is also not reachable from here: the payload is gone by the time a chunk is built,
    /// and threading it out to eight construction sites is a larger change than this is worth.
    ///
    /// Keying on the stored bytes deduplicates exactly what is stored, which is the honest smaller
    /// claim: identical vectors share a slot, and the forward pass still happens. The bf16 row is
    /// what the store writes, so two chunks share a key exactly when they would share a slot.
    public static func mediaVector(kind: FileKind, bf16 row: [UInt16], dim: Int) -> String {
        var h = SHA256()
        h.update(data: Data("2v|\(kind.rawValue)|m\(dim)|".utf8))
        row.withUnsafeBufferPointer { p in
            h.update(data: Data(bytes: p.baseAddress!, count: p.count * MemoryLayout<UInt16>.size))
        }
        return h.finalize().prefix(16).map { String(format: "%02x", $0) }.joined()
    }

    /// Hash of a large payload the caller already has in slices, so a 240 s mel or a page of pixels
    /// need not be copied into one Data first.
    public static func media(kind: FileKind, slices: [Data], preprocess: String, dim: Int) -> String {
        var h = SHA256()
        h.update(data: Data("2|\(kind.rawValue)|\(preprocess)|m\(dim)|".utf8))
        for s in slices { h.update(data: s) }
        return h.finalize().prefix(16).map { String(format: "%02x", $0) }.joined()
    }
}
