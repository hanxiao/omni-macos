import Foundation

/// Moving the index to another folder, checked before anything is touched.
///
/// This lives here, apart from the UI, because it decides whether someone's index survives. The
/// Settings button used to just write the new path into the preferences and re-open: a folder with
/// no write access or no room left the app with an empty index at the new location, the old one
/// stranded with nothing in the app pointing at it, and millions of files reindexing from scratch.
/// On a real library that is 21 GB abandoned and hours of GPU time.
public enum IndexRelocation {

    /// Every file the index is made of, in a given folder.
    ///
    /// The sqlite database and its sidecars: `.vecs` (the vectors, irreplaceable), `.quant` and
    /// `.rows` (the scan tier), `.names` (the filename index), and the WAL/SHM companions of both
    /// databases. `.vecs.new` is deliberately excluded - it is a compaction temp and means nothing
    /// outside the compaction that wrote it.
    public static let fileNames = [
        "index.sqlite", "index.sqlite-wal", "index.sqlite-shm",
        "index.sqlite.names", "index.sqlite.names-wal", "index.sqlite.names-shm",
        "index.sqlite.vecs", "index.sqlite.quant", "index.sqlite.rows",
    ]

    public static func files(in dir: URL) -> [URL] {
        let fm = FileManager.default
        return fileNames.map { dir.appendingPathComponent($0) }.filter { fm.fileExists(atPath: $0.path) }
    }

    public static func byteSize(of urls: [URL]) -> Int64 {
        urls.reduce(0) { sum, u in
            sum + (((try? FileManager.default.attributesOfItem(atPath: u.path))?[.size] as? NSNumber)?
                    .int64Value ?? 0)
        }
    }

    /// Why `dst` cannot hold the index, or nil when it can. `freeBytes` is injectable so the space
    /// rule can be tested without filling a disk.
    public static func refusal(from src: URL, to dst: URL, payload: Int64,
                               freeBytes: ((URL) -> Int64?)? = nil) -> String? {
        let s = src.standardizedFileURL.path, d = dst.standardizedFileURL.path
        if s == d { return "That is already the index folder." }
        // Nesting either way is refused: copying a folder into itself duplicates forever, and
        // putting the index in a parent of its current home makes the old copy a child of the new
        // one, where a later cleanup would take both.
        if d.hasPrefix(s + "/") { return "That folder is inside the current index folder. Choose one outside it." }
        if s.hasPrefix(d + "/") { return "That folder contains the current index folder. Choose a different one." }

        let fm = FileManager.default
        do { try fm.createDirectory(at: dst, withIntermediateDirectories: true) }
        catch { return "That folder could not be created: \(error.localizedDescription)" }

        // Writability proved BY WRITING. `isWritableFile(atPath:)` answers from the POSIX bits and
        // says yes on volumes that then refuse the write - a read-only mount, a full disk, a
        // sandbox denial - which is exactly the case this check exists to catch.
        let probe = dst.appendingPathComponent(".omni-write-probe")
        do { try Data([0]).write(to: probe); try? fm.removeItem(at: probe) }
        catch { return "Omni cannot write to that folder: \(error.localizedDescription)" }

        let free = freeBytes?(dst)
            ?? (try? fm.attributesOfFileSystem(forPath: dst.path)[.systemFreeSize] as? NSNumber)??.int64Value
        if let free, free < payload + payload / 20 {          // payload plus 5% headroom
            return "That volume has \(bytes(free)) free, and the index needs \(bytes(payload)). "
                 + "Free some space or choose another disk."
        }
        return nil
    }

    /// Copy every index file into `dst`, verifying each one's size. Throws with the destination
    /// cleaned up, so a failure leaves the source untouched and nothing half-written behind.
    public static func copy(from src: URL, to dst: URL) throws {
        let fm = FileManager.default
        var done: [URL] = []
        do {
            for f in files(in: src) {
                let dest = dst.appendingPathComponent(f.lastPathComponent)
                if fm.fileExists(atPath: dest.path) { try fm.removeItem(at: dest) }
                try fm.copyItem(at: f, to: dest)
                let a = (try? fm.attributesOfItem(atPath: f.path))?[.size] as? NSNumber
                let b = (try? fm.attributesOfItem(atPath: dest.path))?[.size] as? NSNumber
                guard a?.int64Value == b?.int64Value else {
                    throw NSError(domain: "omni.relocate", code: 1, userInfo: [
                        NSLocalizedDescriptionKey: "\(f.lastPathComponent) did not copy completely."])
                }
                done.append(dest)
            }
        } catch {
            for d in done { try? fm.removeItem(at: d) }
            throw error
        }
    }

    private static func bytes(_ n: Int64) -> String {
        let f = ByteCountFormatter()
        f.countStyle = .file
        f.allowsNonnumericFormatting = false
        return f.string(fromByteCount: n)
    }
}
