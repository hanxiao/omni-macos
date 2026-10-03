import XCTest
@testable import OmniKit

/// Matching cost of a real policy over real paths. Runs only with OMNI_IGNORE_BENCH=<dir> holding
/// `policy.txt` and `paths.txt` (one path a line, `D` or `F` first for folder or file). Prints the
/// decisions' counts, which must not move when the matcher is optimised.
final class IgnoreBenchTests: XCTestCase {
    func testRealPolicyOverRealPaths() throws {
        guard let dir = ProcessInfo.processInfo.environment["OMNI_IGNORE_BENCH"] else { throw XCTSkip("set OMNI_IGNORE_BENCH") }
        let policy = try String(contentsOfFile: dir + "/policy.txt", encoding: .utf8)
        let lines = try String(contentsOfFile: dir + "/paths.txt", encoding: .utf8).split(separator: "\n")
        let entries = lines.map { (isDir: $0.first == "D", path: String($0.dropFirst())) }
        let bases = (ProcessInfo.processInfo.environment["OMNI_IGNORE_BASES"] ?? "").split(separator: ":").map(String.init)
        var parsed = OmniIgnore(text: "")
        let tp = Date()
        for _ in 0 ..< 100 { parsed = OmniIgnore(text: policy, bases: bases) }
        let parseUs = -tp.timeIntervalSinceNow / 100 * 1e6
        func time(_ label: String, _ body: () -> Int) {
            var best = Double.infinity, n = 0
            for _ in 0 ..< 3 { let t = Date(); n = body(); best = min(best, -t.timeIntervalSinceNow) }
            print(String(format: "BENCH %-22@ %8d excluded  %7.0f ns/path  (%d paths)", label as NSString, n, best / Double(entries.count) * 1e9, entries.count))
        }
        print(String(format: "BENCH parse %.1f us for %d rules", parseUs, policy.split(separator: "\n").count))
        time("lowercase+Character[]") { entries.reduce(0) { $0 + Array($1.path.lowercased()).count % 2 } }
        time("lowercase+scalars") { entries.reduce(0) { $0 + Array($1.path.lowercased().unicodeScalars).count % 2 } }
        time("utf8 bytes") { entries.reduce(0) { $0 + Array($1.path.utf8).count % 2 } }
        time("empty policy") { let e = OmniIgnore(text: ""); return entries.reduce(0) { $0 + (e.isIgnored($1.path, isDir: $1.isDir) ? 1 : 0) } }
        time("one name rule") { let e = OmniIgnore(text: "node_modules/"); return entries.reduce(0) { $0 + (e.isIgnored($1.path, isDir: $1.isDir) ? 1 : 0) } }
        time("isIgnored") { entries.reduce(0) { $0 + (parsed.isIgnored($1.path, isDir: $1.isDir) ? 1 : 0) } }
        time("includingAncestors") {
            entries.reduce(0) { acc, e in
                let root = bases.first { e.path.hasPrefix($0 + "/") }
                return acc + (parsed.isIgnoredIncludingAncestors(e.path, isDir: e.isDir, root: root) ? 1 : 0)
            }
        }
        let ex = parsed.excludesIndexedFile(roots: bases)
        time("excludesIndexedFile") { entries.reduce(0) { $0 + (!$1.isDir && ex($1.path) ? 1 : 0) } }
    }
}
