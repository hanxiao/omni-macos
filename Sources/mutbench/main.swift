import Foundation
import OmniKit

// What a MUTATION costs on a real index, as its own target so it can run at any release tag.
// The removals that have no per-file index behind them (folder, kind) walk every row to find their
// victims, and that walk now also has to record the vector slots each victim leaves behind - so it
// is the one mutation shape whose cost scales with the whole index rather than with the edit.
//
// usage: mutbench <dbPath> <folderThatMatchesNothing> [reps]
let args = CommandLine.arguments
guard args.count >= 3 else { print("usage: mutbench <dbPath> <folder> [reps]"); exit(2) }
let url = URL(fileURLWithPath: args[1])
let folder = args[2]
let reps = args.count >= 4 ? (Int(args[3]) ?? 5) : 5
omniSetMemoryLimit(6_000_000_000)
let store = try VectorStore(dbURL: url)
print("mutbench rows=\(store.count)")
var ms: [Double] = []
for _ in 0 ..< reps {
    let t = Date()
    store.deleteUnderFolder(folder)
    ms.append(-t.timeIntervalSinceNow * 1000)
}
ms.sort()
print(String(format: "  deleteUnderFolder (no match)  p50 %6.1f ms  min %6.1f  max %6.1f", ms[ms.count / 2], ms[0], ms[ms.count - 1]))
var hs: [Double] = []
for _ in 0 ..< reps {
    let t = Date()
    _ = store.hasRowsUnder(folder)
    hs.append(-t.timeIntervalSinceNow * 1000)
}
hs.sort()
print(String(format: "  hasRowsUnder (no match)       p50 %6.2f ms", hs[hs.count / 2]))
store.close()
