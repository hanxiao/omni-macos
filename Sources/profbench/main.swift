import Foundation
import OmniKit

// Headless equivalent of the App menu's "Run benchmark": the same runProfilingPass, the same
// IndexSettings.profiling, the same fixed 300-file dataset, into the same kind of fresh temporary
// store. Exists as its OWN target so it can be injected into a worktree at any release tag without
// touching main.swift - whose contents differ between tags and would not compile.
//
// usage: profbench <modelDir> <datasetFolder> [repeats]
let args = CommandLine.arguments
guard args.count >= 3 else {
    FileHandle.standardError.write(Data("usage: profbench <modelDir> <dataset> [repeats]\n".utf8))
    exit(2)
}
let repeats = args.count >= 4 ? (Int(args[3]) ?? 1) : 1
let engine = try await OmniEngine(modelDir: URL(fileURLWithPath: args[1]))
let folder = URL(fileURLWithPath: args[2])

for r in 1 ... repeats {
    let m = try await runProfilingPass(engine: engine, targetURL: folder,
                                       settings: .profiling, shouldCancel: nil) { _ in }
    print(String(format: "profbench run=%d  files=%d  seconds=%.2f  filesPerSec=%.3f  tokensPerSec=%.0f  peakVramMB=%.0f",
                 r, m.files, m.seconds, m.filesPerSec, m.tokensPerSec,
                 Double(m.peakVramDeltaBytes) / 1_048_576))
    fflush(stdout)
}
exit(0)
