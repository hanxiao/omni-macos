import Foundation
import Metal

/// Hardware facts for a profiling report. Hardware-only, no PII: no hostname, username, serial,
/// or file paths. All fields are read with public APIs (the app runs unsandboxed).
public struct HardwareProfile: Sendable, Codable {
    public let chip: String           // "Apple M3 Ultra" (machdep.cpu.brand_string)
    public let hwModel: String        // "Mac15,13" (hw.model)
    public let releaseYear: Int?      // best-effort, from the chip family
    public let macosVersion: String   // "15.2.0"
    public let memoryBytes: Int       // total unified memory
    public let vramBytes: Int?        // Metal recommendedMaxWorkingSetSize
    public let cpuCores: Int
    public let diskInternal: Bool?    // is the boot volume internal storage
    public let diskFileSystem: String? // "apfs"

    public static func collect() -> HardwareProfile {
        HardwareProfile(
            chip: sysctlString("machdep.cpu.brand_string") ?? "Unknown",
            hwModel: sysctlString("hw.model") ?? "Unknown",
            releaseYear: releaseYear(for: sysctlString("machdep.cpu.brand_string") ?? ""),
            macosVersion: osVersionString(),
            memoryBytes: Int(ProcessInfo.processInfo.physicalMemory),
            vramBytes: MTLCreateSystemDefaultDevice().map { Int($0.recommendedMaxWorkingSetSize) },
            cpuCores: ProcessInfo.processInfo.activeProcessorCount,
            diskInternal: bootVolumeInternal(),
            diskFileSystem: bootVolumeFileSystem()
        )
    }

    private static func sysctlString(_ name: String) -> String? {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
        var buf = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &buf, &size, nil, 0) == 0 else { return nil }
        return String(cString: buf).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func osVersionString() -> String {
        let v = ProcessInfo.processInfo.operatingSystemVersion
        return "\(v.majorVersion).\(v.minorVersion).\(v.patchVersion)"
    }

    /// Best-effort Apple-silicon release year by chip family. Ordered most-specific first so
    /// "M3 Ultra" matches before "M3". nil when the chip is unrecognized (e.g. a future M-series).
    static func releaseYear(for chip: String) -> Int? {
        let table: [(String, Int)] = [
            ("M1 Ultra", 2022), ("M1 Pro", 2021), ("M1 Max", 2021), ("M1", 2020),
            ("M2 Ultra", 2023), ("M2 Pro", 2023), ("M2 Max", 2023), ("M2", 2022),
            ("M3 Ultra", 2025), ("M3 Pro", 2023), ("M3 Max", 2023), ("M3", 2023),
            ("M4 Ultra", 2025), ("M4 Pro", 2024), ("M4 Max", 2024), ("M4", 2024),
            ("M5", 2025),
        ]
        for (key, year) in table where chip.contains(key) { return year }
        return nil
    }

    private static func bootVolumeInternal() -> Bool? {
        let v = try? URL(fileURLWithPath: "/").resourceValues(forKeys: [.volumeIsInternalKey])
        return v?.volumeIsInternal
    }

    private static func bootVolumeFileSystem() -> String? {
        var s = statfs()
        guard statfs("/", &s) == 0 else { return nil }
        return withUnsafeBytes(of: &s.f_fstypename) { raw -> String? in
            guard let base = raw.bindMemory(to: CChar.self).baseAddress else { return nil }
            return String(cString: base)
        }
    }
}
