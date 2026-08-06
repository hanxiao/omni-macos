import Foundation
import Darwin
import IOKit
import IOKit.ps
import Metal
import MLX

// Machine facts and run-time conditions for the paper benchmark's export.
//
// Deliberately NOT part of HardwareProfile: that struct is the wire contract of the /omni/profiling
// upload and its shape must not move. Everything new lives here.
//
// The static facts answer "which machine produced this row". The sampled ones answer the question
// that actually decides whether a row is usable: was this machine QUIET while it measured. A
// laptop that thermal-throttled, swapped, or shared its cores with a browser produces numbers that
// look like a design difference and are not one - measurements.md records three occasions where a
// contended run was wrong by up to 1.8x. So thermal state, swap, load average and system-wide CPU
// busy are sampled at every case boundary and stamped, and the dialog warns before the operator
// sends the file.
//
// PRIVACY: this text gets pasted into a chat. No hostname, no username, no serial, no file paths,
// and deliberately NO process names - only a COUNT of busy processes.

/// One sample of the machine's run-time condition. Cheap enough to take at every case boundary.
public struct SystemSnapshot: Sendable, Codable {
    public var thermal: String            // nominal | fair | serious | critical
    public var lowPowerMode: Bool
    public var powerSource: String        // ac | battery | unknown
    public var batteryPercent: Int?
    public var batteryCharging: Bool?
    public var loadAvg1m: Double
    public var loadAvg5m: Double
    public var loadAvg15m: Double
    public var swapUsedMB: Double
    public var memFreeMB: Double          // free + inactive + speculative, i.e. reclaimable
    public var memPressureLevel: Int      // 1 normal / 2 warn / 4 critical (-1 unknown)
    public var memCompressedMB: Double
    public var memWiredMB: Double
    public var footprintMB: Double        // our own phys_footprint
    public var mlxActiveMB: Double
    public var mlxPeakMB: Double
    public var uptimeSeconds: Double
    /// Cumulative system CPU ticks (user, system, idle, nice) - a DIFFERENCE of two snapshots is
    /// what yields a busy percentage; the absolute value is meaningless on its own.
    public var cpuTicks: [UInt64]
    /// Our own cumulative CPU seconds (getrusage user + sys), same differencing rule.
    public var ownCPUSeconds: Double
    public var wallClock: Double          // CFAbsoluteTime-equivalent, for the difference denominator
}

/// Facts that cannot change during a run.
public struct SystemStatics: Sendable, Codable {
    public var chip: String
    public var hwModel: String
    public var cpuPCores: Int
    public var cpuECores: Int
    public var cpuPhysical: Int
    public var cpuLogical: Int
    public var perfLevels: Int
    public var gpuCores: Int?
    public var memoryBytes: Int
    public var recommendedWorkingSetBytes: Int?
    public var unifiedMemory: Bool
    public var osVersion: String
    public var osBuild: String
    public var diskFreeBytes: Int?
    public var diskInternal: Bool?
    public var diskFileSystem: String?
}

public enum SystemProbe {

    // MARK: - Statics

    public static func statics() -> SystemStatics {
        let dev = MTLCreateSystemDefaultDevice()
        return SystemStatics(
            chip: sysctlString("machdep.cpu.brand_string") ?? "Unknown",
            hwModel: sysctlString("hw.model") ?? "Unknown",
            // perflevel0 is the PERFORMANCE cluster and perflevel1 the efficiency one on every
            // shipped Apple-silicon part; a single-cluster part reports nperflevels == 1 and no
            // perflevel1, which reads back as 0 E cores rather than a wrong number.
            cpuPCores: sysctlInt("hw.perflevel0.logicalcpu") ?? 0,
            cpuECores: sysctlInt("hw.perflevel1.logicalcpu") ?? 0,
            cpuPhysical: sysctlInt("hw.physicalcpu") ?? 0,
            cpuLogical: sysctlInt("hw.logicalcpu") ?? ProcessInfo.processInfo.activeProcessorCount,
            perfLevels: sysctlInt("hw.nperflevels") ?? 1,
            gpuCores: gpuCoreCount(),
            memoryBytes: Int(ProcessInfo.processInfo.physicalMemory),
            recommendedWorkingSetBytes: dev.map { Int($0.recommendedMaxWorkingSetSize) },
            unifiedMemory: dev?.hasUnifiedMemory ?? true,
            osVersion: {
                let v = ProcessInfo.processInfo.operatingSystemVersion
                return "\(v.majorVersion).\(v.minorVersion).\(v.patchVersion)"
            }(),
            // The BUILD, not the marketing version: two machines both saying "26.1" can be on
            // different kernels/Metal drivers, and that shows up in these numbers.
            osBuild: sysctlString("kern.osversion") ?? "unknown",
            diskFreeBytes: bootVolumeFreeBytes(),
            diskInternal: bootVolumeInternal(),
            diskFileSystem: bootVolumeFileSystem()
        )
    }

    // MARK: - Sampled

    public static func snapshot() -> SystemSnapshot {
        let (source, pct, charging) = powerSource()
        let vm = vmStats()
        let pageSize = Double(vmPageSize())
        return SystemSnapshot(
            thermal: thermalName(ProcessInfo.processInfo.thermalState),
            lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled,
            powerSource: source,
            batteryPercent: pct,
            batteryCharging: charging,
            loadAvg1m: loadAverage().0,
            loadAvg5m: loadAverage().1,
            loadAvg15m: loadAverage().2,
            swapUsedMB: swapUsedBytes().map { Double($0) / 1_048_576 } ?? -1,
            memFreeMB: vm.map { Double($0.free_count &+ $0.inactive_count &+ $0.speculative_count) * pageSize / 1_048_576 } ?? -1,
            memPressureLevel: sysctlInt("kern.memorystatus_vm_pressure_level") ?? -1,
            memCompressedMB: vm.map { Double($0.compressor_page_count) * pageSize / 1_048_576 } ?? -1,
            memWiredMB: vm.map { Double($0.wire_count) * pageSize / 1_048_576 } ?? -1,
            footprintMB: Double(footprintBytes()) / 1_048_576,
            mlxActiveMB: Double(MLX.Memory.activeMemory) / 1_048_576,
            mlxPeakMB: Double(MLX.Memory.peakMemory) / 1_048_576,
            uptimeSeconds: uptimeSeconds(),
            cpuTicks: cpuTicks(),
            ownCPUSeconds: ownCPUSeconds(),
            wallClock: Date().timeIntervalSince1970
        )
    }

    /// System-wide CPU busy percent between two snapshots, and our own share of it in CORES.
    /// `contended` means somebody ELSE was using more than a fifth of the machine while we measured.
    public static func busy(from a: SystemSnapshot, to b: SystemSnapshot)
        -> (systemBusyPct: Double, ownCores: Double, contended: Bool) {
        guard a.cpuTicks.count == 4, b.cpuTicks.count == 4 else { return (-1, -1, false) }
        var total: Double = 0, idle: Double = 0
        for i in 0 ..< 4 {
            let d = Double(b.cpuTicks[i] &- a.cpuTicks[i])
            total += d
            if i == 2 { idle += d }          // CPU_STATE_IDLE
        }
        guard total > 0 else { return (-1, -1, false) }
        let busyPct = 100.0 * (total - idle) / total
        let wall = max(0.001, b.wallClock - a.wallClock)
        let cores = max(0, b.ownCPUSeconds - a.ownCPUSeconds) / wall
        let logical = Double(max(1, ProcessInfo.processInfo.activeProcessorCount))
        let ownPct = 100.0 * cores / logical
        return (busyPct, cores, busyPct - ownPct > 20)
    }

    /// Count of OTHER processes over 20% of one core, measured across a short window. Names are
    /// deliberately not collected: the export is pasted into a chat. Returns -1 when the kernel
    /// refuses (processes owned by other users are not readable without extra entitlements, so this
    /// is a lower bound by construction).
    public static func processesOver20Percent(window: TimeInterval = 0.30) -> Int {
        guard let first = pidCPUSeconds() else { return -1 }
        Thread.sleep(forTimeInterval: window)
        guard let second = pidCPUSeconds() else { return -1 }
        let me = getpid()
        var n = 0
        for (pid, t1) in second where pid != me {
            guard let t0 = first[pid] else { continue }
            if (t1 - t0) / window > 0.20 { n += 1 }
        }
        return n
    }

    // MARK: - Primitives

    public static func footprintBytes() -> Int {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) { p in
            p.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? Int(info.phys_footprint) : 0
    }

    public static func swapUsedBytes() -> Int? {
        var usage = xsw_usage()
        var size = MemoryLayout<xsw_usage>.size
        guard sysctlbyname("vm.swapusage", &usage, &size, nil, 0) == 0 else { return nil }
        return Int(usage.xsu_used)
    }

    public static func thermalName(_ s: ProcessInfo.ThermalState) -> String {
        switch s {
        case .nominal: "nominal"
        case .fair: "fair"
        case .serious: "serious"
        case .critical: "critical"
        @unknown default: "unknown"
        }
    }

    public static func loadAverage() -> (Double, Double, Double) {
        var a = [Double](repeating: 0, count: 3)
        guard getloadavg(&a, 3) == 3 else { return (-1, -1, -1) }
        return (a[0], a[1], a[2])
    }

    static func sysctlString(_ name: String) -> String? {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
        var buf = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &buf, &size, nil, 0) == 0 else { return nil }
        return String(cString: buf).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func sysctlInt(_ name: String) -> Int? {
        var v: Int64 = 0
        var size = MemoryLayout<Int64>.size
        if sysctlbyname(name, &v, &size, nil, 0) == 0, size == MemoryLayout<Int64>.size { return Int(v) }
        var v32: Int32 = 0
        var size32 = MemoryLayout<Int32>.size
        guard sysctlbyname(name, &v32, &size32, nil, 0) == 0 else { return nil }
        return Int(v32)
    }

    private static func vmPageSize() -> Int {
        var s: UInt = 0
        return host_page_size(mach_host_self(), &s) == KERN_SUCCESS ? Int(s) : 16384
    }

    private static func vmStats() -> vm_statistics64_data_t? {
        var stats = vm_statistics64_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &stats) { p in
            p.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? stats : nil
    }

    private static func cpuTicks() -> [UInt64] {
        var info = host_cpu_load_info()
        var count = mach_msg_type_number_t(MemoryLayout<host_cpu_load_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) { p in
            p.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO, $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return [] }
        return [UInt64(info.cpu_ticks.0), UInt64(info.cpu_ticks.1), UInt64(info.cpu_ticks.2), UInt64(info.cpu_ticks.3)]
    }

    private static func ownCPUSeconds() -> Double {
        var r = rusage()
        guard getrusage(RUSAGE_SELF, &r) == 0 else { return 0 }
        func secs(_ t: timeval) -> Double { Double(t.tv_sec) + Double(t.tv_usec) / 1e6 }
        return secs(r.ru_utime) + secs(r.ru_stime)
    }

    private static func uptimeSeconds() -> Double {
        var tv = timeval()
        var size = MemoryLayout<timeval>.size
        guard sysctlbyname("kern.boottime", &tv, &size, nil, 0) == 0, tv.tv_sec > 0 else { return -1 }
        return Date().timeIntervalSince1970 - Double(tv.tv_sec)
    }

    /// Per-pid cumulative CPU seconds. Processes we cannot read are simply absent from the map.
    private static func pidCPUSeconds() -> [pid_t: Double]? {
        let cap = Int(proc_listpids(UInt32(PROC_ALL_PIDS), 0, nil, 0))
        guard cap > 0 else { return nil }
        var pids = [pid_t](repeating: 0, count: cap / MemoryLayout<pid_t>.size + 32)
        let bytes = pids.withUnsafeMutableBufferPointer { buf -> Int32 in
            proc_listpids(UInt32(PROC_ALL_PIDS), 0, buf.baseAddress, Int32(buf.count * MemoryLayout<pid_t>.size))
        }
        guard bytes > 0 else { return nil }
        var out: [pid_t: Double] = [:]
        for pid in pids.prefix(Int(bytes) / MemoryLayout<pid_t>.size) where pid > 0 {
            var info = rusage_info_v2()
            let rc = withUnsafeMutablePointer(to: &info) { p -> Int32 in
                p.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
                    proc_pid_rusage(pid, RUSAGE_INFO_V2, $0)
                }
            }
            guard rc == 0 else { continue }
            out[pid] = Double(info.ri_user_time &+ info.ri_system_time) / 1e9
        }
        return out.isEmpty ? nil : out
    }

    /// GPU core count, resolved once. The store reads this to place the bf16/int4 crossover, which
    /// is on the base-build path, so it must not re-walk the IORegistry each time.
    nonisolated(unsafe) private static let cachedGPUCores: Int? = gpuCoreCount()
    public static func gpuCores() -> Int? { cachedGPUCores }

    /// GPU core count from the IORegistry. The app is unsandboxed, so no entitlement is needed, and
    /// this costs microseconds - `system_profiler` costs about a second and is never worth it here.
    private static func gpuCoreCount() -> Int? {
        var iterator: io_iterator_t = 0
        guard IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("AGXAccelerator"), &iterator) == KERN_SUCCESS else {
            return nil
        }
        defer { IOObjectRelease(iterator) }
        var service = IOIteratorNext(iterator)
        while service != 0 {
            defer { IOObjectRelease(service) }
            if let raw = IORegistryEntryCreateCFProperty(service, "gpu-core-count" as CFString, kCFAllocatorDefault, 0),
               let n = raw.takeRetainedValue() as? Int {
                return n
            }
            service = IOIteratorNext(iterator)
        }
        return nil
    }

    /// "ac" / "battery" / "unknown", plus percent and charging when a battery exists. A desktop
    /// reports no power sources at all, which is AC.
    private static func powerSource() -> (String, Int?, Bool?) {
        guard let blob = IOPSCopyPowerSourcesInfo()?.takeRetainedValue(),
              let list = IOPSCopyPowerSourcesList(blob)?.takeRetainedValue() as? [CFTypeRef]
        else { return ("unknown", nil, nil) }
        if list.isEmpty { return ("ac", nil, nil) }
        for ps in list {
            guard let d = IOPSGetPowerSourceDescription(blob, ps)?.takeUnretainedValue() as? [String: Any] else { continue }
            let state = d[kIOPSPowerSourceStateKey as String] as? String
            let cur = d[kIOPSCurrentCapacityKey as String] as? Int
            let mx = d[kIOPSMaxCapacityKey as String] as? Int
            let charging = d[kIOPSIsChargingKey as String] as? Bool
            let pct: Int? = (cur != nil && (mx ?? 0) > 0) ? Int(100.0 * Double(cur!) / Double(mx!)) : cur
            return (state == (kIOPSBatteryPowerValue as String) ? "battery" : "ac", pct, charging)
        }
        return ("unknown", nil, nil)
    }

    private static func bootVolumeFreeBytes() -> Int? {
        let v = try? URL(fileURLWithPath: "/").resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return v?.volumeAvailableCapacityForImportantUsage.map { Int($0) }
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
