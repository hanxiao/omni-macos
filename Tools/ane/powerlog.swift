// Build: swiftc -O Tools/ane/powerlog.swift -o powerlog (a measurement tool; never shipped: IOReport is private).
// Energy by unit (CPU, GPU, ANE, DRAM) from the chip's own counters, without sudo: IOReport's
// "Energy Model" group, the interface sudoless monitors (macmon) read. System-wide, so a reading
// is compared against an idle one of the same length.
//   powerlog <seconds>                 energy over an idle window
//   powerlog -- <command> [args...]    energy over the command's run
import Foundation
typealias CopyGroup = @convention(c) (CFString?, CFString?, UInt64, UInt64, UInt64) -> Unmanaged<CFMutableDictionary>?
typealias CreateSub = @convention(c) (UnsafeMutableRawPointer?, CFMutableDictionary, UnsafeMutablePointer<Unmanaged<CFMutableDictionary>?>, UInt64, CFTypeRef?) -> OpaquePointer?
typealias CreateSamples = @convention(c) (OpaquePointer, CFMutableDictionary, CFTypeRef?) -> Unmanaged<CFDictionary>?
typealias Delta = @convention(c) (CFDictionary, CFDictionary, CFTypeRef?) -> Unmanaged<CFDictionary>?
typealias GetStr = @convention(c) (CFDictionary) -> Unmanaged<CFString>?
typealias GetInt = @convention(c) (CFDictionary, Int32) -> Int64
let h = dlopen("/usr/lib/libIOReport.dylib", RTLD_NOW)!
func sym<T>(_ n: String) -> T { unsafeBitCast(dlsym(h, n)!, to: T.self) }
let copyGroup: CopyGroup = sym("IOReportCopyChannelsInGroup"), createSub: CreateSub = sym("IOReportCreateSubscription")
let createSamples: CreateSamples = sym("IOReportCreateSamples"), delta: Delta = sym("IOReportCreateSamplesDelta")
let chName: GetStr = sym("IOReportChannelGetChannelName"), unit: GetStr = sym("IOReportChannelGetUnitLabel")
let intVal: GetInt = sym("IOReportSimpleGetIntegerValue")
let chans = copyGroup("Energy Model" as CFString, nil, 0, 0, 0)!.takeRetainedValue()
var subbed: Unmanaged<CFMutableDictionary>?
let sub = createSub(nil, chans, &subbed, 0, nil)!
let subd = subbed!.takeRetainedValue()
func sample() -> CFDictionary { createSamples(sub, subd, nil)!.takeRetainedValue() }
func joules(_ a: CFDictionary, _ b: CFDictionary) -> [String: Double] {
    let d = delta(a, b, nil)!.takeRetainedValue() as NSDictionary
    var out: [String: Double] = [:]
    for case let ch as NSDictionary in (d["IOReportChannels"] as? [Any]) ?? [] {
        let c = ch as CFDictionary
        guard let name = chName(c)?.takeUnretainedValue() as String? else { continue }
        let u = (unit(c)?.takeUnretainedValue() as String?) ?? ""
        let scale = u == "nJ" ? 1e-9 : u == "uJ" ? 1e-6 : u == "mJ" ? 1e-3 : 1
        out[name, default: 0] += Double(intVal(c, 0)) * scale
    }
    return out
}
let args = Array(CommandLine.arguments.dropFirst())
let t0 = Date(), s0 = sample()
if args.first == "--" {
    let p = Process(); p.executableURL = URL(fileURLWithPath: args[1]); p.arguments = Array(args.dropFirst(2))
    try p.run(); p.waitUntilExit()
} else {
    Thread.sleep(forTimeInterval: Double(args.first ?? "10") ?? 10)
}
let secs = -t0.timeIntervalSinceNow, e = joules(s0, sample())
func sum(_ f: (String) -> Bool) -> Double { e.filter { f($0.key) }.values.reduce(0, +) }
// The per-die and per-unit totals, not their parts (summing both counted every core twice).
let cpu = sum { $0.hasSuffix("CPU Energy") }, gpu = e["GPU Energy"] ?? sum { $0.hasPrefix("GPU") && !$0.contains("SRAM") && !$0.contains("CS") }
let ane = e["ANE Energy"] ?? sum { $0.hasPrefix("ANE") }, dram = sum { $0.hasPrefix("DRAM") }
print(String(format: "POWERLOG %.2fs  CPU %.1f J (%.2f W)  GPU %.1f J (%.2f W)  ANE %.2f J (%.3f W)  DRAM %.1f J (%.2f W)",
             secs, cpu, cpu / secs, gpu, gpu / secs, ane, ane / secs, dram, dram / secs))
if ProcessInfo.processInfo.environment["POWERLOG_CHANNELS"] != nil {
    for (k, v) in e.sorted(by: { $0.key < $1.key }) where v > 0 { print(String(format: "  %@ %.3f J", k as NSString, v)) }
}
