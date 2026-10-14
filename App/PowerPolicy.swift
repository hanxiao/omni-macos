import Foundation
import IOKit.ps
import OmniKit

/// When the BIG work may run: a full pass and a folder's first index. Held while the Mac is in Low
/// Power Mode, while it is hot (thermal state serious or critical), on battery when the user says so,
/// and outside the hours the user sets (the rule: PowerHold.decide). A few changed files always
/// index at once - the watcher's reconcile is not held - because seconds of work is not what drains
/// a battery.
///
/// EVENT-DRIVEN, nothing polls: Low Power Mode and thermal state post notifications, the power
/// source has an IOKit run-loop source, and the set hours have one timer armed for their next edge
/// (re-armed when the clock changes). The owner (AppModel) stands indexing down on a hold and
/// resumes it when the hold lifts, the way an OCR run does. In the app's process, on the main run
/// loop: no helper, nothing extra running.
@MainActor @Observable
final class PowerPolicy {
    /// Called on the main thread whenever the hold changes.
    @ObservationIgnored var onChange: ((PowerHold?) -> Void)?
    private(set) var hold: PowerHold?

    // The user's choices (Settings > Performance).
    var indexOnBattery: Bool { didSet { persist(); evaluate() } }
    var hoursEnabled: Bool { didSet { persist(); evaluate() } }
    /// Minutes after midnight. The window runs across midnight when `end` comes first; equal means all day.
    var hoursStart: Int { didSet { persist(); evaluate() } }
    var hoursEnd: Int { didSet { persist(); evaluate() } }

    @ObservationIgnored private var observers: [NSObjectProtocol] = []
    @ObservationIgnored private var powerSource: CFRunLoopSource?
    @ObservationIgnored private var edgeTimer: Timer?

    init() {
        let d = UserDefaults.standard
        // bool(forKey:) and integer(forKey:), not `as? Bool` / `as? Int`: a launch argument
        // arrives as a string, and the cast read it as missing.
        indexOnBattery = d.object(forKey: "omni.indexOnBattery") == nil ? true : d.bool(forKey: "omni.indexOnBattery")
        hoursEnabled = d.bool(forKey: "omni.indexHours.enabled")
        hoursStart = d.object(forKey: "omni.indexHours.start") == nil ? 60 : d.integer(forKey: "omni.indexHours.start")     // 01:00
        hoursEnd = d.object(forKey: "omni.indexHours.end") == nil ? 420 : d.integer(forKey: "omni.indexHours.end")         // 07:00
    }

    func start() {
        guard observers.isEmpty else { return }
        let nc = NotificationCenter.default
        for name in [Notification.Name.NSProcessInfoPowerStateDidChange,
                     ProcessInfo.thermalStateDidChangeNotification,
                     Notification.Name.NSSystemClockDidChange] {
            observers.append(nc.addObserver(forName: name, object: nil, queue: .main) { [weak self] _ in
                MainActor.assumeIsolated { self?.evaluate() }
            })
        }
        // IOKit calls this on the main run loop when the power source changes (AC <-> battery).
        let context = Unmanaged.passUnretained(self).toOpaque()
        if let src = IOPSNotificationCreateRunLoopSource({ ctx in
            guard let ctx else { return }
            let me = Unmanaged<PowerPolicy>.fromOpaque(ctx).takeUnretainedValue()
            MainActor.assumeIsolated { me.evaluate() }
        }, context)?.takeRetainedValue() {
            CFRunLoopAddSource(CFRunLoopGetMain(), src, .defaultMode)
            powerSource = src
        }
        evaluate()
    }

    /// Recompute the hold, tell the owner if it changed, and arm the timer for the next hours edge.
    func evaluate() {
        let next = PowerHold.decide(lowPower: ProcessInfo.processInfo.isLowPowerModeEnabled,
                                    thermal: ProcessInfo.processInfo.thermalState,
                                    onBattery: Self.onBattery(), indexOnBattery: indexOnBattery,
                                    hours: hoursEnabled ? IndexHours(start: hoursStart, end: hoursEnd) : nil,
                                    now: Date())
        armEdgeTimer()
        guard next != hold else { return }
        hold = next
        omniPerfLog("power hold \(next.map { "\($0)" } ?? "none")")
        onChange?(next)
    }

    private func armEdgeTimer() {
        edgeTimer?.invalidate(); edgeTimer = nil
        guard hoursEnabled, let edge = IndexHours(start: hoursStart, end: hoursEnd).nextEdge(after: Date()) else { return }
        let t = Timer(fire: edge.addingTimeInterval(1), interval: 0, repeats: false) { [weak self] _ in
            MainActor.assumeIsolated { self?.evaluate() }
        }
        RunLoop.main.add(t, forMode: .common)
        edgeTimer = t
    }

    /// On battery power right now (a desktop never is).
    nonisolated static func onBattery() -> Bool {
        guard let info = IOPSCopyPowerSourcesInfo()?.takeRetainedValue(),
              let type = IOPSGetProvidingPowerSourceType(info)?.takeUnretainedValue() as String? else { return false }
        return type == kIOPSBatteryPowerValue
    }

    private func persist() {
        OmniPrefs.set(indexOnBattery, forKey: "omni.indexOnBattery")
        OmniPrefs.set(hoursEnabled, forKey: "omni.indexHours.enabled")
        OmniPrefs.set(hoursStart, forKey: "omni.indexHours.start")
        OmniPrefs.set(hoursEnd, forKey: "omni.indexHours.end")
    }
}

extension PowerHold {
    /// What the Settings status line says.
    var reason: String {
        switch self {
        case .lowPower: return "Waiting while Low Power Mode is on"
        case .hot: return "Waiting for the Mac to cool down"
        case .battery: return "Waiting for power"
        case .outsideHours(let until): return "Waiting until \(until.formatted(date: .omitted, time: .shortened))"
        }
    }
}
