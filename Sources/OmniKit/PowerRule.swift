import Foundation

/// Why the big indexing work is waiting (App/PowerPolicy watches the system and applies this).
public enum PowerHold: Equatable, Sendable {
    case lowPower
    case hot
    case battery
    case outsideHours(until: Date)

    /// The decision, pure, so it is tested here and used as is by the app. Low Power Mode and heat
    /// win over everything; battery only when the user said not to index on it; the set hours last.
    /// Thermal `.serious` and `.critical` hold; `.fair` does not - it is the ordinary state of a Mac
    /// that is working, and holding on it would hold most of the time.
    public static func decide(lowPower: Bool, thermal: ProcessInfo.ThermalState, onBattery: Bool,
                              indexOnBattery: Bool, hours: IndexHours?, now: Date) -> PowerHold? {
        if lowPower { return .lowPower }
        if thermal == .serious || thermal == .critical { return .hot }
        if onBattery && !indexOnBattery { return .battery }
        if let hours, !hours.contains(now) { return .outsideHours(until: hours.nextStart(after: now)) }
        return nil
    }
}

/// The hours of the day indexing may run, as minutes after midnight: across midnight when `end`
/// comes first (22:00 to 06:00), all day when they are equal.
public struct IndexHours: Equatable, Sendable {
    public var start: Int
    public var end: Int
    public var calendar: Calendar

    public init(start: Int, end: Int, calendar: Calendar = .current) {
        self.start = start; self.end = end; self.calendar = calendar
    }

    private func minute(_ d: Date) -> Int {
        let c = calendar.dateComponents([.hour, .minute], from: d)
        return (c.hour ?? 0) * 60 + (c.minute ?? 0)
    }

    public func contains(_ d: Date) -> Bool {
        if start == end { return true }
        let m = minute(d)
        return start < end ? (m >= start && m < end) : (m >= start || m < end)
    }

    /// When the window next opens after `d` (`d` itself while it is open).
    public func nextStart(after d: Date) -> Date {
        contains(d) ? d : next(minute: start, after: d)
    }

    /// The next time `contains` flips after `d`; nil when it never does (all day).
    public func nextEdge(after d: Date) -> Date? {
        if start == end { return nil }
        return next(minute: contains(d) ? end : start, after: d)
    }

    /// The next wall-clock time at `m` after `d`. A time the clock skips (the spring-forward night)
    /// resolves to the next one that exists.
    private func next(minute m: Int, after d: Date) -> Date {
        calendar.nextDate(after: d, matching: DateComponents(hour: m / 60, minute: m % 60),
                          matchingPolicy: .nextTime) ?? d.addingTimeInterval(3600)
    }
}
