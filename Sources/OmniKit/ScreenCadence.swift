import Foundation

/// WHEN A SHOT IS TAKEN (docs/screenshots.md, "Nobody at the Mac"). A shot follows what the user
/// does, not a clock, and nothing is taken once the user has gone:
///
/// - While there is input (keys, pointer, scroll) a shot every `interval`.
/// - When input stops, one more shot, marked settled: what the screen was left showing.
/// - After that a look at 30 s; a look that finds the screen as it was doubles the wait, one that
///   stores a shot keeps it at 30 s (the user reading, a build running, a video).
/// - `away` after the last input, nothing at all until there is input again, whatever the screen
///   does by itself.
///
/// `away` is the system's own measure of a user at the Mac: powerd keeps the user "present, active"
/// for five minutes after the last input (`kIOPMDefaultUserActivityTimeout`, IOPMLibPrivate.h) and
/// has no other evidence of a person. Its further level, "present, passive", is any app holding
/// the display awake, which reads the same on a Mac kept awake with nobody there.
///
/// 0.20.1 had no `away`: a look that stored a shot kept the 30 s for as long as the screen changed,
/// and a menu bar clock changes it every minute. An unattended Mac with its display on recorded
/// about a shot a minute all day.
public struct ScreenCadence: Sendable {
    public static let interval = 5.0
    public static let firstLook = 30.0
    public static let away = 300.0

    public enum Step: Equatable, Sendable {
        case none
        /// The user is at work.
        case shot
        /// The user has just stopped.
        case settled
        /// No input for a while, and the user is still taken to be there.
        case look
    }

    private var wasActive = false
    private var lookGap = ScreenCadence.firstLook
    private var nextLook = 0.0

    public init() {}

    /// What to do now. `now` is any clock that only goes forward, in seconds; `idle` is the time
    /// since the last input.
    public mutating func step(now: Double, idle: Double) -> Step {
        if idle < Self.interval {
            wasActive = true
            lookGap = Self.firstLook
            nextLook = now + lookGap
            return .shot
        }
        guard idle < Self.away else {
            wasActive = false
            return .none
        }
        if wasActive {
            wasActive = false
            nextLook = now + lookGap
            return .settled
        }
        return now >= nextLook ? .look : .none
    }

    /// The settled shot could not be taken (the screen locked, a password field): it is still owed.
    public mutating func settledNotTaken() { wasActive = true }

    /// A look came back, with or without a shot stored.
    public mutating func looked(now: Double, stored: Bool) {
        lookGap = stored ? Self.firstLook : lookGap * 2
        nextLook = now + lookGap
    }

    /// Recording stopped.
    public mutating func reset() { self = ScreenCadence() }
}
