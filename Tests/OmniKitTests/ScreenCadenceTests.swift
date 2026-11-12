import XCTest
@testable import OmniKit

/// When a shot is taken. These pin that a Mac nobody is at records nothing, however much its screen
/// changes by itself: 0.20.1 took about a shot a minute all day from a display left on.
final class ScreenCadenceTests: XCTestCase {
    /// One tick every `ScreenCadence.interval`; `lastInput` is when the user last touched the Mac
    /// and `stores` what a look finds. The steps taken, by the second.
    private func run(_ cadence: inout ScreenCadence, from start: Double, to end: Double, lastInput: (Double) -> Double,
                     stores: Bool) -> [(at: Double, step: ScreenCadence.Step)] {
        var out: [(Double, ScreenCadence.Step)] = []
        var t = start
        while t <= end {
            let step = cadence.step(now: t, idle: t - lastInput(t))
            if step == .look { cadence.looked(now: t, stored: stores) }
            if step != .none { out.append((t, step)) }
            t += ScreenCadence.interval
        }
        return out
    }

    func testAShotEveryTickWhileThereIsInputThenOneSettled() {
        var cadence = ScreenCadence()
        // Input until second 60, then none.
        let steps = run(&cadence, from: 0, to: 80, lastInput: { min($0, 60) }, stores: true)
        XCTAssertEqual(steps.filter { $0.step == .shot }.count, 13)
        XCTAssertEqual(steps.filter { $0.step == .settled }.map(\.at), [65])
    }

    func testAStillScreenIsLookedAtThreeTimesAndThenLeftAlone() {
        var cadence = ScreenCadence()
        let steps = run(&cadence, from: 0, to: 12 * 3600, lastInput: { _ in 0 }, stores: false)
        XCTAssertEqual(steps.map(\.at), [0, 5, 35, 95, 215])
        XCTAssertEqual(steps.map(\.step), [.shot, .settled, .look, .look, .look])
    }

    /// The display left on with a clock on it: every look finds a changed screen.
    func testAScreenThatChangesByItselfIsNotRecordedOnceTheUserHasGone() {
        var cadence = ScreenCadence()
        let steps = run(&cadence, from: 0, to: 12 * 3600, lastInput: { _ in 0 }, stores: true)
        let looks = steps.filter { $0.step == .look }
        XCTAssertEqual(looks.map(\.at), [35, 65, 95, 125, 155, 185, 215, 245, 275])
        XCTAssertNil(steps.first { $0.at >= ScreenCadence.away }, "a shot with nobody at the Mac")
    }

    /// The negative control for the test above: without `away` the same screen is a look every
    /// 30 s for as long as it changes, which is what 0.20.1 did.
    func testTheLooksAreWhatAwayStops() {
        var cadence = ScreenCadence()
        var looks = 0
        var t = 0.0
        while t <= 3600 {
            // Idle held under `away`: the user is always taken to be there.
            let step = cadence.step(now: t, idle: t == 0 ? 0 : min(t, ScreenCadence.away - 1))
            if step == .look { cadence.looked(now: t, stored: true); looks += 1 }
            t += ScreenCadence.interval
        }
        XCTAssertEqual(looks, 119)
    }

    func testTheUserBackIsRecordedAtOnce() {
        var cadence = ScreenCadence()
        // At the Mac at 0, gone until 3600, then input for 20 s.
        let steps = run(&cadence, from: 0, to: 3700, lastInput: { $0 < 3600 ? 0 : min($0, 3620) }, stores: true)
        let back = steps.filter { $0.at >= 3600 }
        XCTAssertEqual(back.prefix(5).map(\.at), [3600, 3605, 3610, 3615, 3620])
        XCTAssertEqual(back.prefix(5).map(\.step), [.shot, .shot, .shot, .shot, .shot])
        XCTAssertEqual(back.first { $0.step == .settled }?.at, 3625)
        XCTAssertNil(steps.first { $0.at >= ScreenCadence.away && $0.at < 3600 })
    }

    func testASettledShotThatCouldNotBeTakenIsStillOwed() {
        var cadence = ScreenCadence()
        XCTAssertEqual(cadence.step(now: 0, idle: 0), .shot)
        XCTAssertEqual(cadence.step(now: 5, idle: 5), .settled)
        cadence.settledNotTaken()
        XCTAssertEqual(cadence.step(now: 10, idle: 10), .settled)
        XCTAssertEqual(cadence.step(now: 15, idle: 15), .none)
        // Owed when the user goes: dropped, not taken hours later.
        cadence.settledNotTaken()
        XCTAssertEqual(cadence.step(now: 400, idle: 400), .none)
        XCTAssertEqual(cadence.step(now: 405, idle: 100), .look)
    }

    func testALookThatStoresNothingDoublesTheWaitAndOneThatStoresResetsIt() {
        var cadence = ScreenCadence()
        _ = cadence.step(now: 0, idle: 0)
        XCTAssertEqual(cadence.step(now: 5, idle: 5), .settled)
        XCTAssertEqual(cadence.step(now: 35, idle: 35), .look)
        cadence.looked(now: 35, stored: false)
        XCTAssertEqual(cadence.step(now: 90, idle: 90), .none)
        XCTAssertEqual(cadence.step(now: 95, idle: 95), .look)
        cadence.looked(now: 95, stored: true)
        XCTAssertEqual(cadence.step(now: 120, idle: 120), .none)
        XCTAssertEqual(cadence.step(now: 125, idle: 125), .look)
    }
}
