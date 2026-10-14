import XCTest
@testable import OmniKit

final class PowerRuleTests: XCTestCase {
    private var utc: Calendar { var c = Calendar(identifier: .gregorian); c.timeZone = TimeZone(identifier: "UTC")!; return c }
    private func at(_ h: Int, _ m: Int, day: Int = 13) -> Date {
        utc.date(from: DateComponents(year: 2026, month: 10, day: day, hour: h, minute: m))!
    }

    func testOrderOfReasons() {
        let now = at(12, 0)
        XCTAssertEqual(PowerHold.decide(lowPower: true, thermal: .critical, onBattery: true, indexOnBattery: false, hours: nil, now: now), .lowPower)
        XCTAssertEqual(PowerHold.decide(lowPower: false, thermal: .serious, onBattery: false, indexOnBattery: true, hours: nil, now: now), .hot)
        XCTAssertNil(PowerHold.decide(lowPower: false, thermal: .fair, onBattery: false, indexOnBattery: true, hours: nil, now: now),
                     "fair is a working Mac's ordinary state, not a hold")
        XCTAssertEqual(PowerHold.decide(lowPower: false, thermal: .nominal, onBattery: true, indexOnBattery: false, hours: nil, now: now), .battery)
        XCTAssertNil(PowerHold.decide(lowPower: false, thermal: .nominal, onBattery: true, indexOnBattery: true, hours: nil, now: now),
                     "on battery is fine unless the user said otherwise")
    }

    func testHoursAcrossMidnight() {
        let h = IndexHours(start: 22 * 60, end: 6 * 60, calendar: utc)
        XCTAssertTrue(h.contains(at(23, 30)))
        XCTAssertTrue(h.contains(at(5, 59)))
        XCTAssertFalse(h.contains(at(6, 0)))
        XCTAssertFalse(h.contains(at(12, 0)))
        XCTAssertEqual(h.nextStart(after: at(12, 0)), at(22, 0))
        XCTAssertEqual(h.nextEdge(after: at(23, 0)), at(6, 0, day: 14))
        XCTAssertEqual(h.nextEdge(after: at(12, 0)), at(22, 0))
        let hold = PowerHold.decide(lowPower: false, thermal: .nominal, onBattery: false, indexOnBattery: true, hours: h, now: at(12, 0))
        XCTAssertEqual(hold, .outsideHours(until: at(22, 0)))
    }

    func testSameStartAndEndMeansAllDay() {
        let h = IndexHours(start: 60, end: 60, calendar: utc)
        XCTAssertTrue(h.contains(at(3, 0)))
        XCTAssertNil(h.nextEdge(after: at(3, 0)))
    }

    /// A start the clock skips (02:30 on the spring-forward night) resolves to the next time that exists.
    func testASkippedStartTimeStillComes() {
        var berlin = Calendar(identifier: .gregorian); berlin.timeZone = TimeZone(identifier: "Europe/Berlin")!
        let h = IndexHours(start: 2 * 60 + 30, end: 5 * 60, calendar: berlin)
        let before = berlin.date(from: DateComponents(year: 2026, month: 3, day: 28, hour: 20))!
        let next = h.nextStart(after: before)
        XCTAssertGreaterThan(next, before)
        XCTAssertLessThan(next.timeIntervalSince(before), 24 * 3600)
    }
}
