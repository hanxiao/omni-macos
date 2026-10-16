import XCTest
@testable import OmniKit

/// The benchmark leaves a case off a Mac only for that Mac's physical memory, never for what is
/// open on it (PaperCaseCatalog.fits). Through bench-v9 a 16 GB M2 with its owner's apps open
/// skipped every query and write row; at 16 GB every case and every rung below the RAM tiers runs.
final class BenchCapacityTests: XCTestCase {
    private let gib = 1 << 30

    func testEveryCaseRunsOnA16GBMac() {
        let memory = 16 * gib
        for spec in PaperCaseCatalog.specs(memoryBytes: memory) {
            if let peak = spec.arithmeticPeakMB {
                XCTAssertTrue(PaperCaseCatalog.fits(peakMB: peak, memoryBytes: memory),
                              "\(spec.id.rawValue) needs \(peak) MB")
            }
        }
        // The ladders' rungs a 16 GB Mac is tiered to run, each by the rule its body applies.
        for rows in PaperCaseCatalog.scanLadder(memoryBytes: memory) {
            XCTAssertTrue(PaperCaseCatalog.fits(peakMB: PaperCaseCatalog.storePeakMB(rows: rows, quantized: true), memoryBytes: memory), "scan \(rows)")
        }
    }

    func testTheRuleIsAboutTheMacNotTheMoment() {
        // A pure function of physical memory: the same answer however much is free.
        XCTAssertTrue(PaperCaseCatalog.fits(peakMB: 4_000, memoryBytes: 16 * gib))
        XCTAssertFalse(PaperCaseCatalog.fits(peakMB: 9_000, memoryBytes: 16 * gib))
        XCTAssertTrue(PaperCaseCatalog.fits(peakMB: 9_000, memoryBytes: 24 * gib))
    }
}
