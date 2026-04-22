import XCTest
@testable import GazeLock

final class OneEuroFilterTests: XCTestCase {
    func testFirstSampleReturnsInputVerbatim() {
        let f = OneEuroFilter()
        XCTAssertEqual(f.filter(42.0, timestamp: 0.0), 42.0, accuracy: 1e-12)
    }

    func testStationarySignalConvergesToConstant() {
        let f = OneEuroFilter()
        var result = 0.0
        for i in 0..<100 {
            result = f.filter(7.0, timestamp: Double(i) / 60.0)
        }
        XCTAssertEqual(result, 7.0, accuracy: 1e-6)
    }

    func testResponseToStepIsSmoothedNotInstant() {
        let f = OneEuroFilter()
        // Warm up on stationary signal
        for i in 0..<30 {
            _ = f.filter(0.0, timestamp: Double(i) / 60.0)
        }
        // Step to 10
        let firstStep = f.filter(10.0, timestamp: 30.0 / 60.0)
        // Filter should not jump all the way to 10 immediately
        XCTAssertGreaterThan(firstStep, 0.0)
        XCTAssertLessThan(firstStep, 10.0)
    }

    func testResetClearsState() {
        let f = OneEuroFilter()
        _ = f.filter(5.0, timestamp: 0.0)
        _ = f.filter(5.5, timestamp: 1.0 / 60.0)
        f.reset()
        // After reset, next sample is verbatim (first-sample rule)
        XCTAssertEqual(f.filter(100.0, timestamp: 2.0 / 60.0), 100.0, accuracy: 1e-12)
    }
}
