import XCTest
@testable import GazeLock

final class IrisKalmanTests: XCTestCase {
    func testFirstMeasurementReturnsVerbatim() {
        let k = IrisKalman()
        let p = k.update(measurement: Vec2(100, 200), timestamp: 0.0)
        XCTAssertEqual(p.x, 100.0, accuracy: 1e-9)
        XCTAssertEqual(p.y, 200.0, accuracy: 1e-9)
    }

    func testStationarySequenceConverges() {
        let k = IrisKalman()
        var result = Vec2.zero
        for i in 0..<30 {
            result = k.update(measurement: Vec2(50, 75), timestamp: Double(i) / 60.0)
        }
        XCTAssertEqual(result.x, 50.0, accuracy: 0.5)
        XCTAssertEqual(result.y, 75.0, accuracy: 0.5)
    }

    func testTracksLinearMotionRoughly() {
        let k = IrisKalman()
        // Linear motion: x grows by 1 per tick, y constant.
        var last = Vec2.zero
        for i in 0..<20 {
            last = k.update(
                measurement: Vec2(Double(i), 10.0),
                timestamp: Double(i) / 60.0
            )
        }
        // After many steps, should be near the measured position
        XCTAssertEqual(last.x, 19.0, accuracy: 2.0)
        XCTAssertEqual(last.y, 10.0, accuracy: 1.0)
    }
}
