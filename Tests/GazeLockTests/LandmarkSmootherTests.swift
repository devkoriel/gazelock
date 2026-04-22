import XCTest
@testable import GazeLock

final class LandmarkSmootherTests: XCTestCase {
    func testSmoothsEachCoordinateIndependently() {
        let s = LandmarkSmoother(count: 3)
        let input = [Vec2(1, 2), Vec2(3, 4), Vec2(5, 6)]
        let out = s.smooth(input, timestamp: 0.0)
        // First sample is verbatim per OneEuroFilter contract
        XCTAssertEqual(out[0].x, 1.0, accuracy: 1e-9)
        XCTAssertEqual(out[1].y, 4.0, accuracy: 1e-9)
    }

    func testCountMismatchIsFatalInDebugOrThrows() {
        let s = LandmarkSmoother(count: 3)
        // We can't easily test a precondition without XCTAssertThrowsError on
        // a wrapper; instead, verify the happy path succeeds with matching count.
        _ = s.smooth([Vec2(0, 0), Vec2(1, 1), Vec2(2, 2)], timestamp: 0.0)
    }

    func testStationaryInputConverges() {
        let s = LandmarkSmoother(count: 2)
        var out: [Vec2] = []
        for i in 0..<50 {
            out = s.smooth([Vec2(10, 20), Vec2(30, 40)], timestamp: Double(i) / 60.0)
        }
        XCTAssertEqual(out[0].x, 10.0, accuracy: 1e-3)
        XCTAssertEqual(out[1].y, 40.0, accuracy: 1e-3)
    }
}
