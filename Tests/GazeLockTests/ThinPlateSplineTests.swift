import XCTest
@testable import GazeLock

final class ThinPlateSplineTests: XCTestCase {
    func testExactRecoveryOnControlPoints() {
        let source = [Vec2(0, 0), Vec2(10, 0), Vec2(0, 10), Vec2(10, 10), Vec2(5, 5)]
        let target = source.map { Vec2($0.x + 2.0, $0.y + 1.0) }
        let tps = ThinPlateSpline.fit(source: source, target: target)
        let recovered = tps.evaluate(at: source)
        for (recov, expected) in zip(recovered, target) {
            XCTAssertEqual(recov.x, expected.x, accuracy: 1e-3)
            XCTAssertEqual(recov.y, expected.y, accuracy: 1e-3)
        }
    }

    func testFlowFieldShape() {
        let source = [
            Vec2(0, 0), Vec2(95, 0), Vec2(0, 71), Vec2(95, 71), Vec2(47.5, 35.5),
        ]
        var target = source
        target[4] = Vec2(50, 35)  // shift center 2.5 px right
        let tps = ThinPlateSpline.fit(source: source, target: target)
        let flow = FlowField.from(tps: tps, width: 96, height: 72)
        XCTAssertEqual(flow.width, 96)
        XCTAssertEqual(flow.height, 72)
        XCTAssertEqual(flow.data.count, 96 * 72)
        // Corners should be near-identity (anchored)
        let tl = flow[0, 0]
        XCTAssertEqual(tl.x, 0, accuracy: 0.5)
        XCTAssertEqual(tl.y, 0, accuracy: 0.5)
    }
}
