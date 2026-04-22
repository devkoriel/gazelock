import XCTest
@testable import GazeLock

final class LookAwayDetectorTests: XCTestCase {
    private func pose(yawDeg: Double, pitchDeg: Double = 0, rollDeg: Double = 0) -> HeadPose {
        HeadPose(
            yaw: yawDeg * .pi / 180,
            pitch: pitchDeg * .pi / 180,
            roll: rollDeg * .pi / 180
        )
    }

    func testFullyEngagedWhenHeadFrontal() {
        let d = LookAwayDetector(sensitivity: .normal)
        for i in 0..<30 {
            _ = d.compute(headPose: pose(yawDeg: 0), timestampSeconds: Double(i) / 60.0)
        }
        let alpha = d.compute(headPose: pose(yawDeg: 0), timestampSeconds: 1.0)
        XCTAssertGreaterThan(alpha, 0.9)
    }

    func testFullyDisengagedWhenHeadPastHighYaw() {
        let d = LookAwayDetector(sensitivity: .normal)
        for i in 0..<30 {
            _ = d.compute(headPose: pose(yawDeg: 45), timestampSeconds: Double(i) / 60.0)
        }
        let alpha = d.compute(headPose: pose(yawDeg: 45), timestampSeconds: 1.0)
        XCTAssertLessThan(alpha, 0.05)
    }

    func testTransitionBandSmoothlyFades() {
        let d = LookAwayDetector(sensitivity: .normal)
        var midAlpha: Double = 0
        for i in 0..<120 {
            midAlpha = d.compute(headPose: pose(yawDeg: 27.5), timestampSeconds: Double(i) / 60.0)
        }
        XCTAssertEqual(midAlpha, 0.5, accuracy: 0.15)
    }

    func testReEngageLockoutBlocksInstantReturn() {
        let d = LookAwayDetector(sensitivity: .normal)
        for i in 0..<60 {
            _ = d.compute(headPose: pose(yawDeg: 45), timestampSeconds: Double(i) / 60.0)
        }
        let alpha1 = d.compute(headPose: pose(yawDeg: 0), timestampSeconds: 60.0 / 60.0)
        XCTAssertEqual(alpha1, 0.0, accuracy: 0.001, "Should be in lockout")
        let alpha2 = d.compute(headPose: pose(yawDeg: 0), timestampSeconds: 60.0 / 60.0 + 0.25)
        XCTAssertEqual(alpha2, 0.0, accuracy: 0.001)
        let alpha3 = d.compute(headPose: pose(yawDeg: 0), timestampSeconds: 60.0 / 60.0 + 0.35)
        XCTAssertGreaterThan(alpha3, 0.0)
    }

    func testPitchAloneDisengages() {
        let d = LookAwayDetector(sensitivity: .normal)
        for i in 0..<60 {
            _ = d.compute(headPose: pose(yawDeg: 0, pitchDeg: 40), timestampSeconds: Double(i) / 60.0)
        }
        let alpha = d.compute(headPose: pose(yawDeg: 0, pitchDeg: 40), timestampSeconds: 1.0)
        XCTAssertLessThan(alpha, 0.05)
    }

    func testTighterSensitivityDisengagesSooner() {
        let normalD = LookAwayDetector(sensitivity: .normal)
        let tightD = LookAwayDetector(sensitivity: .tight)
        for i in 0..<120 {
            _ = normalD.compute(headPose: pose(yawDeg: 20), timestampSeconds: Double(i) / 60.0)
            _ = tightD.compute(headPose: pose(yawDeg: 20), timestampSeconds: Double(i) / 60.0)
        }
        let nAlpha = normalD.compute(headPose: pose(yawDeg: 20), timestampSeconds: 2.0)
        let tAlpha = tightD.compute(headPose: pose(yawDeg: 20), timestampSeconds: 2.0)
        XCTAssertGreaterThan(nAlpha, tAlpha)
    }

    func testResetClearsSmoothedState() {
        let d = LookAwayDetector(sensitivity: .normal)
        for i in 0..<60 {
            _ = d.compute(headPose: pose(yawDeg: 45), timestampSeconds: Double(i) / 60.0)
        }
        d.reset()
        let alpha = d.compute(headPose: pose(yawDeg: 0), timestampSeconds: 2.0)
        XCTAssertGreaterThan(alpha, 0.9)
    }
}
