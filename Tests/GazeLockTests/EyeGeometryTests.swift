import XCTest
@testable import GazeLock

final class EyeGeometryTests: XCTestCase {
    private func squareLandmarks() -> EyeLandmarks {
        EyeLandmarks(
            eyeContour: [Vec2(10, 36), Vec2(20, 30), Vec2(86, 36), Vec2(48, 48)],
            pupilCenter: Vec2(48, 36),
            irisRadiusPx: 10.0
        )
    }

    func testIdentityPoseMovesIrisNowhere() {
        let target = EyeGeometry.targetIrisPx(
            eye: squareLandmarks(),
            headPose: .identity
        )
        XCTAssertEqual(target.x, 48.0, accuracy: 1e-6)
        XCTAssertEqual(target.y, 36.0, accuracy: 1e-6)
    }

    func testYawRightShiftsTargetLeft() {
        let target = EyeGeometry.targetIrisPx(
            eye: squareLandmarks(),
            headPose: HeadPose(yaw: 10.0 * .pi / 180.0, pitch: 0, roll: 0)
        )
        XCTAssertLessThan(target.x, 48.0)
        XCTAssertEqual(target.y, 36.0, accuracy: 0.5)
    }

    func testPitchUpShiftsTargetDown() {
        // Positive pitch = looking up in head frame. The iris target
        // corresponding to "looks at camera" should move down in image
        // coords.
        let target = EyeGeometry.targetIrisPx(
            eye: squareLandmarks(),
            headPose: HeadPose(yaw: 0, pitch: 10.0 * .pi / 180.0, roll: 0)
        )
        XCTAssertGreaterThan(target.y, 36.0)
        XCTAssertEqual(target.x, 48.0, accuracy: 0.5)
    }
}
