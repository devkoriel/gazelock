import XCTest
@testable import GazeLock

final class EyeGeometryAimTests: XCTestCase {
    private func mkEye(pupilX: Double = 100, pupilY: Double = 100, irisPx: Double = 24) -> EyeLandmarks {
        EyeLandmarks(
            eyeContour: [
                Vec2(pupilX - 18, pupilY - 8),
                Vec2(pupilX - 12, pupilY - 12),
                Vec2(pupilX, pupilY - 14),
                Vec2(pupilX + 12, pupilY - 12),
                Vec2(pupilX + 18, pupilY - 8),
                Vec2(pupilX + 18, pupilY + 8),
                Vec2(pupilX + 12, pupilY + 12),
                Vec2(pupilX, pupilY + 14)
            ],
            pupilCenter: Vec2(pupilX, pupilY),
            irisRadiusPx: irisPx
        )
    }

    func testZeroAimMatchesLegacyBehaviour() {
        let eye = mkEye()
        let pose = HeadPose(yaw: 0, pitch: 0, roll: 0)
        let legacy = EyeGeometry.targetIrisPx(eye: eye, headPose: pose)
        let zeroed = EyeGeometry.targetIrisPx(
            eye: eye,
            headPose: pose,
            verticalAimDeg: 0,
            horizontalAimDeg: 0
        )
        XCTAssertEqual(legacy.x, zeroed.x, accuracy: 1e-6)
        XCTAssertEqual(legacy.y, zeroed.y, accuracy: 1e-6)
    }

    func testPositiveVerticalAimShiftsTargetDownInImage() {
        let eye = mkEye()
        let pose = HeadPose(yaw: 0, pitch: 0, roll: 0)
        let base = EyeGeometry.targetIrisPx(eye: eye, headPose: pose, verticalAimDeg: 0)
        let shifted = EyeGeometry.targetIrisPx(eye: eye, headPose: pose, verticalAimDeg: 2.0)
        XCTAssertGreaterThan(shifted.y, base.y)
        // Expected: 12mm * tan(2°) * (24/6) = 12 * 0.0349 * 4 ≈ 1.68 px
        XCTAssertEqual(shifted.y - base.y, 1.68, accuracy: 0.1)
    }

    func testNegativeVerticalAimShiftsTargetUp() {
        let eye = mkEye()
        let pose = HeadPose(yaw: 0, pitch: 0, roll: 0)
        let base = EyeGeometry.targetIrisPx(eye: eye, headPose: pose, verticalAimDeg: 0)
        let shifted = EyeGeometry.targetIrisPx(eye: eye, headPose: pose, verticalAimDeg: -2.0)
        XCTAssertLessThan(shifted.y, base.y)
    }

    func testPositiveHorizontalAimShiftsTargetRight() {
        let eye = mkEye()
        let pose = HeadPose(yaw: 0, pitch: 0, roll: 0)
        let base = EyeGeometry.targetIrisPx(eye: eye, headPose: pose, horizontalAimDeg: 0)
        let shifted = EyeGeometry.targetIrisPx(eye: eye, headPose: pose, horizontalAimDeg: 3.0)
        XCTAssertGreaterThan(shifted.x, base.x)
    }

    func testAimCompoundsWithHeadPose() {
        let eye = mkEye()
        let poseYawed = HeadPose(yaw: 0.2, pitch: 0, roll: 0)
        let zeroAim = EyeGeometry.targetIrisPx(eye: eye, headPose: poseYawed, verticalAimDeg: 0)
        let withAim = EyeGeometry.targetIrisPx(eye: eye, headPose: poseYawed, verticalAimDeg: 2.0)
        XCTAssertGreaterThan(withAim.y, zeroAim.y)
    }
}
