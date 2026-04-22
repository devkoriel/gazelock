import XCTest
@testable import GazeLock

final class UserCalibrationTests: XCTestCase {
    func testRoundTripsViaJSON() throws {
        let cal = UserCalibration(
            baseProfileId: BuiltinProfiles.macbookBuiltIn.id,
            cameraCentroid: .init(yawRad: 0.01, pitchRad: -0.02, rollRad: 0),
            primaryScreenCentroid: .init(yawRad: 0, pitchRad: 0.05, rollRad: 0),
            secondaryScreenCentroids: [
                .init(yawRad: -0.5, pitchRad: 0.02, rollRad: 0),
                .init(yawRad: 0.5, pitchRad: 0.02, rollRad: 0),
            ]
        )
        let data = try JSONEncoder().encode(cal)
        let decoded = try JSONDecoder().decode(UserCalibration.self, from: data)
        XCTAssertEqual(decoded.baseProfileId, cal.baseProfileId)
        XCTAssertEqual(decoded.cameraCentroid, cal.cameraCentroid)
        XCTAssertEqual(decoded.primaryScreenCentroid, cal.primaryScreenCentroid)
        XCTAssertEqual(decoded.secondaryScreenCentroids.count, 2)
    }

    func testCentroidEquality() {
        let a = UserCalibration.Centroid(yawRad: 0.1, pitchRad: 0.2, rollRad: 0.3)
        let b = UserCalibration.Centroid(yawRad: 0.1, pitchRad: 0.2, rollRad: 0.3)
        XCTAssertEqual(a, b)
    }

    func testCreatedAtDefaultsToNow() {
        let before = Date()
        let cal = UserCalibration(
            baseProfileId: "x",
            cameraCentroid: .init(yawRad: 0, pitchRad: 0, rollRad: 0),
            primaryScreenCentroid: .init(yawRad: 0, pitchRad: 0, rollRad: 0),
            secondaryScreenCentroids: []
        )
        let after = Date()
        XCTAssertLessThanOrEqual(before, cal.createdAt)
        XCTAssertLessThanOrEqual(cal.createdAt, after)
    }
}
