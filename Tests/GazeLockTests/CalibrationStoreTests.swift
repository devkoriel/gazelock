import XCTest
@testable import GazeLock

@MainActor
final class CalibrationStoreTests: XCTestCase {
    private func uniqueTempURL() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathComponent("calibration.json")
    }

    func testSaveAndReload() throws {
        let tmp = uniqueTempURL()
        defer { try? FileManager.default.removeItem(at: tmp.deletingLastPathComponent()) }

        let store = CalibrationStore(fileURL: tmp)
        XCTAssertNil(store.current)

        let cal = UserCalibration(
            baseProfileId: BuiltinProfiles.externalTop.id,
            cameraCentroid: .init(yawRad: 0, pitchRad: 0, rollRad: 0),
            primaryScreenCentroid: .init(yawRad: 0, pitchRad: 0.1, rollRad: 0),
            secondaryScreenCentroids: []
        )
        store.save(cal)

        let store2 = CalibrationStore(fileURL: tmp)
        XCTAssertNotNil(store2.current)
        XCTAssertEqual(store2.current?.baseProfileId, cal.baseProfileId)
    }

    func testClear() {
        let tmp = uniqueTempURL()
        let store = CalibrationStore(fileURL: tmp)
        let cal = UserCalibration(
            baseProfileId: "x",
            cameraCentroid: .init(yawRad: 0, pitchRad: 0, rollRad: 0),
            primaryScreenCentroid: .init(yawRad: 0, pitchRad: 0, rollRad: 0),
            secondaryScreenCentroids: []
        )
        store.save(cal)
        store.clear()
        XCTAssertNil(store.current)
        XCTAssertFalse(FileManager.default.fileExists(atPath: tmp.path))
    }

    func testLoadFromMissingFileReturnsNil() {
        let tmp = uniqueTempURL()
        let store = CalibrationStore(fileURL: tmp)
        XCTAssertNil(store.current)
    }
}
