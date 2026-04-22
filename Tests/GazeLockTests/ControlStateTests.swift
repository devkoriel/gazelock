import XCTest
@testable import GazeLock

final class ControlStateTests: XCTestCase {
    func testCodableRoundtrip() throws {
        let original = ControlState(
            isEnabled: true,
            intensity: 0.42,
            sourceCameraUniqueID: "ABC123",
            activePresetName: "TestPreset"
        )
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(ControlState.self, from: data)
        XCTAssertEqual(original, decoded)
    }

    func testDefaultValues() {
        let state = ControlState.default
        XCTAssertTrue(state.isEnabled)
        XCTAssertEqual(state.intensity, 0.7, accuracy: 1e-9)
        XCTAssertNil(state.sourceCameraUniqueID)
        XCTAssertEqual(state.activePresetName, "Default")
    }
}
