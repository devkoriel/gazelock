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

final class ControlStateNaturalGazeTests: XCTestCase {
    func testDefaultStateHasMaxineFriendlyDefaults() {
        let s = ControlState.default
        XCTAssertEqual(s.setupProfileId, BuiltinProfiles.externalTop.id)
        XCTAssertEqual(s.verticalAimDeg, -2.0, accuracy: 0.001)
        XCTAssertEqual(s.horizontalAimDeg, 0.0, accuracy: 0.001)
        XCTAssertEqual(s.sensitivity, .normal)
    }

    func testRoundTripsAllFields() throws {
        var s = ControlState()
        s.verticalAimDeg = 3.5
        s.horizontalAimDeg = -1.2
        s.sensitivity = .tight
        s.setupProfileId = BuiltinProfiles.macbookBuiltIn.id
        let data = try JSONEncoder().encode(s)
        let decoded = try JSONDecoder().decode(ControlState.self, from: data)
        XCTAssertEqual(decoded, s)
    }

    func testDecodeOldJsonFillsDefaults() throws {
        let oldJson = #"{"isEnabled":true,"intensity":0.5,"activePresetName":"X"}"#.data(using: .utf8)!
        let decoded = try JSONDecoder().decode(ControlState.self, from: oldJson)
        XCTAssertTrue(decoded.isEnabled)
        XCTAssertEqual(decoded.intensity, 0.5, accuracy: 0.001)
        XCTAssertEqual(decoded.setupProfileId, BuiltinProfiles.externalTop.id)
        XCTAssertEqual(decoded.sensitivity, .normal)
    }

    func testBuiltinProfilesLookupFallsBackToExternalTop() {
        let bogus = BuiltinProfiles.byId("nonexistent")
        XCTAssertEqual(bogus.id, BuiltinProfiles.externalTop.id)
    }

    func testSensitivityThresholdsMonotonic() {
        XCTAssertLessThan(Sensitivity.tight.yawTransitionDeg.high, Sensitivity.normal.yawTransitionDeg.high)
        XCTAssertLessThan(Sensitivity.normal.yawTransitionDeg.high, Sensitivity.loose.yawTransitionDeg.high)
    }
}
