import XCTest
@testable import GazeLock

final class AutoDetectTests: XCTestCase {
    func testMacBookCamSingleScreen() {
        let p = AutoDetect.proposeProfile(cameraName: "FaceTime HD Camera", screenCount: 1)
        XCTAssertEqual(p.id, BuiltinProfiles.macbookBuiltIn.id)
    }

    func testMacBookCamMultiScreen() {
        let p = AutoDetect.proposeProfile(cameraName: "FaceTime HD Camera (Built-in)", screenCount: 2)
        XCTAssertEqual(p.id, BuiltinProfiles.macbookWithExternal.id)
    }

    func testNonMacBookCamDefaultsExternalTop() {
        let p = AutoDetect.proposeProfile(cameraName: "Logitech BRIO", screenCount: 1)
        XCTAssertEqual(p.id, BuiltinProfiles.externalTop.id)
    }

    func testUnknownCamWithMultipleScreens() {
        let p = AutoDetect.proposeProfile(cameraName: nil, screenCount: 3)
        XCTAssertEqual(p.id, BuiltinProfiles.externalTop.id)
    }

    func testCaseInsensitiveMatching() {
        let p = AutoDetect.proposeProfile(cameraName: "FACETIME HD CAMERA", screenCount: 1)
        XCTAssertEqual(p.id, BuiltinProfiles.macbookBuiltIn.id)
    }
}
