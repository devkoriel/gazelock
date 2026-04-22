import XCTest
@testable import GazeLock

final class EntitlementCheckerTests: XCTestCase {
    func testStubReturnsPro() {
        XCTAssertEqual(EntitlementChecker.shared.currentLevel(), .pro)
    }

    func testEveryProFeatureAvailable() {
        for feature in ProFeature.allCases {
            XCTAssertTrue(
                EntitlementChecker.shared.isFeatureAvailable(feature),
                "Pro feature \(feature) should be available in Phase 3b stub"
            )
        }
    }
}
