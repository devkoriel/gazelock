import XCTest
@testable import GazeLock

final class PlaceholderTests: XCTestCase {
    func testBundleLoads() {
        let bundle = Bundle(for: Self.self)
        XCTAssertNotNil(bundle, "Test host bundle must load")
    }

    func testAppTarget_hasNonEmptyName() {
        let name = Bundle.main.infoDictionary?["CFBundleName"] as? String
        XCTAssertFalse((name ?? "").isEmpty, "CFBundleName must be populated")
    }
}
