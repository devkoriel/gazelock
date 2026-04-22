import XCTest
@testable import GazeLock

final class AppProfileOverridesTests: XCTestCase {
    func testLookupByBundleId() {
        var overrides = AppProfileOverrides()
        overrides.setProfile(BuiltinProfiles.macbookBuiltIn.id, for: "us.zoom.xos")
        XCTAssertEqual(
            overrides.profileId(for: "us.zoom.xos"),
            BuiltinProfiles.macbookBuiltIn.id
        )
        XCTAssertNil(overrides.profileId(for: "com.example.other"))
        XCTAssertNil(overrides.profileId(for: nil))
    }

    func testRemoveProfile() {
        var overrides = AppProfileOverrides(["x.y.z": BuiltinProfiles.externalTop.id])
        overrides.removeProfile(for: "x.y.z")
        XCTAssertNil(overrides.profileId(for: "x.y.z"))
    }

    func testCodableRoundTrip() throws {
        var overrides = AppProfileOverrides()
        overrides.setProfile(BuiltinProfiles.externalTop.id, for: "us.zoom.xos")
        overrides.setProfile(BuiltinProfiles.externalBottom.id, for: "com.hnc.Discord")
        let data = try JSONEncoder().encode(overrides)
        let decoded = try JSONDecoder().decode(AppProfileOverrides.self, from: data)
        XCTAssertEqual(decoded, overrides)
    }

    func testEmptyInitializer() {
        let overrides = AppProfileOverrides()
        XCTAssertTrue(overrides.bundleIdToProfileId.isEmpty)
    }

    func testDefaultsKeyConstant() {
        XCTAssertEqual(AppProfileOverridesDefaultsKey.key, "AppProfileOverrides")
    }
}
