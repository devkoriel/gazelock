import XCTest
@testable import GazeLock

@MainActor
final class AppProfileObserverTests: XCTestCase {
    func testInitialStatePreservedWhenNoOverride() {
        let store = ControlStateStore(initial: ControlState(
            setupProfileId: BuiltinProfiles.externalBottom.id
        ))
        let overrides = AppProfileOverrides()
        _ = AppProfileObserver(
            overrides: overrides,
            defaultProfileId: BuiltinProfiles.externalTop.id,
            store: store
        )
        XCTAssertEqual(store.state.setupProfileId, BuiltinProfiles.externalBottom.id)
    }

    func testSetOverridesIsIdempotent() {
        let store = ControlStateStore(initial: ControlState())
        var overrides = AppProfileOverrides()
        overrides.setProfile(BuiltinProfiles.macbookBuiltIn.id, for: "us.zoom.xos")

        let obs = AppProfileObserver(
            overrides: overrides,
            defaultProfileId: BuiltinProfiles.externalTop.id,
            store: store
        )
        // Calling setOverrides with same data shouldn't crash or loop
        obs.setOverrides(overrides)
        obs.setOverrides(overrides)
    }

    func testStopDoesNotCrashBeforeStart() {
        let store = ControlStateStore(initial: ControlState())
        let overrides = AppProfileOverrides()
        let obs = AppProfileObserver(
            overrides: overrides,
            defaultProfileId: BuiltinProfiles.externalTop.id,
            store: store
        )
        obs.stop()  // never started
    }
}
