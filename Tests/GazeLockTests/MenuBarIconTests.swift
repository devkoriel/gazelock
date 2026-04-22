import XCTest
@testable import GazeLock

final class MenuBarIconTests: XCTestCase {
    func testStateSymbolMapping() {
        XCTAssertEqual(MenuBarIconState.off.symbolName, "eye")
        XCTAssertEqual(MenuBarIconState.onIdle.symbolName, "eye.fill")
        XCTAssertEqual(MenuBarIconState.onStreaming.symbolName, "eye.fill")
        XCTAssertEqual(MenuBarIconState.error.symbolName, "exclamationmark.triangle.fill")
    }

    func testOnlyStreamingPulses() {
        XCTAssertFalse(MenuBarIconState.off.shouldPulse)
        XCTAssertFalse(MenuBarIconState.onIdle.shouldPulse)
        XCTAssertTrue(MenuBarIconState.onStreaming.shouldPulse)
        XCTAssertFalse(MenuBarIconState.error.shouldPulse)
    }
}
