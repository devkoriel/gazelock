import SwiftUI
import XCTest
@testable import GazeLock

@MainActor
final class PopoverViewSnapshotTests: XCTestCase {
    func testPopoverViewInstantiates() {
        let store = ControlStateStore(initial: .default)
        let view = PopoverView(
            store: store,
            previewState: PreviewState(),
            onOpenWindow: {}
        )
        // Sanity — hosting controller resolves without crashing.
        let controller = NSHostingController(rootView: view)
        XCTAssertNotNil(controller.view)
    }

    @MainActor
    func testPopoverRendersWithTightSensitivityAndAim() {
        let store = ControlStateStore(initial: ControlState(
            isEnabled: true,
            intensity: 0.5,
            activePresetName: "Default",
            setupProfileId: BuiltinProfiles.macbookBuiltIn.id,
            verticalAimDeg: 4.5,
            horizontalAimDeg: 0.0,
            sensitivity: .tight
        ))
        let view = PopoverView(
            store: store,
            previewState: PreviewState(),
            onOpenWindow: {}
        )
        let controller = NSHostingController(rootView: view)
        XCTAssertNotNil(controller.view)
    }
}
