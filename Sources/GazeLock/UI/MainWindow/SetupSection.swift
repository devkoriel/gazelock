import SwiftUI

public struct SetupSection: View {
    @Bindable var store: ControlStateStore
    var calibrationStore: CalibrationStore
    var onRunAutoDetect: () -> Void
    var onLaunchCalibration: () -> Void
    public init(
        store: ControlStateStore,
        calibrationStore: CalibrationStore,
        onRunAutoDetect: @escaping () -> Void,
        onLaunchCalibration: @escaping () -> Void
    ) {
        self.store = store
        self.calibrationStore = calibrationStore
        self.onRunAutoDetect = onRunAutoDetect
        self.onLaunchCalibration = onLaunchCalibration
    }
    public var body: some View {
        VStack {
            MainWindowStyle.sectionTitle("Setup")
            Text("Profile picker + auto-detect + calibrate (P3c.6)")
                .foregroundStyle(PopoverStyle.textSecondary)
        }
        .padding(MainWindowStyle.contentPadding)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }
}
