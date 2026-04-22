import SwiftUI

public struct SetupSection: View {
    @Bindable var store: ControlStateStore
    @Bindable var calibrationStore: CalibrationStore
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
        ScrollView {
            VStack(alignment: .leading, spacing: MainWindowStyle.sectionSpacing) {
                MainWindowStyle.sectionTitle("Active Setup")
                activeCard

                MainWindowStyle.sectionTitle("Built-in Profiles")
                profilesGrid

                MainWindowStyle.sectionTitle("Calibration")
                calibrationCard
            }
            .padding(MainWindowStyle.contentPadding)
            .frame(maxWidth: .infinity, alignment: .topLeading)
        }
    }

    private var activeCard: some View {
        let profile = BuiltinProfiles.byId(store.state.setupProfileId)
        return HStack {
            VStack(alignment: .leading, spacing: 8) {
                Text(profile.name)
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(PopoverStyle.textPrimary)
                Text("Default aim \(String(format: "%+.1f", profile.defaultVerticalAimDeg))° · sensitivity \(profile.defaultSensitivity.rawValue)")
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(PopoverStyle.textTertiary)
            }
            Spacer()
            Button("Auto-detect") { onRunAutoDetect() }
                .buttonStyle(.bordered)
        }
        .padding(16)
        .background(RoundedRectangle(cornerRadius: 10).fill(PopoverStyle.backgroundElevated))
    }

    private var profilesGrid: some View {
        LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
            ForEach(BuiltinProfiles.all) { profile in
                profileCard(profile)
            }
        }
    }

    private func profileCard(_ profile: SetupProfile) -> some View {
        let isActive = profile.id == store.state.setupProfileId
        return Button {
            store.setSetupProfileId(profile.id)
        } label: {
            HStack {
                Image(systemName: isActive ? "checkmark.circle.fill" : "circle")
                    .foregroundStyle(isActive ? PopoverStyle.accent : PopoverStyle.textTertiary)
                Text(profile.name)
                    .font(.system(size: 12))
                    .foregroundStyle(PopoverStyle.textPrimary)
                Spacer()
            }
            .padding(12)
            .background(RoundedRectangle(cornerRadius: 8).fill(PopoverStyle.backgroundElevated))
        }
        .buttonStyle(.plain)
    }

    private var calibrationCard: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                if let cal = calibrationStore.current {
                    Text("Calibrated \(cal.createdAt.formatted(.dateTime.day().month().year()))")
                        .font(.system(size: 12))
                        .foregroundStyle(PopoverStyle.textPrimary)
                    Text("\(cal.secondaryScreenCentroids.count + 1) screen(s) recorded")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(PopoverStyle.textTertiary)
                } else {
                    Text("No calibration yet")
                        .font(.system(size: 12))
                        .foregroundStyle(PopoverStyle.textSecondary)
                    Text("Multi-monitor disengagement uses yaw/pitch only")
                        .font(.system(size: 10))
                        .foregroundStyle(PopoverStyle.textTertiary)
                }
            }
            Spacer()
            Button("Calibrate") { onLaunchCalibration() }
                .buttonStyle(.borderedProminent)
                .tint(PopoverStyle.accent)
        }
        .padding(16)
        .background(RoundedRectangle(cornerRadius: 10).fill(PopoverStyle.backgroundElevated))
    }
}
