import SwiftUI
import AppKit

public struct DashboardSection: View {
    @Bindable var store: ControlStateStore
    @Binding var beforeImage: NSImage?
    @Binding var afterImage: NSImage?

    public init(
        store: ControlStateStore,
        beforeImage: Binding<NSImage?>,
        afterImage: Binding<NSImage?>
    ) {
        self.store = store
        self._beforeImage = beforeImage
        self._afterImage = afterImage
    }

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: MainWindowStyle.sectionSpacing) {
                MainWindowStyle.sectionTitle("Live Preview")
                LivePreview(beforeImage: $beforeImage, afterImage: $afterImage)
                    .frame(height: 200)

                MainWindowStyle.sectionTitle("Correction Status")
                HStack(spacing: 24) {
                    stat("PROFILE", BuiltinProfiles.byId(store.state.setupProfileId).name)
                    stat("INTENSITY", String(format: "%.0f%%", store.state.intensity * 100))
                    stat("SENSITIVITY", store.state.sensitivity.rawValue.capitalized)
                    stat("AIM", String(format: "%+.1f°", store.state.verticalAimDeg))
                }

                Spacer()
            }
            .padding(MainWindowStyle.contentPadding)
            .frame(maxWidth: .infinity, alignment: .topLeading)
        }
    }

    private func stat(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(.system(size: 9, weight: .medium))
                .tracking(1.5)
                .foregroundStyle(PopoverStyle.textTertiary)
            Text(value)
                .font(.system(size: 16, weight: .semibold, design: .monospaced))
                .foregroundStyle(PopoverStyle.accent)
        }
    }
}
