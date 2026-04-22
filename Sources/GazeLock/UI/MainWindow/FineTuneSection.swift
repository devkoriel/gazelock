import SwiftUI

public struct FineTuneSection: View {
    @Bindable var store: ControlStateStore

    public init(store: ControlStateStore) {
        self.store = store
    }

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: MainWindowStyle.sectionSpacing) {
                MainWindowStyle.sectionTitle("Correction")
                field("Intensity", Binding(
                    get: { store.state.intensity },
                    set: { store.setIntensity($0) }
                ), in: 0...1, format: "%.0f%%", scale: 100)

                MainWindowStyle.sectionTitle("Aim")
                field("Vertical aim", Binding(
                    get: { store.state.verticalAimDeg },
                    set: { store.setVerticalAimDeg($0) }
                ), in: -10...10, format: "%+.1f°")
                field("Horizontal aim", Binding(
                    get: { store.state.horizontalAimDeg },
                    set: { store.setHorizontalAimDeg($0) }
                ), in: -10...10, format: "%+.1f°")

                MainWindowStyle.sectionTitle("Disengagement")
                HStack {
                    Text("Sensitivity")
                        .frame(width: 160, alignment: .leading)
                        .foregroundStyle(PopoverStyle.textSecondary)
                    Picker("", selection: Binding(
                        get: { store.state.sensitivity },
                        set: { store.setSensitivity($0) }
                    )) {
                        Text("Loose").tag(Sensitivity.loose)
                        Text("Normal").tag(Sensitivity.normal)
                        Text("Tight").tag(Sensitivity.tight)
                    }
                    .pickerStyle(.segmented)
                    .frame(width: 240)
                    Spacer()
                }

                Spacer()
            }
            .padding(MainWindowStyle.contentPadding)
            .frame(maxWidth: .infinity, alignment: .topLeading)
        }
    }

    private func field(
        _ label: String,
        _ binding: Binding<Double>,
        in range: ClosedRange<Double>,
        format: String,
        scale: Double = 1.0
    ) -> some View {
        HStack {
            Text(label)
                .frame(width: 160, alignment: .leading)
                .foregroundStyle(PopoverStyle.textSecondary)
            Slider(value: binding, in: range)
                .tint(PopoverStyle.accent)
            Text(String(format: format, binding.wrappedValue * scale))
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(PopoverStyle.accent)
                .frame(width: 60, alignment: .trailing)
        }
    }
}
