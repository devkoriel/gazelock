import AppKit
import SwiftUI

public struct PopoverView: View {
    @Bindable var store: ControlStateStore
    @Bindable var previewState: PreviewState
    var onOpenWindow: () -> Void

    @State private var showingSetupSheet = false

    public init(
        store: ControlStateStore,
        previewState: PreviewState,
        onOpenWindow: @escaping () -> Void
    ) {
        self.store = store
        self.previewState = previewState
        self.onOpenWindow = onOpenWindow
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: PopoverStyle.spacing) {
            header
            LivePreview(beforeImage: $previewState.before, afterImage: $previewState.after)
            toggleRow
            intensityRow
            setupRow
            verticalAimRow
            sensitivityRow
            Divider().overlay(PopoverStyle.border)
            footer
        }
        .padding(PopoverStyle.spacingLoose)
        .frame(width: PopoverStyle.popoverWidth)
        .background(PopoverStyle.backgroundPrimary)
        .sheet(isPresented: $showingSetupSheet) {
            SetupPickerSheet(current: store.state.setupProfileId) { profile in
                store.setSetupProfileId(profile.id)
            }
        }
    }

    private var header: some View {
        HStack(spacing: PopoverStyle.spacingTight) {
            Circle()
                .fill(statusColor)
                .frame(width: 6, height: 6)
                .shadow(color: statusColor.opacity(0.6), radius: 3)
            Text("GAZELOCK")
                .font(PopoverStyle.displayFont(size: 12))
                .tracking(1.2)
                .foregroundStyle(PopoverStyle.textPrimary)
            Spacer()
        }
    }

    private var toggleRow: some View {
        HStack {
            Text("CORRECTION")
                .font(PopoverStyle.labelFont())
                .tracking(0.4)
                .foregroundStyle(PopoverStyle.textSecondary)
            Spacer()
            Toggle("", isOn: Binding(
                get: { store.state.isEnabled },
                set: { store.setEnabled($0) }
            ))
            .toggleStyle(.switch)
            .tint(PopoverStyle.accent)
        }
    }

    private var intensityRow: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("INTENSITY")
                    .font(PopoverStyle.monoFont())
                    .tracking(1.2)
                    .foregroundStyle(PopoverStyle.textTertiary)
                Spacer()
                Text(String(format: "%.0f", store.state.intensity * 100))
                    .font(PopoverStyle.monoFont())
                    .foregroundStyle(PopoverStyle.accent)
            }
            Slider(value: Binding(
                get: { store.state.intensity },
                set: { store.setIntensity($0) }
            ), in: 0...1)
            .tint(PopoverStyle.accent)
        }
    }

    private var footer: some View {
        HStack {
            Spacer()
            Button(action: onOpenWindow) {
                Text("OPEN WINDOW →")
                    .font(PopoverStyle.labelFont(size: 9))
                    .tracking(1.5)
                    .foregroundStyle(PopoverStyle.textTertiary)
            }
            .buttonStyle(.plain)
            Spacer()
        }
    }

    private var statusColor: Color {
        store.state.isEnabled ? PopoverStyle.accent : PopoverStyle.textTertiary
    }

    private var setupRow: some View {
        HStack {
            Text("SETUP")
                .font(PopoverStyle.labelFont())
                .tracking(0.4)
                .foregroundStyle(PopoverStyle.textSecondary)
            Spacer()
            Button {
                showingSetupSheet = true
            } label: {
                HStack(spacing: 4) {
                    Text(BuiltinProfiles.byId(store.state.setupProfileId).name)
                        .font(PopoverStyle.monoFont(size: 10))
                        .foregroundStyle(PopoverStyle.accent)
                    Image(systemName: "chevron.down")
                        .font(.system(size: 8, weight: .semibold))
                        .foregroundStyle(PopoverStyle.accent)
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(
                    RoundedRectangle(cornerRadius: 4)
                        .stroke(PopoverStyle.border, lineWidth: 1)
                )
            }
            .buttonStyle(.plain)
        }
    }

    private var verticalAimRow: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("VERTICAL AIM")
                    .font(PopoverStyle.monoFont())
                    .tracking(1.2)
                    .foregroundStyle(PopoverStyle.textTertiary)
                Spacer()
                Text(String(format: "%+.1f°", store.state.verticalAimDeg))
                    .font(PopoverStyle.monoFont())
                    .foregroundStyle(PopoverStyle.accent)
            }
            Slider(value: Binding(
                get: { store.state.verticalAimDeg },
                set: { store.setVerticalAimDeg($0) }
            ), in: -30...30, step: 0.5)
                .tint(PopoverStyle.accent)
        }
    }

    private var sensitivityRow: some View {
        HStack {
            Text("SENSITIVITY")
                .font(PopoverStyle.labelFont())
                .tracking(0.4)
                .foregroundStyle(PopoverStyle.textSecondary)
            Spacer()
            Picker("", selection: Binding(
                get: { store.state.sensitivity },
                set: { store.setSensitivity($0) }
            )) {
                Text("Loose").tag(Sensitivity.loose)
                Text("Normal").tag(Sensitivity.normal)
                Text("Tight").tag(Sensitivity.tight)
            }
            .pickerStyle(.segmented)
            .frame(width: 150)
        }
    }
}
