import SwiftUI
import AppKit

public struct AppProfilesSection: View {
    @Binding var overrides: AppProfileOverrides
    @State private var showingPicker = false

    public init(overrides: Binding<AppProfileOverrides>) {
        self._overrides = overrides
    }

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: MainWindowStyle.sectionSpacing) {
                MainWindowStyle.sectionTitle("Overrides")
                if overrides.bundleIdToProfileId.isEmpty {
                    emptyState
                } else {
                    table
                }
                HStack {
                    Spacer()
                    Button("Add override…") { showingPicker = true }
                        .buttonStyle(.borderedProminent)
                        .tint(PopoverStyle.accent)
                }

                Spacer()
            }
            .padding(MainWindowStyle.contentPadding)
            .frame(maxWidth: .infinity, alignment: .topLeading)
        }
        .sheet(isPresented: $showingPicker) {
            RunningAppsPicker { bundleId, _ in
                var mut = overrides
                mut.setProfile(BuiltinProfiles.externalTop.id, for: bundleId)
                overrides = mut
            }
        }
    }

    private var emptyState: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("No per-app overrides yet.")
                .foregroundStyle(PopoverStyle.textSecondary)
            Text("GazeLock will use your default profile for every app.")
                .font(.system(size: 10))
                .foregroundStyle(PopoverStyle.textTertiary)
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 10).fill(PopoverStyle.backgroundElevated))
    }

    private var table: some View {
        VStack(spacing: 0) {
            ForEach(Array(overrides.bundleIdToProfileId.keys.sorted()), id: \.self) { bundleId in
                row(bundleId: bundleId)
                if bundleId != overrides.bundleIdToProfileId.keys.sorted().last {
                    Divider().overlay(PopoverStyle.border)
                }
            }
        }
        .background(RoundedRectangle(cornerRadius: 10).fill(PopoverStyle.backgroundElevated))
    }

    private func row(bundleId: String) -> some View {
        let currentProfileId = overrides.bundleIdToProfileId[bundleId] ?? BuiltinProfiles.externalTop.id
        return HStack {
            Text(bundleId)
                .font(.system(size: 12, design: .monospaced))
                .foregroundStyle(PopoverStyle.textPrimary)
                .lineLimit(1)
            Spacer()
            Picker("", selection: Binding(
                get: { currentProfileId },
                set: { newId in
                    var mut = overrides
                    mut.setProfile(newId, for: bundleId)
                    overrides = mut
                }
            )) {
                ForEach(BuiltinProfiles.all) { p in
                    Text(p.name).tag(p.id)
                }
            }
            .frame(width: 280)
            .labelsHidden()
            Button {
                var mut = overrides
                mut.removeProfile(for: bundleId)
                overrides = mut
            } label: {
                Image(systemName: "trash")
            }
            .buttonStyle(.plain)
            .foregroundStyle(PopoverStyle.textTertiary)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }
}

struct RunningAppsPicker: View {
    let onSelect: (String, String) -> Void
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("PICK AN APP")
                    .font(.system(size: 11, weight: .semibold))
                    .tracking(1.5)
                    .foregroundStyle(PopoverStyle.textSecondary)
                Spacer()
                Button("Cancel") { dismiss() }
                    .buttonStyle(.plain)
                    .foregroundStyle(PopoverStyle.accent)
            }
            .padding()

            Divider().overlay(PopoverStyle.border)

            ScrollView {
                LazyVStack(alignment: .leading, spacing: 0) {
                    ForEach(runningApps(), id: \.bundleId) { item in
                        Button {
                            onSelect(item.bundleId, item.name)
                            dismiss()
                        } label: {
                            HStack {
                                Text(item.name)
                                    .font(.system(size: 12))
                                    .foregroundStyle(PopoverStyle.textPrimary)
                                Spacer()
                                Text(item.bundleId)
                                    .font(.system(size: 10, design: .monospaced))
                                    .foregroundStyle(PopoverStyle.textTertiary)
                            }
                            .padding(.horizontal, 16)
                            .padding(.vertical, 8)
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        Divider().overlay(PopoverStyle.border)
                    }
                }
            }
        }
        .frame(width: 520, height: 420)
        .background(PopoverStyle.backgroundPrimary)
    }

    private struct AppItem {
        let bundleId: String
        let name: String
    }

    private func runningApps() -> [AppItem] {
        NSWorkspace.shared.runningApplications
            .filter { $0.activationPolicy == .regular }
            .compactMap { app -> AppItem? in
                guard let bundleId = app.bundleIdentifier else { return nil }
                return AppItem(
                    bundleId: bundleId,
                    name: app.localizedName ?? bundleId
                )
            }
            .sorted { $0.name.lowercased() < $1.name.lowercased() }
    }
}
