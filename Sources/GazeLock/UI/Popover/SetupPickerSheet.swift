import SwiftUI

/// Modal sheet listing all built-in SetupProfiles. Selecting one
/// calls the callback and dismisses.
public struct SetupPickerSheet: View {
    let current: String
    let onSelect: (SetupProfile) -> Void

    @Environment(\.dismiss) private var dismiss

    public init(current: String, onSelect: @escaping (SetupProfile) -> Void) {
        self.current = current
        self.onSelect = onSelect
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("YOUR SETUP")
                    .font(PopoverStyle.labelFont())
                    .tracking(1.5)
                    .foregroundStyle(PopoverStyle.textSecondary)
                Spacer()
                Button("Done") { dismiss() }
                    .buttonStyle(.plain)
                    .font(PopoverStyle.displayFont(size: 11))
                    .foregroundStyle(PopoverStyle.accent)
            }
            .padding(.horizontal, PopoverStyle.spacingLoose)
            .padding(.top, PopoverStyle.spacingLoose)
            .padding(.bottom, PopoverStyle.spacing)

            ForEach(BuiltinProfiles.all) { profile in
                Button {
                    onSelect(profile)
                    dismiss()
                } label: {
                    HStack {
                        Text(profile.name)
                            .font(PopoverStyle.displayFont(size: 12))
                            .foregroundStyle(PopoverStyle.textPrimary)
                        Spacer()
                        if profile.id == current {
                            Image(systemName: "checkmark")
                                .foregroundStyle(PopoverStyle.accent)
                        }
                    }
                    .padding(.horizontal, PopoverStyle.spacingLoose)
                    .padding(.vertical, PopoverStyle.spacing)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)

                if profile.id != BuiltinProfiles.all.last?.id {
                    Divider().overlay(PopoverStyle.border)
                }
            }

            Spacer(minLength: PopoverStyle.spacingLoose)
        }
        .frame(width: PopoverStyle.popoverWidth)
        .background(PopoverStyle.backgroundPrimary)
    }
}
