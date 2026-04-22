import SwiftUI

public struct AppProfilesSection: View {
    @Binding var overrides: AppProfileOverrides
    public init(overrides: Binding<AppProfileOverrides>) { self._overrides = overrides }
    public var body: some View {
        VStack {
            MainWindowStyle.sectionTitle("App Profiles")
            Text("Per-app override table (P3c.9)")
                .foregroundStyle(PopoverStyle.textSecondary)
        }
        .padding(MainWindowStyle.contentPadding)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }
}
