import SwiftUI

public struct DashboardSection: View {
    @Bindable var store: ControlStateStore
    public init(store: ControlStateStore) { self.store = store }
    public var body: some View {
        VStack {
            MainWindowStyle.sectionTitle("Dashboard")
            Text("Live preview + stats (P3c.5)")
                .foregroundStyle(PopoverStyle.textSecondary)
        }
        .padding(MainWindowStyle.contentPadding)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }
}
