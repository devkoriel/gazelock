import SwiftUI

public struct FineTuneSection: View {
    @Bindable var store: ControlStateStore
    public init(store: ControlStateStore) { self.store = store }
    public var body: some View {
        VStack {
            MainWindowStyle.sectionTitle("Fine-tune")
            Text("Full-size controls (P3c.7)")
                .foregroundStyle(PopoverStyle.textSecondary)
        }
        .padding(MainWindowStyle.contentPadding)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }
}
