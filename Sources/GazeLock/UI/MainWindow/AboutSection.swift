import SwiftUI

public struct AboutSection: View {
    public init() {}
    public var body: some View {
        VStack {
            MainWindowStyle.sectionTitle("About")
            Text("Version / license / subscription (P3c.10)")
                .foregroundStyle(PopoverStyle.textSecondary)
        }
        .padding(MainWindowStyle.contentPadding)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }
}
