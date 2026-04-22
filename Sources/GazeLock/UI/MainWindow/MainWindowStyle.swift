import SwiftUI

/// Precision Dark tokens scaled for the main window (larger surfaces
/// than the popover). Matches PopoverStyle where applicable.
public enum MainWindowStyle {
    public static let windowWidth: CGFloat = 900
    public static let windowHeight: CGFloat = 600
    public static let minWindowWidth: CGFloat = 800
    public static let minWindowHeight: CGFloat = 500
    public static let sidebarWidth: CGFloat = 200
    public static let contentPadding: CGFloat = 24
    public static let sectionSpacing: CGFloat = 20

    public static func sectionTitle(_ text: String) -> some View {
        Text(text.uppercased())
            .font(.system(size: 11, weight: .semibold, design: .default))
            .tracking(1.8)
            .foregroundStyle(PopoverStyle.textSecondary)
    }
}
