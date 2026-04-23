import SwiftUI

/// Precision Dark design tokens — spec §8.3.
public enum PopoverStyle {
    // Colors
    public static let backgroundPrimary = Color(red: 0, green: 0, blue: 0)              // #000000
    public static let backgroundElevated = Color(red: 0x1C / 255.0, green: 0x1C / 255.0, blue: 0x1E / 255.0)
    public static let border = Color(red: 0x2A / 255.0, green: 0x2A / 255.0, blue: 0x2D / 255.0)
    public static let accent = Color(red: 0, green: 0xFF / 255.0, blue: 0x88 / 255.0)       // #00FF88

    public static let textPrimary = Color.white.opacity(0.95)
    public static let textSecondary = Color.white.opacity(0.65)
    public static let textTertiary = Color.white.opacity(0.45)

    // Typography
    public static func displayFont(size: CGFloat = 13) -> Font {
        .system(size: size, weight: .semibold, design: .default)
    }

    public static func monoFont(size: CGFloat = 11) -> Font {
        .system(size: size, weight: .regular, design: .monospaced)
    }

    public static func labelFont(size: CGFloat = 10) -> Font {
        .system(size: size, weight: .medium, design: .default)
    }

    // Spacing
    public static let spacingTight: CGFloat = 8
    public static let spacing: CGFloat = 12
    public static let spacingLoose: CGFloat = 16

    public static let popoverWidth: CGFloat = 260
    public static let cornerRadius: CGFloat = 10
}
