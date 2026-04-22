import SwiftUI

public struct OnboardingWelcome: View {
    public init() {}
    public var body: some View {
        VStack(spacing: 16) {
            Text("GAZELOCK")
                .font(.system(size: 12, weight: .semibold))
                .tracking(3)
                .foregroundStyle(PopoverStyle.accent)
            Text("Natural eye contact on every video call.")
                .font(.system(size: 20, weight: .semibold))
                .foregroundStyle(PopoverStyle.textPrimary)
                .multilineTextAlignment(.center)
            Text(
                "GazeLock corrects the mislocation between your camera and what you're looking "
                + "at on screen — so the person you're talking to feels naturally engaged."
            )
                .font(.system(size: 12))
                .foregroundStyle(PopoverStyle.textSecondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 440)
        }
    }
}
