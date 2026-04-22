import SwiftUI

public struct OnboardingDone: View {
    public init() {}
    public var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "checkmark.seal.fill")
                .font(.system(size: 48))
                .foregroundStyle(PopoverStyle.accent)
            Text("You're all set.")
                .font(.system(size: 20, weight: .semibold))
                .foregroundStyle(PopoverStyle.textPrimary)
            Text("Open Zoom, Meet, FaceTime, OBS — anything — and pick 'GazeLock Camera'.")
                .font(.system(size: 12))
                .foregroundStyle(PopoverStyle.textSecondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 440)
        }
    }
}
