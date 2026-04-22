import SwiftUI

public struct OnboardingExtension: View {
    @ObservedObject var coord: OnboardingCoordinator
    let onRequestInstall: () -> Void

    public init(coord: OnboardingCoordinator, onRequestInstall: @escaping () -> Void) {
        self.coord = coord
        self.onRequestInstall = onRequestInstall
    }

    public var body: some View {
        VStack(spacing: 16) {
            Text("INSTALL VIRTUAL CAMERA")
                .font(.system(size: 11, weight: .semibold))
                .tracking(2)
                .foregroundStyle(PopoverStyle.accent)
            Text("GazeLock installs a system extension that shows up as 'GazeLock Camera' in Zoom, Meet, FaceTime, and any other app.")
                .font(.system(size: 12))
                .foregroundStyle(PopoverStyle.textSecondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 440)
            if coord.extensionActivated {
                Label("Extension installed", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(PopoverStyle.accent)
            } else {
                Button("Install…") { onRequestInstall() }
                    .buttonStyle(.borderedProminent)
                    .tint(PopoverStyle.accent)
                Text("You may need to approve this in System Settings > Privacy & Security.")
                    .font(.system(size: 10))
                    .foregroundStyle(PopoverStyle.textTertiary)
            }
        }
    }
}
