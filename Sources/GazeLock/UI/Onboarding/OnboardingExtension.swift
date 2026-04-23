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
            switch coord.extInstallState {
            case .idle:
                Button("Install…") { onRequestInstall() }
                    .buttonStyle(.borderedProminent)
                    .tint(PopoverStyle.accent)
                Text("You may need to approve in System Settings > Privacy & Security.")
                    .font(.system(size: 10))
                    .foregroundStyle(PopoverStyle.textTertiary)
            case .submitted:
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text("Installing…")
                        .foregroundStyle(PopoverStyle.textSecondary)
                }
            case .waitingForUserApproval:
                VStack(spacing: 8) {
                    Text("Waiting for your approval")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(PopoverStyle.accent)
                    Button("Open System Settings") {
                        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_SystemExtensions") {
                            NSWorkspace.shared.open(url)
                        }
                    }
                    .buttonStyle(.bordered)
                    Text("Click Allow at the bottom of Privacy & Security.")
                        .font(.system(size: 10))
                        .foregroundStyle(PopoverStyle.textTertiary)
                }
            case .activated:
                Label("Extension installed", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(PopoverStyle.accent)
            case .failed(let msg):
                VStack(spacing: 8) {
                    Label("Install failed", systemImage: "exclamationmark.triangle.fill")
                        .foregroundStyle(Color.red.opacity(0.9))
                    Text(msg)
                        .font(.system(size: 10))
                        .foregroundStyle(PopoverStyle.textTertiary)
                        .multilineTextAlignment(.center)
                        .frame(maxWidth: 420)
                    Button("Retry") { onRequestInstall() }
                        .buttonStyle(.bordered)
                }
            }
        }
    }
}
