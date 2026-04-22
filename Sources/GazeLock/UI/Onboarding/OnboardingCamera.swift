import SwiftUI

public struct OnboardingCamera: View {
    @ObservedObject var coord: OnboardingCoordinator
    let onRequest: () async -> Bool

    public init(coord: OnboardingCoordinator, onRequest: @escaping () async -> Bool) {
        self.coord = coord
        self.onRequest = onRequest
    }

    public var body: some View {
        VStack(spacing: 16) {
            Text("GRANT CAMERA ACCESS")
                .font(.system(size: 11, weight: .semibold))
                .tracking(2)
                .foregroundStyle(PopoverStyle.accent)
            Text("GazeLock reads from your webcam locally — nothing leaves your machine.")
                .font(.system(size: 12))
                .foregroundStyle(PopoverStyle.textSecondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 440)
            if coord.cameraAuthorized {
                Label("Camera access granted", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(PopoverStyle.accent)
            } else {
                Button("Grant access") {
                    Task {
                        let ok = await onRequest()
                        coord.cameraAuthorized = ok
                    }
                }
                .buttonStyle(.borderedProminent)
                .tint(PopoverStyle.accent)
            }
        }
    }
}
