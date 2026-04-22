import SwiftUI

public struct OnboardingSetup: View {
    @ObservedObject var coord: OnboardingCoordinator
    @State private var proposed: SetupProfile?

    public init(coord: OnboardingCoordinator) {
        self.coord = coord
    }

    public var body: some View {
        VStack(spacing: 16) {
            Text("PICK YOUR SETUP")
                .font(.system(size: 11, weight: .semibold))
                .tracking(2)
                .foregroundStyle(PopoverStyle.accent)
            if let p = proposed {
                VStack(spacing: 4) {
                    Text("Detected: \(p.name)")
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(PopoverStyle.textPrimary)
                    Button("Use this") {
                        coord.pickedProfileId = p.id
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(PopoverStyle.accent)
                }
            }
            Picker("Or pick manually", selection: $coord.pickedProfileId) {
                ForEach(BuiltinProfiles.all) { profile in
                    Text(profile.name).tag(profile.id)
                }
            }
            .pickerStyle(.menu)
            .frame(maxWidth: 420)
        }
        .onAppear {
            let detected = AutoDetect.proposeProfileLive()
            proposed = detected
            if coord.pickedProfileId == BuiltinProfiles.externalTop.id {
                coord.pickedProfileId = detected.id
            }
        }
    }
}
