import SwiftUI
import AVFoundation
import SystemExtensions

public enum OnboardingStep: Int, CaseIterable {
    case welcome, installExtension, cameraPermission, pickSetup, done
}

public enum ExtensionInstallState: Equatable {
    case idle
    case submitted
    case waitingForUserApproval
    case activated
    case failed(String)
}

@MainActor
public final class OnboardingCoordinator: ObservableObject {
    @Published public var step: OnboardingStep = .welcome
    @Published public var extensionActivated: Bool = false
    @Published public var extInstallState: ExtensionInstallState = .idle
    @Published public var cameraAuthorized: Bool = false
    @Published public var pickedProfileId: String = BuiltinProfiles.externalTop.id

    public init() {}

    public var canAdvance: Bool {
        switch step {
        case .welcome: return true
        case .installExtension: return extensionActivated
        case .cameraPermission: return cameraAuthorized
        case .pickSetup: return true
        case .done: return true
        }
    }

    public func advance() {
        guard canAdvance else { return }
        if let next = OnboardingStep(rawValue: step.rawValue + 1) {
            step = next
        }
    }
}

public struct OnboardingSheet: View {
    @StateObject private var coord: OnboardingCoordinator
    let onFinish: (_ pickedProfileId: String) -> Void
    let onRequestExtensionInstall: () -> Void
    let onRequestCameraPermission: () async -> Bool

    public init(
        coordinator: OnboardingCoordinator = OnboardingCoordinator(),
        onFinish: @escaping (String) -> Void,
        onRequestExtensionInstall: @escaping () -> Void,
        onRequestCameraPermission: @escaping () async -> Bool
    ) {
        _coord = StateObject(wrappedValue: coordinator)
        self.onFinish = onFinish
        self.onRequestExtensionInstall = onRequestExtensionInstall
        self.onRequestCameraPermission = onRequestCameraPermission
    }

    public var body: some View {
        VStack(spacing: 24) {
            Spacer()
            content
            Spacer()
            Divider().overlay(PopoverStyle.border)
            controls
        }
        .padding(32)
        .frame(width: 560, height: 440)
        .background(PopoverStyle.backgroundPrimary)
    }

    @ViewBuilder private var content: some View {
        switch coord.step {
        case .welcome:
            OnboardingWelcome()
        case .installExtension:
            OnboardingExtension(coord: coord, onRequestInstall: onRequestExtensionInstall)
        case .cameraPermission:
            OnboardingCamera(coord: coord, onRequest: onRequestCameraPermission)
        case .pickSetup:
            OnboardingSetup(coord: coord)
        case .done:
            OnboardingDone()
        }
    }

    @ViewBuilder private var controls: some View {
        HStack {
            Spacer()
            if coord.step == .done {
                Button("Finish") { onFinish(coord.pickedProfileId) }
                    .buttonStyle(.borderedProminent)
                    .tint(PopoverStyle.accent)
            } else {
                Button("Next") {
                    coord.advance()
                }
                .buttonStyle(.borderedProminent)
                .tint(PopoverStyle.accent)
                .disabled(!coord.canAdvance)
            }
        }
    }
}
