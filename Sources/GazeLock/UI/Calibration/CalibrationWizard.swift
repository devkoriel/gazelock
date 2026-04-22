import SwiftUI
import AppKit

public enum CalibrationStage: Int {
    case intro
    case camera
    case primary
    case secondary
    case summary
}

public struct CalibrationWizard: View {
    @Bindable var capture: CalibrationCapture
    let secondaryScreenNames: [String]
    let baseProfileId: String
    let onSave: (UserCalibration) -> Void
    let onCancel: () -> Void

    @State private var stage: CalibrationStage = .intro
    @State private var cameraCentroid: UserCalibration.Centroid?
    @State private var primaryCentroid: UserCalibration.Centroid?
    @State private var secondaryCentroids: [UserCalibration.Centroid] = []
    @State private var currentSecondaryIndex: Int = 0

    public init(
        capture: CalibrationCapture,
        secondaryScreenNames: [String],
        baseProfileId: String,
        onSave: @escaping (UserCalibration) -> Void,
        onCancel: @escaping () -> Void
    ) {
        self.capture = capture
        self.secondaryScreenNames = secondaryScreenNames
        self.baseProfileId = baseProfileId
        self.onSave = onSave
        self.onCancel = onCancel
    }

    public var body: some View {
        VStack(spacing: 24) {
            Spacer()
            content
            Spacer()
            if stage != .intro && stage != .summary {
                progress
            }
            Divider().overlay(PopoverStyle.border)
            controls
        }
        .padding(32)
        .frame(width: 520, height: 420)
        .background(PopoverStyle.backgroundPrimary)
    }

    @ViewBuilder private var content: some View {
        switch stage {
        case .intro:
            VStack(spacing: 12) {
                Text("CALIBRATION")
                    .font(.system(size: 11, weight: .semibold))
                    .tracking(2)
                    .foregroundStyle(PopoverStyle.accent)
                Text("Let's record where your head points for each display.")
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(PopoverStyle.textPrimary)
                    .multilineTextAlignment(.center)
                Text("You'll look at your camera and each screen for a couple seconds while we capture head-pose samples.")
                    .font(.system(size: 12))
                    .foregroundStyle(PopoverStyle.textSecondary)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 380)
            }
        case .camera:
            recordingView(prompt: "Look at your camera")
        case .primary:
            recordingView(prompt: "Look at the center of your main screen")
        case .secondary:
            let name = secondaryScreenNames.indices.contains(currentSecondaryIndex)
                ? secondaryScreenNames[currentSecondaryIndex]
                : "Screen \(currentSecondaryIndex + 2)"
            recordingView(prompt: "Look at \(name)")
        case .summary:
            summaryView
        }
    }

    private func recordingView(prompt: String) -> some View {
        VStack(spacing: 16) {
            Text(prompt)
                .font(.system(size: 18, weight: .semibold))
                .foregroundStyle(PopoverStyle.textPrimary)
                .multilineTextAlignment(.center)
            if capture.isRecording {
                ProgressView(
                    value: Double(capture.framesCollected),
                    total: Double(capture.framesTarget)
                )
                .tint(PopoverStyle.accent)
                .frame(width: 300)
                Text("Hold steady…")
                    .font(.system(size: 11))
                    .foregroundStyle(PopoverStyle.textTertiary)
            } else {
                Text("Press Start when ready.")
                    .font(.system(size: 12))
                    .foregroundStyle(PopoverStyle.textSecondary)
            }
        }
    }

    private var summaryView: some View {
        VStack(spacing: 12) {
            Image(systemName: "checkmark.seal.fill")
                .font(.system(size: 48))
                .foregroundStyle(PopoverStyle.accent)
            Text("CALIBRATION COMPLETE")
                .font(.system(size: 11, weight: .semibold))
                .tracking(2)
                .foregroundStyle(PopoverStyle.accent)
            Text("\(1 + secondaryCentroids.count) screen(s) recorded")
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(PopoverStyle.textPrimary)
        }
    }

    private var progress: some View {
        HStack(spacing: 8) {
            ForEach([CalibrationStage.camera, .primary, .secondary], id: \.rawValue) { s in
                Circle()
                    .fill(s.rawValue <= stage.rawValue ? PopoverStyle.accent : PopoverStyle.border)
                    .frame(width: 6, height: 6)
            }
        }
    }

    @ViewBuilder private var controls: some View {
        HStack {
            Button("Cancel", role: .cancel) {
                capture.cancel()
                onCancel()
            }
            Spacer()
            switch stage {
            case .intro:
                Button("Begin") { stage = .camera }
                    .buttonStyle(.borderedProminent)
                    .tint(PopoverStyle.accent)
            case .camera, .primary, .secondary:
                if capture.isRecording {
                    Text("Recording…")
                        .foregroundStyle(PopoverStyle.textSecondary)
                } else {
                    Button("Start") { startCurrentStage() }
                        .buttonStyle(.borderedProminent)
                        .tint(PopoverStyle.accent)
                }
            case .summary:
                Button("Save") { saveAndClose() }
                    .buttonStyle(.borderedProminent)
                    .tint(PopoverStyle.accent)
            }
        }
    }

    private func startCurrentStage() {
        capture.startRecording { centroid in
            advance(with: centroid)
        }
    }

    private func advance(with centroid: UserCalibration.Centroid) {
        switch stage {
        case .intro: break
        case .camera:
            cameraCentroid = centroid
            stage = .primary
        case .primary:
            primaryCentroid = centroid
            if secondaryScreenNames.isEmpty {
                stage = .summary
            } else {
                currentSecondaryIndex = 0
                stage = .secondary
            }
        case .secondary:
            secondaryCentroids.append(centroid)
            currentSecondaryIndex += 1
            if currentSecondaryIndex >= secondaryScreenNames.count {
                stage = .summary
            }
        case .summary: break
        }
    }

    private func saveAndClose() {
        guard let cam = cameraCentroid, let primary = primaryCentroid else {
            onCancel()
            return
        }
        let cal = UserCalibration(
            baseProfileId: baseProfileId,
            cameraCentroid: cam,
            primaryScreenCentroid: primary,
            secondaryScreenCentroids: secondaryCentroids
        )
        onSave(cal)
    }
}
