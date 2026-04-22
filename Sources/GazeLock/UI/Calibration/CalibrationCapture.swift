import Foundation
import Observation

/// Records N head-pose samples and returns the mean as a Centroid.
/// Caller pushes samples via `feed(headPose:)`; the capture self-terminates
/// after `framesTarget` have been collected and invokes the completion.
@MainActor
@Observable
public final class CalibrationCapture {
    public private(set) var isRecording: Bool = false
    public private(set) var framesCollected: Int = 0
    public private(set) var framesTarget: Int = 30

    private var samples: [UserCalibration.Centroid] = []
    private var onComplete: ((UserCalibration.Centroid) -> Void)?

    public init() {}

    public func startRecording(
        framesTarget: Int = 30,
        onComplete: @escaping (UserCalibration.Centroid) -> Void
    ) {
        self.framesTarget = framesTarget
        self.framesCollected = 0
        self.samples = []
        self.onComplete = onComplete
        self.isRecording = true
    }

    public func feed(headPose: HeadPose) {
        guard isRecording else { return }
        samples.append(.init(
            yawRad: headPose.yaw,
            pitchRad: headPose.pitch,
            rollRad: headPose.roll
        ))
        framesCollected += 1
        if framesCollected >= framesTarget {
            finish()
        }
    }

    public func cancel() {
        isRecording = false
        onComplete = nil
    }

    private func finish() {
        isRecording = false
        let n = Double(max(samples.count, 1))
        let mean = UserCalibration.Centroid(
            yawRad: samples.map(\.yawRad).reduce(0, +) / n,
            pitchRad: samples.map(\.pitchRad).reduce(0, +) / n,
            rollRad: samples.map(\.rollRad).reduce(0, +) / n
        )
        onComplete?(mean)
        onComplete = nil
    }
}
