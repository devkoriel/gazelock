import Foundation

/// Computes per-frame correction-intensity multiplier alpha ∈ [0, 1]
/// from head pose with IIR smoothing + re-engage lockout. Spec §6.8.
///
/// Algorithm:
/// 1. Map |yaw| and |pitch| through a smoothstep over the sensitivity's
///    transition band → instantaneous pose-based disengage factor.
/// 2. IIR-smooth across frames (mix = 0.15 → ~150 ms time constant at 60 fps).
/// 3. Below 0.1 smoothed = fully disengaged — stamp the time.
/// 4. Lockout: until `reEngageLockoutMs` elapsed from last disengage stamp,
///    force alpha = 0, preventing snap-back oscillation.
public final class LookAwayDetector {
    private var smoothedAlpha: Double = 1.0
    private var disengagedAtMs: Double?

    public var sensitivity: Sensitivity

    public init(sensitivity: Sensitivity = .normal) {
        self.sensitivity = sensitivity
    }

    /// `timestampSeconds` is the frame capture time (AVCapture PTS in seconds).
    public func compute(headPose: HeadPose, timestampSeconds: TimeInterval) -> Double {
        let yawDeg = abs(headPose.yaw * 180.0 / .pi)
        let pitchDeg = abs(headPose.pitch * 180.0 / .pi)

        let yawFactor = smoothstep(sensitivity.yawTransitionDeg.low, sensitivity.yawTransitionDeg.high, yawDeg)
        let pitchFactor = smoothstep(sensitivity.pitchTransitionDeg.low, sensitivity.pitchTransitionDeg.high, pitchDeg)
        let posePenalty = max(yawFactor, pitchFactor)
        let instantAlpha = 1.0 - posePenalty

        let mix = 0.15
        smoothedAlpha = smoothedAlpha * (1.0 - mix) + instantAlpha * mix

        let nowMs = timestampSeconds * 1000.0
        let fullyDisengaged = smoothedAlpha < 0.1

        if fullyDisengaged {
            disengagedAtMs = nowMs
            return 0.0
        }

        if let t = disengagedAtMs, (nowMs - t) < sensitivity.reEngageLockoutMs {
            return 0.0
        }

        return smoothedAlpha
    }

    public func reset() {
        smoothedAlpha = 1.0
        disengagedAtMs = nil
    }
}

@inlinable
func smoothstep(_ edge0: Double, _ edge1: Double, _ x: Double) -> Double {
    let t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)
}
