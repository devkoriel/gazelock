import Foundation

/// Look-away disengagement sensitivity. Maps to transition-band thresholds
/// used by `LookAwayDetector`. Spec §6.8.
public enum Sensitivity: String, Codable, Sendable, CaseIterable {
    case loose
    case normal
    case tight

    /// Yaw transition band in degrees (low, high).
    public var yawTransitionDeg: (low: Double, high: Double) {
        switch self {
        case .loose:  return (30, 40)
        case .normal: return (25, 30)
        case .tight:  return (18, 23)
        }
    }

    /// Pitch transition band in degrees (low, high).
    public var pitchTransitionDeg: (low: Double, high: Double) {
        switch self {
        case .loose:  return (18, 28)
        case .normal: return (15, 25)
        case .tight:  return (10, 18)
        }
    }

    /// Milliseconds to stay disengaged before allowing re-engage.
    public var reEngageLockoutMs: Double {
        switch self {
        case .loose:  return 200
        case .normal: return 300
        case .tight:  return 450
        }
    }
}
