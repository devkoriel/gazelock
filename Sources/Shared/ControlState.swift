import Foundation

/// State that drives the pipeline. Lives in both the main app and the
/// Camera Extension — the main app owns the authoritative copy; the
/// extension receives updates via XPC and caches the latest.
///
/// Keep this type small — every field is serialised across process
/// boundaries on every update.
public struct ControlState: Codable, Sendable, Equatable {
    public var isEnabled: Bool
    public var intensity: Double
    public var sourceCameraUniqueID: String?
    public var activePresetName: String

    // Phase 3b.5: natural gaze adaptation
    public var setupProfileId: String
    public var verticalAimDeg: Double
    public var horizontalAimDeg: Double
    public var sensitivity: Sensitivity

    public init(
        isEnabled: Bool = true,
        intensity: Double = 0.7,
        sourceCameraUniqueID: String? = nil,
        activePresetName: String = "Default",
        setupProfileId: String = BuiltinProfiles.externalTop.id,
        verticalAimDeg: Double = -2.0,
        horizontalAimDeg: Double = 0.0,
        sensitivity: Sensitivity = .normal
    ) {
        self.isEnabled = isEnabled
        self.intensity = intensity
        self.sourceCameraUniqueID = sourceCameraUniqueID
        self.activePresetName = activePresetName
        self.setupProfileId = setupProfileId
        self.verticalAimDeg = verticalAimDeg
        self.horizontalAimDeg = horizontalAimDeg
        self.sensitivity = sensitivity
    }

    public static let `default` = ControlState()

    enum CodingKeys: String, CodingKey {
        case isEnabled, intensity, sourceCameraUniqueID, activePresetName
        case setupProfileId, verticalAimDeg, horizontalAimDeg, sensitivity
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        isEnabled = try c.decodeIfPresent(Bool.self, forKey: .isEnabled) ?? true
        intensity = try c.decodeIfPresent(Double.self, forKey: .intensity) ?? 0.7
        sourceCameraUniqueID = try c.decodeIfPresent(String.self, forKey: .sourceCameraUniqueID)
        activePresetName = try c.decodeIfPresent(String.self, forKey: .activePresetName) ?? "Default"
        setupProfileId = try c.decodeIfPresent(String.self, forKey: .setupProfileId) ?? BuiltinProfiles.externalTop.id
        verticalAimDeg = try c.decodeIfPresent(Double.self, forKey: .verticalAimDeg) ?? -2.0
        horizontalAimDeg = try c.decodeIfPresent(Double.self, forKey: .horizontalAimDeg) ?? 0.0
        sensitivity = try c.decodeIfPresent(Sensitivity.self, forKey: .sensitivity) ?? .normal
    }
}
