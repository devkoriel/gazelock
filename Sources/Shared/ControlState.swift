import Foundation

/// State that drives the pipeline. Lives in both the main app and the
/// Camera Extension — the main app owns the authoritative copy; the
/// extension receives updates via XPC and caches the latest.
///
/// Keep this type small — every field is serialised across process
/// boundaries on every update.
public struct ControlState: Codable, Sendable, Equatable {
    /// Is gaze correction active?
    public var isEnabled: Bool

    /// [0, 1] — scales the target-gaze displacement pre-warp.
    /// 0 = passthrough, 1 = full correction.
    public var intensity: Double

    /// Which physical camera to capture from.
    public var sourceCameraUniqueID: String?

    /// Active preset name (for Phase 3c's main-window tabs). Phase 3b
    /// carries the field but does not read it.
    public var activePresetName: String

    public init(
        isEnabled: Bool = true,
        intensity: Double = 0.7,
        sourceCameraUniqueID: String? = nil,
        activePresetName: String = "Default"
    ) {
        self.isEnabled = isEnabled
        self.intensity = intensity
        self.sourceCameraUniqueID = sourceCameraUniqueID
        self.activePresetName = activePresetName
    }

    public static let `default` = ControlState()
}
