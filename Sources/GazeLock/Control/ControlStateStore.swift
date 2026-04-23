import Foundation
import Observation

/// Observable, main-actor-isolated container for the current `ControlState`.
/// SwiftUI views bind to it directly; mutations trigger the
/// "push to extension" side effect via the provided closure.
@MainActor
@Observable
public final class ControlStateStore {
    public private(set) var state: ControlState

    private let onPushNeeded: @MainActor (ControlState) -> Void

    public init(
        initial: ControlState = .default,
        onPushNeeded: @escaping @MainActor (ControlState) -> Void = { _ in }
    ) {
        self.state = initial
        self.onPushNeeded = onPushNeeded
    }

    public func setEnabled(_ value: Bool) {
        var new = state
        new.isEnabled = value
        state = new
        onPushNeeded(new)
    }

    public func setIntensity(_ value: Double) {
        var new = state
        new.intensity = min(max(value, 0), 1)
        state = new
        onPushNeeded(new)
    }

    public func setSourceCamera(uniqueID: String?) {
        var new = state
        new.sourceCameraUniqueID = uniqueID
        state = new
        onPushNeeded(new)
    }

    public func setSetupProfileId(_ id: String) {
        var new = state
        new.setupProfileId = id
        // When profile changes, cascade its defaults into aim/sensitivity.
        let profile = BuiltinProfiles.byId(id)
        new.verticalAimDeg = profile.defaultVerticalAimDeg
        new.horizontalAimDeg = profile.defaultHorizontalAimDeg
        new.sensitivity = profile.defaultSensitivity
        state = new
        onPushNeeded(new)
    }

    public func setVerticalAimDeg(_ deg: Double) {
        var new = state
        new.verticalAimDeg = min(max(deg, -30.0), 30.0)
        state = new
        onPushNeeded(new)
    }

    public func setHorizontalAimDeg(_ deg: Double) {
        var new = state
        new.horizontalAimDeg = min(max(deg, -30.0), 30.0)
        state = new
        onPushNeeded(new)
    }

    public func setSensitivity(_ value: Sensitivity) {
        var new = state
        new.sensitivity = value
        state = new
        onPushNeeded(new)
    }
}
