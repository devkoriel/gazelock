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
}
