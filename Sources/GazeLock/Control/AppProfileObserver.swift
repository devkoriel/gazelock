import AppKit
import Foundation

/// Listens for app activation changes and swaps the active setup profile
/// based on AppProfileOverrides. Spec §8.10.
@MainActor
public final class AppProfileObserver {
    private var overrides: AppProfileOverrides
    private let store: ControlStateStore
    private let defaultProfileId: String
    private var observation: NSObjectProtocol?

    public init(
        overrides: AppProfileOverrides,
        defaultProfileId: String,
        store: ControlStateStore
    ) {
        self.overrides = overrides
        self.store = store
        self.defaultProfileId = defaultProfileId
    }

    public func setOverrides(_ overrides: AppProfileOverrides) {
        self.overrides = overrides
        if let current = NSWorkspace.shared.frontmostApplication?.bundleIdentifier {
            apply(for: current)
        }
    }

    public func start() {
        observation = NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.didActivateApplicationNotification,
            object: nil,
            queue: .main
        ) { [weak self] note in
            guard
                let self,
                let app = note.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication
            else { return }
            MainActor.assumeIsolated {
                self.apply(for: app.bundleIdentifier)
            }
        }
        // Apply for the currently-front app on startup
        if let current = NSWorkspace.shared.frontmostApplication?.bundleIdentifier {
            apply(for: current)
        }
    }

    public func stop() {
        if let observation {
            NSWorkspace.shared.notificationCenter.removeObserver(observation)
        }
        observation = nil
    }

    private func apply(for bundleId: String?) {
        let target = overrides.profileId(for: bundleId) ?? defaultProfileId
        if store.state.setupProfileId != target {
            store.setSetupProfileId(target)
        }
    }
}
