import AppKit
import Foundation

/// Owns the NSStatusItem, icon, and popover attach point. Main-actor
/// isolated because it touches AppKit.
@MainActor
public final class MenuBarController {
    private let statusItem: NSStatusItem
    private var currentState: MenuBarIconState = .off
    private var pulseTimer: Timer?

    public var onClick: (() -> Void)?

    public init() {
        self.statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        configureButton()
        applyState(.off)
    }

    public func setState(_ state: MenuBarIconState) {
        guard state != currentState else { return }
        currentState = state
        applyState(state)
    }

    private func configureButton() {
        guard let button = statusItem.button else { return }
        button.action = #selector(buttonClicked(_:))
        button.target = self
    }

    private func applyState(_ state: MenuBarIconState) {
        guard let button = statusItem.button else { return }
        button.image = NSImage(systemSymbolName: state.symbolName, accessibilityDescription: "GazeLock")
        button.contentTintColor = state.tint

        pulseTimer?.invalidate()
        pulseTimer = nil
        if state.shouldPulse {
            startPulse(button)
        } else {
            button.alphaValue = 1.0
        }
    }

    private func startPulse(_ button: NSStatusBarButton) {
        var alpha: CGFloat = 1.0
        var direction: CGFloat = -0.05
        pulseTimer = Timer.scheduledTimer(withTimeInterval: 0.08, repeats: true) { _ in
            alpha += direction
            if alpha <= 0.6 { direction = 0.05 }
            if alpha >= 1.0 { direction = -0.05 }
            button.alphaValue = alpha
        }
    }

    @objc private func buttonClicked(_ sender: Any?) {
        onClick?()
    }

    public var button: NSStatusBarButton? {
        statusItem.button
    }
}
