import AppKit
import Foundation

/// Menu-bar icon state — spec §8.6.
public enum MenuBarIconState: Equatable {
    case off              // greyscale
    case onIdle           // solid blue; correction on but no consumer
    case onStreaming      // pulsing; actively streaming to a consumer
    case error            // red; camera missing or extension disabled

    /// Map each state to an SF Symbol + tint color for the status item.
    public var symbolName: String {
        switch self {
        case .off: return "eye"
        case .onIdle, .onStreaming: return "eye.fill"
        case .error: return "exclamationmark.triangle.fill"
        }
    }

    public var tint: NSColor {
        switch self {
        case .off: return NSColor.secondaryLabelColor
        case .onIdle: return NSColor.systemBlue
        case .onStreaming: return NSColor.systemGreen
        case .error: return NSColor.systemRed
        }
    }

    public var shouldPulse: Bool {
        if case .onStreaming = self { return true }
        return false
    }
}
