import AppKit
import Foundation
import Observation

/// Observable holder for the live BEFORE/AFTER preview frames. SwiftUI
/// views bind to this and re-render on every frame update — replacing
/// the previous pattern where plain `var` properties on AppDelegate
/// silently skipped view invalidation.
@MainActor
@Observable
public final class PreviewState {
    public var before: NSImage?
    public var after: NSImage?

    public init() {}

    public func update(before: NSImage?, after: NSImage?) {
        self.before = before
        self.after = after
    }
}
