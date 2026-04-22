import Foundation

/// NSXPC service protocol. The main app hosts; the Camera Extension
/// subscribes.
///
/// Methods are deliberately minimal — control state is a blob, no
/// per-field RPC. Replies are async via completion handlers (NSXPC
/// does not support Swift concurrency directly).
@objc public protocol ControlServiceProtocol {
    /// Extension asks the main app for the current control state on
    /// startup or after a reconnect.
    func fetchControlState(reply: @escaping (Data) -> Void)

    /// Main app pushes a new control state to the extension. Payload
    /// is a JSON-encoded `ControlState` blob.
    func pushControlState(_ payload: Data, reply: @escaping (Bool) -> Void)

    /// Heartbeat so the extension can detect a dead connection.
    func ping(reply: @escaping (Bool) -> Void)
}

public enum ControlServiceConstants {
    /// Mach service name — must match between server + client + Info.plist.
    public static let machServiceName = "com.gazelock.GazeLock.ControlService"
}
