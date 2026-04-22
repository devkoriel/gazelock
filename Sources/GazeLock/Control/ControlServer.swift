import Foundation

/// Hosts the NSXPC service the Camera Extension subscribes to.
///
/// The server owns the authoritative `ControlState` mirror (it's
/// the main app's `ControlStateStore`'s state). When the extension
/// calls `fetchControlState`, we return the current state; when the
/// main app's store changes, `broadcast(_:)` updates the cache and
/// the extension will pick up the new state on its next poll.
public final class ControlServer: NSObject {
    private let listener: NSXPCListener
    private let queue = DispatchQueue(label: "com.gazelock.GazeLock.ControlServer")

    /// Provided by the caller; returns the current authoritative state.
    private let currentStateProvider: () -> ControlState

    public init(currentStateProvider: @escaping () -> ControlState) {
        self.listener = NSXPCListener(machServiceName: ControlServiceConstants.machServiceName)
        self.currentStateProvider = currentStateProvider
        super.init()
        listener.delegate = self
    }

    public func start() {
        listener.resume()
    }

    public func broadcast(_ state: ControlState) {
        // For Phase 3b the server is poll-based: the extension calls
        // fetchControlState on startup + periodically via its retry
        // timer. We don't need to actively push. Keeping this method
        // as the API surface so Phase 4a can switch to real-time push
        // without changing callers.
        _ = state
    }
}

extension ControlServer: NSXPCListenerDelegate {
    public func listener(_ listener: NSXPCListener, shouldAcceptNewConnection newConnection: NSXPCConnection) -> Bool {
        newConnection.exportedInterface = NSXPCInterface(with: ControlServiceProtocol.self)
        newConnection.exportedObject = self
        newConnection.resume()
        return true
    }
}

extension ControlServer: ControlServiceProtocol {
    public func fetchControlState(reply: @escaping (Data) -> Void) {
        let state = currentStateProvider()
        let data = (try? JSONEncoder().encode(state)) ?? Data()
        reply(data)
    }

    public func pushControlState(_ payload: Data, reply: @escaping (Bool) -> Void) {
        // Phase 3b: main app is authoritative, extension does not push.
        _ = payload
        reply(false)
    }

    public func ping(reply: @escaping (Bool) -> Void) {
        reply(true)
    }
}
