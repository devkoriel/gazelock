import Foundation

/// Extension-side NSXPC client. Receives pushed state updates from the
/// main app's ControlServer (Phase 3c push-XPC, spec §8.11). Still
/// falls back to `fetchControlState` on initial connect.
public final class ControlClient: NSObject, @unchecked Sendable {
    private var connection: NSXPCConnection?
    private var cachedState = ControlState.default
    private let queue = DispatchQueue(label: "com.gazelock.GazeLock.ControlClient.cache")
    private let retryQueue = DispatchQueue(label: "com.gazelock.GazeLock.ControlClient.retry")
    private var retryTimer: DispatchSourceTimer?

    public override init() { super.init() }

    public func currentState() -> ControlState {
        queue.sync { cachedState }
    }

    public func start() {
        attemptConnection()
        scheduleRetry()
    }

    public func stop() {
        retryTimer?.cancel()
        retryTimer = nil
        connection?.invalidate()
        connection = nil
    }

    private func attemptConnection() {
        let conn = NSXPCConnection(
            machServiceName: ControlServiceConstants.machServiceName,
            options: []
        )
        conn.remoteObjectInterface = NSXPCInterface(with: ControlServiceProtocol.self)
        // Expose our side of the protocol so the server can push to us.
        conn.exportedInterface = NSXPCInterface(with: ControlServiceProtocol.self)
        conn.exportedObject = self

        conn.interruptionHandler = { [weak self] in
            self?.connection = nil
        }
        conn.invalidationHandler = { [weak self] in
            self?.connection = nil
        }
        conn.resume()
        self.connection = conn

        guard let proxy = conn.remoteObjectProxy as? ControlServiceProtocol else {
            return
        }
        proxy.fetchControlState { [weak self] data in
            guard let self else { return }
            guard let state = try? JSONDecoder().decode(ControlState.self, from: data) else {
                return
            }
            self.queue.sync {
                self.cachedState = state
            }
        }
    }

    private func scheduleRetry() {
        let timer = DispatchSource.makeTimerSource(queue: retryQueue)
        timer.schedule(deadline: .now() + 5, repeating: 5.0)
        timer.setEventHandler { [weak self] in
            guard let self else { return }
            if self.connection == nil {
                self.attemptConnection()
            }
        }
        timer.resume()
        retryTimer = timer
    }
}

extension ControlClient: ControlServiceProtocol {
    public func fetchControlState(reply: @escaping (Data) -> Void) {
        // Server shouldn't need to ask the extension for state; ack empty.
        reply(Data())
    }

    public func pushControlState(_ payload: Data, reply: @escaping (Bool) -> Void) {
        guard let state = try? JSONDecoder().decode(ControlState.self, from: payload) else {
            reply(false)
            return
        }
        queue.sync {
            self.cachedState = state
        }
        reply(true)
    }

    public func ping(reply: @escaping (Bool) -> Void) {
        reply(true)
    }
}
