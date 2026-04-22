import Foundation

/// Extension-side NSXPC client. Connects to the main app's
/// ControlServer and caches the current ControlState. Thread-safe;
/// readers call `currentState()` from any queue.
public final class ControlClient {
    private var connection: NSXPCConnection?
    private var cachedState = ControlState.default
    private let queue = DispatchQueue(label: "com.gazelock.GazeLock.ControlClient.cache")
    private let retryQueue = DispatchQueue(label: "com.gazelock.GazeLock.ControlClient.retry")
    private var retryTimer: DispatchSourceTimer?

    public init() {}

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
