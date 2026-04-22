import Foundation

/// Hosts the NSXPC service the Camera Extension subscribes to.
/// Bidirectional: server also pushes state updates to connected clients
/// whenever `broadcast(_:)` is called. Spec §8.11.
public final class ControlServer: NSObject {
    private let listener: NSXPCListener
    private let queue = DispatchQueue(label: "com.gazelock.GazeLock.ControlServer")
    private var connections: [NSXPCConnection] = []

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
        guard let data = try? JSONEncoder().encode(state) else { return }
        queue.sync {
            let valid = self.connections
            for conn in valid {
                (conn.remoteObjectProxy as? ControlServiceProtocol)?
                    .pushControlState(data) { _ in
                        // fire-and-forget; client ack is advisory
                    }
            }
        }
    }
}

extension ControlServer: NSXPCListenerDelegate {
    public func listener(
        _ listener: NSXPCListener,
        shouldAcceptNewConnection newConnection: NSXPCConnection
    ) -> Bool {
        newConnection.exportedInterface = NSXPCInterface(with: ControlServiceProtocol.self)
        newConnection.exportedObject = self
        // Server calls into client via remoteObjectProxy; same protocol.
        newConnection.remoteObjectInterface = NSXPCInterface(with: ControlServiceProtocol.self)

        newConnection.invalidationHandler = { [weak self, weak newConnection] in
            guard let self, let newConnection else { return }
            self.queue.sync {
                self.connections.removeAll { $0 === newConnection }
            }
        }
        newConnection.interruptionHandler = newConnection.invalidationHandler

        queue.sync {
            self.connections.append(newConnection)
        }
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
        // Phase 3c: main app is still authoritative. Client is not allowed
        // to push upstream. Ack false so misbehaving clients learn.
        _ = payload
        reply(false)
    }

    public func ping(reply: @escaping (Bool) -> Void) {
        reply(true)
    }
}
