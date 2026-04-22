import Foundation
import CoreMediaIO

final class CameraExtensionProvider: NSObject, CMIOExtensionProviderSource {
    private(set) var provider: CMIOExtensionProvider!
    private var deviceSource: CameraExtensionDevice!

    init(clientQueue: DispatchQueue) {
        super.init()
        provider = CMIOExtensionProvider(source: self, clientQueue: clientQueue)
        deviceSource = CameraExtensionDevice(
            localizedName: "GazeLock Camera",
            deviceID: UUID()
        )
        do {
            try provider.addDevice(deviceSource.device)
        } catch {
            fatalError("Failed to add GazeLock Camera device: \(error)")
        }
    }

    func connect(to client: CMIOExtensionClient) throws {}

    func disconnect(from client: CMIOExtensionClient) {}

    var availableProperties: Set<CMIOExtensionProperty> {
        [.providerManufacturer]
    }

    func providerProperties(
        forProperties properties: Set<CMIOExtensionProperty>
    ) throws -> CMIOExtensionProviderProperties {
        let state = CMIOExtensionProviderProperties(dictionary: [:])
        if properties.contains(.providerManufacturer) {
            state.manufacturer = "GazeLock contributors"
        }
        return state
    }

    func setProviderProperties(_ providerProperties: CMIOExtensionProviderProperties) throws {}
}
