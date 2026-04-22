import Foundation
import CoreMediaIO

final class CameraExtensionDevice: NSObject, CMIOExtensionDeviceSource {
    private(set) var device: CMIOExtensionDevice!
    private var streamSource: CameraExtensionStream!

    init(localizedName: String, deviceID: UUID) {
        super.init()
        device = CMIOExtensionDevice(
            localizedName: localizedName,
            deviceID: deviceID,
            legacyDeviceID: deviceID.uuidString,
            source: self
        )

        streamSource = CameraExtensionStream(localizedName: "\(localizedName) Stream")
        do {
            try device.addStream(streamSource.stream)
        } catch {
            fatalError("Failed to add camera stream: \(error)")
        }
    }

    var availableProperties: Set<CMIOExtensionProperty> {
        [.deviceTransportType, .deviceModel]
    }

    func deviceProperties(
        forProperties properties: Set<CMIOExtensionProperty>
    ) throws -> CMIOExtensionDeviceProperties {
        let state = CMIOExtensionDeviceProperties(dictionary: [:])
        if properties.contains(.deviceTransportType) {
            state.transportType = kIOAudioDeviceTransportTypeVirtual
        }
        if properties.contains(.deviceModel) {
            state.model = "GazeLock Virtual Camera v0.1"
        }
        return state
    }

    func setDeviceProperties(_ deviceProperties: CMIOExtensionDeviceProperties) throws {}
}
