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

    /// FourCC 'virt' — marks this as a virtual device (equivalent to
    /// `kIOAudioDeviceTransportTypeVirtual` from `<IOKit/audio/IOAudioTypes.h>`
    /// but written inline to avoid the IOKit import. The `Int` type matches
    /// `CMIOExtensionDeviceProperties.transportType` in the current SDK.
    private static let transportTypeVirtual: Int = 0x76697274

    var availableProperties: Set<CMIOExtensionProperty> {
        [.deviceTransportType, .deviceModel]
    }

    func deviceProperties(
        forProperties properties: Set<CMIOExtensionProperty>
    ) throws -> CMIOExtensionDeviceProperties {
        let state = CMIOExtensionDeviceProperties(dictionary: [:])
        if properties.contains(.deviceTransportType) {
            state.transportType = Self.transportTypeVirtual
        }
        if properties.contains(.deviceModel) {
            state.model = "GazeLock Virtual Camera v0.1"
        }
        return state
    }

    func setDeviceProperties(_ deviceProperties: CMIOExtensionDeviceProperties) throws {}
}
