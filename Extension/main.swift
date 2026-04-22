import Foundation
import CoreMediaIO

let providerSource = CameraExtensionProvider(
    clientQueue: DispatchQueue(label: "com.gazelock.GazeLock.CameraExtension.clientQueue")
)
CMIOExtensionProvider.startService(provider: providerSource.provider)
CFRunLoopRun()
