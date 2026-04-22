import AVFoundation
import AppKit
import Foundation

public enum AutoDetect {
    /// Propose a SetupProfile from the current camera + screen configuration.
    public static func proposeProfile(
        cameraName: String?,
        screenCount: Int
    ) -> SetupProfile {
        let name = (cameraName ?? "").lowercased()
        let isMacBookCam = name.contains("facetime hd camera")
            || name.contains("built-in")
            || name.contains("macbook")

        if isMacBookCam && screenCount <= 1 {
            return BuiltinProfiles.macbookBuiltIn
        }
        if isMacBookCam && screenCount >= 2 {
            return BuiltinProfiles.macbookWithExternal
        }
        return BuiltinProfiles.externalTop
    }

    /// Convenience: runs detection against the live environment.
    @MainActor
    public static func proposeProfileLive() -> SetupProfile {
        let camName = AVCaptureDevice.default(for: .video)?.localizedName
        let screenCount = NSScreen.screens.count
        return proposeProfile(cameraName: camName, screenCount: screenCount)
    }
}
