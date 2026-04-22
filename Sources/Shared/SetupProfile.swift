import Foundation

/// Camera-position profile. Encodes setup geometry as default aim deltas
/// plus a disengagement sensitivity baseline. Spec §6.8.
public struct SetupProfile: Codable, Sendable, Equatable, Identifiable {
    public let id: String
    public let name: String
    public let defaultVerticalAimDeg: Double
    public let defaultHorizontalAimDeg: Double
    public let defaultSensitivity: Sensitivity

    public init(
        id: String,
        name: String,
        defaultVerticalAimDeg: Double,
        defaultHorizontalAimDeg: Double,
        defaultSensitivity: Sensitivity
    ) {
        self.id = id
        self.name = name
        self.defaultVerticalAimDeg = defaultVerticalAimDeg
        self.defaultHorizontalAimDeg = defaultHorizontalAimDeg
        self.defaultSensitivity = defaultSensitivity
    }
}

public enum BuiltinProfiles {
    public static let macbookBuiltIn = SetupProfile(
        id: "macbook_builtin",
        name: "MacBook (built-in cam)",
        defaultVerticalAimDeg: -2.0,
        defaultHorizontalAimDeg: 0.0,
        defaultSensitivity: .tight
    )
    public static let externalTop = SetupProfile(
        id: "external_top_cam",
        name: "External monitor, webcam on top",
        defaultVerticalAimDeg: -2.0,
        defaultHorizontalAimDeg: 0.0,
        defaultSensitivity: .normal
    )
    public static let externalBottom = SetupProfile(
        id: "external_bottom_cam",
        name: "External monitor, webcam on bottom",
        defaultVerticalAimDeg: -2.0,
        defaultHorizontalAimDeg: 0.0,
        defaultSensitivity: .normal
    )
    public static let macbookWithExternal = SetupProfile(
        id: "macbook_plus_external",
        name: "MacBook + external monitor (cam on external top)",
        defaultVerticalAimDeg: -2.0,
        defaultHorizontalAimDeg: 0.0,
        defaultSensitivity: .normal
    )

    public static let all: [SetupProfile] = [
        macbookBuiltIn,
        externalTop,
        externalBottom,
        macbookWithExternal
    ]

    public static func byId(_ id: String) -> SetupProfile {
        all.first { $0.id == id } ?? externalTop
    }
}
