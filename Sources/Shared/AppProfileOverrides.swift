import Foundation

/// Per-app profile overrides. Maps bundle identifiers to setup profile IDs.
/// Persisted as a UserDefaults dictionary. Spec §8.10.
public struct AppProfileOverrides: Codable, Sendable, Equatable {
    public var bundleIdToProfileId: [String: String]

    public init(_ map: [String: String] = [:]) {
        self.bundleIdToProfileId = map
    }

    public func profileId(for bundleId: String?) -> String? {
        guard let bundleId else { return nil }
        return bundleIdToProfileId[bundleId]
    }

    public mutating func setProfile(_ profileId: String, for bundleId: String) {
        bundleIdToProfileId[bundleId] = profileId
    }

    public mutating func removeProfile(for bundleId: String) {
        bundleIdToProfileId.removeValue(forKey: bundleId)
    }
}

public enum AppProfileOverridesDefaultsKey {
    public static let key = "AppProfileOverrides"
}
