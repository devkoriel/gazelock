import Foundation

/// Runtime check for which features the current user is entitled to.
/// Phase 3b stub: always returns `.pro` so the UI shows everything.
/// Phase 4a replaces with StoreKit + license-key validation.
public final class EntitlementChecker {
    public static let shared = EntitlementChecker()

    private init() {}

    public func currentLevel() -> EntitlementLevel {
        .pro
    }

    public func isFeatureAvailable(_ feature: ProFeature) -> Bool {
        currentLevel() == .pro
    }
}

public enum ProFeature: String, CaseIterable {
    case multiAppProfiles
    case skinToneCalibration
    case priorityProcessing
    case advancedCustomization
    case perMonitorCalibration
}
