import SwiftUI

public struct AboutSection: View {
    public init() {}

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: MainWindowStyle.sectionSpacing) {
                MainWindowStyle.sectionTitle("GazeLock")
                VStack(alignment: .leading, spacing: 6) {
                    let buildVer = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "0.3.0"
                    let buildNum = Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "dev"
                    Text("Version \(buildVer) (build \(buildNum))")
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundStyle(PopoverStyle.textPrimary)
                    Text("macOS \(ProcessInfo.processInfo.operatingSystemVersionString)")
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(PopoverStyle.textSecondary)
                }
                .padding(16)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(RoundedRectangle(cornerRadius: 10).fill(PopoverStyle.backgroundElevated))

                MainWindowStyle.sectionTitle("Subscription")
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("ENTITLEMENT")
                            .font(.system(size: 9, weight: .medium))
                            .tracking(1.5)
                            .foregroundStyle(PopoverStyle.textTertiary)
                        Text("PRO")
                            .font(.system(size: 10, weight: .semibold, design: .monospaced))
                            .tracking(1)
                            .foregroundStyle(PopoverStyle.accent)
                    }
                    Text("Real billing lands in Phase 4a (StoreKit + Shopify + PayPal).")
                        .font(.system(size: 11))
                        .foregroundStyle(PopoverStyle.textSecondary)
                }
                .padding(16)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(RoundedRectangle(cornerRadius: 10).fill(PopoverStyle.backgroundElevated))

                MainWindowStyle.sectionTitle("Licenses")
                VStack(alignment: .leading, spacing: 6) {
                    Text("MIT License — see LICENSE in the repository.")
                        .font(.system(size: 12))
                        .foregroundStyle(PopoverStyle.textSecondary)
                    Text("Third-party licenses: THIRD_PARTY_LICENSES.md (CC0 assets + MIT datasets).")
                        .font(.system(size: 11))
                        .foregroundStyle(PopoverStyle.textTertiary)
                }
                .padding(16)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(RoundedRectangle(cornerRadius: 10).fill(PopoverStyle.backgroundElevated))

                Spacer()
            }
            .padding(MainWindowStyle.contentPadding)
            .frame(maxWidth: .infinity, alignment: .topLeading)
        }
    }
}
