import SwiftUI

public enum MainWindowSection: String, CaseIterable, Identifiable {
    case dashboard, setup, fineTune, appProfiles, about
    public var id: String { rawValue }
    public var title: String {
        switch self {
        case .dashboard: return "Dashboard"
        case .setup: return "Setup"
        case .fineTune: return "Fine-tune"
        case .appProfiles: return "App Profiles"
        case .about: return "About"
        }
    }
    public var symbol: String {
        switch self {
        case .dashboard: return "gauge"
        case .setup: return "camera"
        case .fineTune: return "slider.horizontal.3"
        case .appProfiles: return "app.badge"
        case .about: return "info.circle"
        }
    }
}

public struct MainWindow: View {
    @Bindable var store: ControlStateStore
    var calibrationStore: CalibrationStore
    @Binding var overrides: AppProfileOverrides
    @Bindable var previewState: PreviewState
    var onLaunchCalibration: () -> Void
    var onRunAutoDetect: () -> Void

    @State private var selection: MainWindowSection = .dashboard

    public init(
        store: ControlStateStore,
        calibrationStore: CalibrationStore,
        overrides: Binding<AppProfileOverrides>,
        previewState: PreviewState,
        onLaunchCalibration: @escaping () -> Void,
        onRunAutoDetect: @escaping () -> Void
    ) {
        self.store = store
        self.calibrationStore = calibrationStore
        self._overrides = overrides
        self.previewState = previewState
        self.onLaunchCalibration = onLaunchCalibration
        self.onRunAutoDetect = onRunAutoDetect
    }

    public var body: some View {
        NavigationSplitView {
            List(MainWindowSection.allCases, selection: $selection) { section in
                NavigationLink(value: section) {
                    Label(section.title, systemImage: section.symbol)
                        .font(.system(size: 13))
                }
                .tag(section)
            }
            .navigationSplitViewColumnWidth(MainWindowStyle.sidebarWidth)
            .background(PopoverStyle.backgroundPrimary)
        } detail: {
            detailView
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(PopoverStyle.backgroundPrimary)
        }
        .frame(
            minWidth: MainWindowStyle.minWindowWidth,
            idealWidth: MainWindowStyle.windowWidth,
            minHeight: MainWindowStyle.minWindowHeight,
            idealHeight: MainWindowStyle.windowHeight
        )
    }

    @ViewBuilder
    private var detailView: some View {
        switch selection {
        case .dashboard:
            DashboardSection(store: store, previewState: previewState)
        case .setup:
            SetupSection(
                store: store,
                calibrationStore: calibrationStore,
                onRunAutoDetect: onRunAutoDetect,
                onLaunchCalibration: onLaunchCalibration
            )
        case .fineTune:
            FineTuneSection(store: store)
        case .appProfiles:
            AppProfilesSection(overrides: $overrides)
        case .about:
            AboutSection()
        }
    }
}
