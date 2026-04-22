import AppKit
import CoreVideo
import Foundation
import SwiftUI

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private var menuBarController: MenuBarController!
    private var popoverController: NSPopover!
    private var controlStore: ControlStateStore!
    private var controlServer: ControlServer!
    private var cameraCapture: CameraCapture!
    private var pipeline: FramePipeline!

    // Main window (Phase 3c)
    private var mainWindow: NSWindow?
    private var calibrationStore: CalibrationStore!
    private var appProfileObserver: AppProfileObserver!
    private var overrides = AppProfileOverrides()

    // Preview state shown in the popover
    private var previewBefore: NSImage?
    private var previewAfter: NSImage?

    // Calibration wizard (P3c.8)
    private var calibrationCapture: CalibrationCapture?
    private var calibrationHost: NSHostingController<CalibrationWizard>?
    private var calibrationDetector: LandmarkDetector?

    func applicationDidFinishLaunching(_ notification: Notification) {
        setupPipeline()
        setupControlPlane()
        setupMainWindowDependencies()
        setupMenuBar()
        setupPopover()
        startCapture()
    }

    private func setupMainWindowDependencies() {
        calibrationStore = CalibrationStore()

        if let data = UserDefaults.standard.data(forKey: AppProfileOverridesDefaultsKey.key),
           let saved = try? JSONDecoder().decode(AppProfileOverrides.self, from: data) {
            overrides = saved
        }
        appProfileObserver = AppProfileObserver(
            overrides: overrides,
            defaultProfileId: controlStore.state.setupProfileId,
            store: controlStore
        )
        appProfileObserver.start()
    }

    private func setupPipeline() {
        do {
            pipeline = try FramePipeline()
        } catch {
            presentFatal("Could not initialise frame pipeline: \(error)")
            return
        }
    }

    private func setupControlPlane() {
        // Authoritative state store; changes push to the extension.
        controlStore = ControlStateStore(
            initial: .default,
            onPushNeeded: { [weak self] newState in
                self?.controlServer?.broadcast(newState)
            }
        )
        controlServer = ControlServer { [weak self] in
            self?.controlStore.state ?? .default
        }
        controlServer.start()
    }

    private func setupMenuBar() {
        menuBarController = MenuBarController()
        menuBarController.setState(.onIdle)
        menuBarController.onClick = { [weak self] in
            self?.togglePopover()
        }
    }

    private func setupPopover() {
        popoverController = NSPopover()
        popoverController.behavior = .transient
        popoverController.contentSize = NSSize(width: PopoverStyle.popoverWidth, height: 250)
        let root = PopoverView(
            store: controlStore,
            beforeImage: Binding(
                get: { [weak self] in self?.previewBefore },
                set: { [weak self] in self?.previewBefore = $0 }
            ),
            afterImage: Binding(
                get: { [weak self] in self?.previewAfter },
                set: { [weak self] in self?.previewAfter = $0 }
            ),
            onOpenWindow: { [weak self] in
                self?.showMainWindow()
            }
        )
        popoverController.contentViewController = NSHostingController(rootView: root)
    }

    private func startCapture() {
        cameraCapture = CameraCapture()
        // Capture non-isolated references locally so the off-main sample
        // callback can use them without crossing actor isolation.
        // `FramePipeline` is a final class with no actor isolation; it
        // is safe to invoke from the capture queue.
        let pipelineRef = pipeline!
        cameraCapture.onFrame = { pb, ts in
            // Render BEFORE off-main (pure CoreImage → NSImage; safe).
            let before = PreviewFrameRenderer.nsImage(from: pb)
            // Hop to main only to read the latest state, then run
            // the pipeline off-main using the captured reference.
            Task { @MainActor [weak self] in
                guard let self else { return }
                // Feed calibration if active
                if let capture = self.calibrationCapture, capture.isRecording,
                   let detector = self.calibrationDetector,
                   let landmarks = try? detector.detect(in: pb, timestamp: ts) {
                    capture.feed(headPose: landmarks.headPoseRadians)
                }
                let intensity = self.controlStore.state.intensity
                let vAim = self.controlStore.state.verticalAimDeg
                let hAim = self.controlStore.state.horizontalAimDeg
                let sens = self.controlStore.state.sensitivity
                // Run pipeline work without blocking the main actor.
                let after: NSImage? = await Self.processOffMain(
                    pipeline: pipelineRef,
                    pixelBuffer: pb,
                    timestamp: ts,
                    intensity: intensity,
                    verticalAimDeg: vAim,
                    horizontalAimDeg: hAim,
                    sensitivity: sens
                )
                self.previewBefore = before
                self.previewAfter = after
            }
        }
        do {
            try cameraCapture.start()
        } catch {
            menuBarController.setState(.error)
            presentFatal("Could not start camera: \(error)")
        }
    }

    /// Runs the pipeline and renders an NSImage off the main actor.
    /// `FramePipeline` carries no actor isolation, so this is valid.
    private static func processOffMain(
        pipeline: FramePipeline,
        pixelBuffer: CVPixelBuffer,
        timestamp: TimeInterval,
        intensity: Double,
        verticalAimDeg: Double,
        horizontalAimDeg: Double,
        sensitivity: Sensitivity
    ) async -> NSImage? {
        let out: CVPixelBuffer = (try? pipeline.process(
            pixelBuffer: pixelBuffer,
            timestamp: timestamp,
            intensity: intensity,
            verticalAimDeg: verticalAimDeg,
            horizontalAimDeg: horizontalAimDeg,
            sensitivity: sensitivity
        )) ?? pixelBuffer
        return PreviewFrameRenderer.nsImage(from: out)
    }

    private func togglePopover() {
        guard let button = menuBarController.button else { return }
        if popoverController.isShown {
            popoverController.performClose(nil)
        } else {
            popoverController.show(relativeTo: button.bounds, of: button, preferredEdge: .minY)
        }
    }

    private func showMainWindow() {
        if mainWindow == nil {
            let overridesBinding = Binding<AppProfileOverrides>(
                get: { [weak self] in self?.overrides ?? AppProfileOverrides() },
                set: { [weak self] newVal in self?.setAppProfileOverrides(newVal) }
            )
            let root = MainWindow(
                store: controlStore,
                calibrationStore: calibrationStore,
                overrides: overridesBinding,
                beforeImage: Binding(
                    get: { [weak self] in self?.previewBefore },
                    set: { [weak self] in self?.previewBefore = $0 }
                ),
                afterImage: Binding(
                    get: { [weak self] in self?.previewAfter },
                    set: { [weak self] in self?.previewAfter = $0 }
                ),
                onLaunchCalibration: { [weak self] in self?.launchCalibration() },
                onRunAutoDetect: { [weak self] in self?.runAutoDetect() }
            )
            let window = NSWindow(
                contentRect: NSRect(
                    x: 0, y: 0,
                    width: MainWindowStyle.windowWidth,
                    height: MainWindowStyle.windowHeight
                ),
                styleMask: [.titled, .closable, .miniaturizable, .resizable],
                backing: .buffered,
                defer: false
            )
            window.title = "GazeLock"
            window.center()
            window.contentViewController = NSHostingController(rootView: root)
            window.isReleasedWhenClosed = false
            window.delegate = self
            mainWindow = window
        }
        NSApp.setActivationPolicy(.regular)
        mainWindow?.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    private func setAppProfileOverrides(_ newVal: AppProfileOverrides) {
        overrides = newVal
        appProfileObserver?.setOverrides(newVal)
        if let data = try? JSONEncoder().encode(newVal) {
            UserDefaults.standard.set(data, forKey: AppProfileOverridesDefaultsKey.key)
        }
    }

    private func launchCalibration() {
        let capture = CalibrationCapture()
        calibrationCapture = capture
        calibrationDetector = LandmarkDetector()

        let secondaryNames = NSScreen.screens.dropFirst().enumerated().map { index, screen in
            let name = screen.localizedName
            return name.isEmpty ? "Screen \(index + 2)" : name
        }

        let wizard = CalibrationWizard(
            capture: capture,
            secondaryScreenNames: secondaryNames,
            baseProfileId: controlStore.state.setupProfileId,
            onSave: { [weak self] cal in
                self?.calibrationStore.save(cal)
                self?.dismissCalibration()
            },
            onCancel: { [weak self] in
                self?.dismissCalibration()
            }
        )
        let host = NSHostingController(rootView: wizard)
        calibrationHost = host

        guard let mainWindow else {
            presentInfo("Open the main window before calibrating.")
            dismissCalibration()
            return
        }
        mainWindow.contentViewController?.presentAsSheet(host)
    }

    private func dismissCalibration() {
        if let host = calibrationHost, let parent = host.parent {
            parent.dismiss(host)
        }
        calibrationHost = nil
        calibrationCapture = nil
        calibrationDetector = nil
    }

    private func runAutoDetect() {
        let proposed = AutoDetect.proposeProfileLive()
        let alert = NSAlert()
        alert.messageText = "Detected: \(proposed.name)"
        alert.informativeText = "Apply this setup?"
        alert.addButton(withTitle: "Use it")
        alert.addButton(withTitle: "Cancel")
        if alert.runModal() == .alertFirstButtonReturn {
            controlStore.setSetupProfileId(proposed.id)
        }
    }

    private func presentFatal(_ message: String) {
        let alert = NSAlert()
        alert.messageText = "GazeLock"
        alert.informativeText = message
        alert.alertStyle = .critical
        alert.addButton(withTitle: "Quit")
        alert.runModal()
        NSApp.terminate(nil)
    }

    private func presentInfo(_ message: String) {
        let alert = NSAlert()
        alert.messageText = "GazeLock"
        alert.informativeText = message
        alert.alertStyle = .informational
        alert.addButton(withTitle: "OK")
        alert.runModal()
    }
}

extension AppDelegate: NSWindowDelegate {
    func windowWillClose(_ notification: Notification) {
        if (notification.object as? NSWindow) === mainWindow {
            NSApp.setActivationPolicy(.accessory)
        }
    }
}
