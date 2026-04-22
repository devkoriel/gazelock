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

    // Preview state shown in the popover
    private var previewBefore: NSImage?
    private var previewAfter: NSImage?

    func applicationDidFinishLaunching(_ notification: Notification) {
        setupPipeline()
        setupControlPlane()
        setupMenuBar()
        setupPopover()
        startCapture()
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
                // Phase 3c adds the real main window; stub here.
                self?.presentInfo("Main window arrives in Phase 3c.")
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
