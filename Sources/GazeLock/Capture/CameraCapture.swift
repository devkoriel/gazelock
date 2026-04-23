import AVFoundation
import CoreMedia
import CoreVideo
import Foundation

/// AVCaptureSession wrapper that delivers CVPixelBuffers on a
/// user-provided queue. Not thread-safe on its own — caller owns a
/// single instance per process.
public final class CameraCapture: NSObject {
    public enum CaptureError: Error {
        case noDevices
        case deviceNotFound(uniqueID: String)
        case cannotAddInput
        case cannotAddOutput
    }

    private let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sampleQueue = DispatchQueue(label: "com.gazelock.GazeLock.CameraCapture.samples")
    private let sessionQueue = DispatchQueue(label: "com.gazelock.GazeLock.CameraCapture.session")

    /// Delivered on `sampleQueue`. Caller must not block.
    public var onFrame: ((CVPixelBuffer, TimeInterval) -> Void)?

    public func start(deviceUniqueID: String? = nil) throws {
        // Pick device BEFORE the configuration block so we can throw Swift
        // errors out cleanly before touching the session's lock.
        let device: AVCaptureDevice?
        if let uid = deviceUniqueID {
            device = AVCaptureDevice(uniqueID: uid)
            if device == nil {
                throw CaptureError.deviceNotFound(uniqueID: uid)
            }
        } else {
            let discovery = AVCaptureDevice.DiscoverySession(
                deviceTypes: [.builtInWideAngleCamera, .external],
                mediaType: .video,
                position: .unspecified
            )
            device = discovery.devices.first
        }
        guard let dev = device else { throw CaptureError.noDevices }

        let input = try AVCaptureDeviceInput(device: dev)

        // Configure the session. CRITICAL: we MUST call commitConfiguration
        // before startRunning, otherwise AVCaptureSession throws an ObjC
        // exception that Swift can't catch (process crash).
        session.beginConfiguration()

        // Remove existing inputs/outputs (idempotent start)
        for existing in session.inputs { session.removeInput(existing) }
        for existing in session.outputs { session.removeOutput(existing) }

        guard session.canAddInput(input) else {
            session.commitConfiguration()
            throw CaptureError.cannotAddInput
        }
        session.addInput(input)

        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        ]
        videoOutput.setSampleBufferDelegate(self, queue: sampleQueue)
        videoOutput.alwaysDiscardsLateVideoFrames = true
        guard session.canAddOutput(videoOutput) else {
            session.commitConfiguration()
            throw CaptureError.cannotAddOutput
        }
        session.addOutput(videoOutput)

        if session.canSetSessionPreset(.hd1280x720) {
            session.sessionPreset = .hd1280x720
        }

        session.commitConfiguration()

        // startRunning is synchronous and blocks until the session starts.
        // Apple's docs say it MUST be called off the main queue.
        sessionQueue.async { [session] in
            session.startRunning()
        }
    }

    public func stop() {
        sessionQueue.async { [session] in
            session.stopRunning()
        }
    }

    public func isRunning() -> Bool {
        session.isRunning
    }
}

extension CameraCapture: AVCaptureVideoDataOutputSampleBufferDelegate {
    public func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        let seconds = CMTimeGetSeconds(pts)
        onFrame?(pixelBuffer, seconds.isFinite ? seconds : 0)
    }
}
