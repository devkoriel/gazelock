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

    /// Delivered on `sampleQueue`. Caller must not block.
    public var onFrame: ((CVPixelBuffer, TimeInterval) -> Void)?

    public func start(deviceUniqueID: String? = nil) throws {
        session.beginConfiguration()
        defer { session.commitConfiguration() }

        // Pick device
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

        // Remove existing inputs/outputs (idempotent start)
        for input in session.inputs { session.removeInput(input) }
        for output in session.outputs { session.removeOutput(output) }

        let input = try AVCaptureDeviceInput(device: dev)
        guard session.canAddInput(input) else { throw CaptureError.cannotAddInput }
        session.addInput(input)

        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        videoOutput.setSampleBufferDelegate(self, queue: sampleQueue)
        videoOutput.alwaysDiscardsLateVideoFrames = true
        guard session.canAddOutput(videoOutput) else { throw CaptureError.cannotAddOutput }
        session.addOutput(videoOutput)

        session.sessionPreset = .hd1280x720

        session.startRunning()
    }

    public func stop() {
        session.stopRunning()
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
