import Foundation
import CoreMediaIO
import CoreVideo
import CoreMedia
import AVFoundation
import os

final class CameraExtensionStream: NSObject, CMIOExtensionStreamSource, @unchecked Sendable {
    private(set) var stream: CMIOExtensionStream!
    let formats: [CMIOExtensionStreamFormat]

    private static let logger = Logger(
        subsystem: "com.gazelock.GazeLock.CameraExtension",
        category: "Stream"
    )

    private var streamingCounter: Int = 0

    // Guards against `stream.send` racing with teardown: `AVCaptureSession.stopRunning()`
    // does not synchronously drain in-flight frames, so a frame dispatched just before
    // `stopStream` can otherwise reach `emitSampleBuffer` after the stream is torn down.
    private let stateLock = NSLock()
    private var isStreaming = false  // guarded by stateLock

    private let width: Int32 = 1280
    private let height: Int32 = 720
    private let fps: Int32 = 60

    private let capture = CameraCapture()
    private let controlClient: ControlClient
    private let pipeline: FramePipeline?

    init(localizedName: String, controlClient: ControlClient) {
        self.controlClient = controlClient

        // Pipeline init can fail (Metal unavailable, refiner missing, etc.).
        // Fall back to passthrough so the extension never crashes the host.
        let builtPipeline: FramePipeline?
        do {
            builtPipeline = try FramePipeline()
        } catch {
            Self.logger.error(
                "FramePipeline init failed: \(error.localizedDescription, privacy: .public) — passthrough"
            )
            builtPipeline = nil
        }
        self.pipeline = builtPipeline

        let formatDescription = try! makeFormatDescription(width: width, height: height)
        let videoStreamFormat = CMIOExtensionStreamFormat(
            formatDescription: formatDescription,
            maxFrameDuration: CMTime(value: 1, timescale: fps),
            minFrameDuration: CMTime(value: 1, timescale: fps),
            validFrameDurations: nil
        )
        self.formats = [videoStreamFormat]

        super.init()

        stream = CMIOExtensionStream(
            localizedName: localizedName,
            streamID: UUID(),
            direction: .source,
            clockType: .hostTime,
            source: self
        )
    }

    var availableProperties: Set<CMIOExtensionProperty> {
        [.streamActiveFormatIndex, .streamFrameDuration]
    }

    func streamProperties(
        forProperties properties: Set<CMIOExtensionProperty>
    ) throws -> CMIOExtensionStreamProperties {
        let state = CMIOExtensionStreamProperties(dictionary: [:])
        if properties.contains(.streamActiveFormatIndex) {
            state.activeFormatIndex = 0
        }
        if properties.contains(.streamFrameDuration) {
            state.frameDuration = CMTime(value: 1, timescale: fps)
        }
        return state
    }

    func setStreamProperties(_ streamProperties: CMIOExtensionStreamProperties) throws {}

    func authorizedToStartStream(for client: CMIOExtensionClient) -> Bool { true }

    func startStream() throws {
        streamingCounter += 1
        guard streamingCounter == 1 else { return }

        Self.logger.info("startStream: activating capture → pipeline → emit path")

        controlClient.start()

        capture.onFrame = { [weak self] pixelBuffer, timestamp in
            self?.processFrame(pixelBuffer: pixelBuffer, timestamp: timestamp)
        }

        // TODO(phase3c): observe ControlState changes and restart capture if sourceCameraUniqueID changes mid-stream.
        let sourceID = controlClient.currentState().sourceCameraUniqueID
        do {
            try capture.start(deviceUniqueID: sourceID)
        } catch {
            Self.logger.error("CameraCapture.start failed: \(error.localizedDescription, privacy: .public)")
            // Roll back partial start so the next startStream retries cleanly.
            controlClient.stop()
            streamingCounter = 0
            throw error
        }
        stateLock.withLock { self.isStreaming = true }
    }

    func stopStream() throws {
        streamingCounter = max(0, streamingCounter - 1)
        guard streamingCounter == 0 else { return }

        Self.logger.info("stopStream: tearing down capture + control client")

        // Drop any in-flight frames before they can reach `stream.send` post-teardown.
        stateLock.withLock { self.isStreaming = false }
        capture.stop()
        capture.onFrame = nil
        controlClient.stop()
    }

    private func processFrame(pixelBuffer: CVPixelBuffer, timestamp: TimeInterval) {
        let state = controlClient.currentState()
        guard state.isEnabled, let pipeline else {
            emitSampleBuffer(pixelBuffer: pixelBuffer, timestamp: timestamp)
            return
        }

        let output: CVPixelBuffer
        do {
            output = try pipeline.process(
                pixelBuffer: pixelBuffer,
                timestamp: timestamp,
                intensity: state.intensity
            )
        } catch {
            Self.logger.error(
                "pipeline.process failed: \(error.localizedDescription, privacy: .public) — emitting raw frame"
            )
            output = pixelBuffer
        }
        emitSampleBuffer(pixelBuffer: output, timestamp: timestamp)
    }

    private func emitSampleBuffer(pixelBuffer: CVPixelBuffer, timestamp: TimeInterval) {
        // Drop frames that arrive after teardown has begun; prevents `stream.send`
        // on a torn-down CMIO stream.
        let canSend = stateLock.withLock { self.isStreaming }
        guard canSend else { return }

        var formatDesc: CMFormatDescription?
        let fdStatus = CMVideoFormatDescriptionCreateForImageBuffer(
            allocator: kCFAllocatorDefault,
            imageBuffer: pixelBuffer,
            formatDescriptionOut: &formatDesc
        )
        guard fdStatus == noErr, let desc = formatDesc else {
            Self.logger.error("CMVideoFormatDescriptionCreateForImageBuffer failed: \(fdStatus)")
            return
        }

        let hostTime = CMClockGetTime(CMClockGetHostTimeClock())
        var timingInfo = CMSampleTimingInfo(
            duration: CMTime(value: 1, timescale: fps),
            presentationTimeStamp: hostTime,
            decodeTimeStamp: .invalid
        )
        var sampleBuffer: CMSampleBuffer?
        let sbStatus = CMSampleBufferCreateReadyWithImageBuffer(
            allocator: kCFAllocatorDefault,
            imageBuffer: pixelBuffer,
            formatDescription: desc,
            sampleTiming: &timingInfo,
            sampleBufferOut: &sampleBuffer
        )
        guard sbStatus == noErr, let buffer = sampleBuffer else {
            Self.logger.error("CMSampleBufferCreateReadyWithImageBuffer failed: \(sbStatus)")
            return
        }

        // `hostTimeInNanoseconds` expects nanoseconds; `mach_absolute_time()`
        // returns Mach ticks (≈ 41.67 ns per tick on Apple Silicon), so we
        // use the nanosecond-native clock API instead.
        let hostTimeNs = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        stream.send(buffer, discontinuity: [], hostTimeInNanoseconds: hostTimeNs)
    }
}

private func makeFormatDescription(width: Int32, height: Int32) throws -> CMFormatDescription {
    var desc: CMFormatDescription?
    let status = CMVideoFormatDescriptionCreate(
        allocator: kCFAllocatorDefault,
        codecType: kCVPixelFormatType_32BGRA,
        width: width,
        height: height,
        extensions: nil,
        formatDescriptionOut: &desc
    )
    guard status == noErr, let description = desc else {
        throw NSError(domain: "CameraExtensionStream", code: Int(status))
    }
    return description
}
