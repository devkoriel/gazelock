import Foundation
import CoreMediaIO
import CoreVideo
import CoreMedia

final class CameraExtensionStream: NSObject, CMIOExtensionStreamSource, @unchecked Sendable {
    private(set) var stream: CMIOExtensionStream!
    let availableFormats: [CMIOExtensionStreamFormat]

    // TODO(phase3): `streamingCounter` and `timer` are mutated from both the
    // framework's clientQueue (via startStream/stopStream) and timerQueue (via
    // emitFrame). Today the framework serialises the lifecycle calls and the
    // timer only mutates inside them, so there is no realistic race — but an
    // actor wrapper would make this explicit.
    private var streamingCounter: Int = 0
    private var timer: DispatchSourceTimer?
    private let timerQueue = DispatchQueue(
        label: "com.gazelock.GazeLock.CameraExtension.streamTimer",
        qos: .userInteractive
    )

    private let width: Int32 = 1280
    private let height: Int32 = 720
    private let fps: Int32 = 60

    init(localizedName: String) {
        let formatDescription = try! makeFormatDescription(width: width, height: height)
        let videoStreamFormat = CMIOExtensionStreamFormat(
            formatDescription: formatDescription,
            maxFrameDuration: CMTime(value: 1, timescale: fps),
            minFrameDuration: CMTime(value: 1, timescale: fps),
            validFrameDurations: nil
        )
        self.availableFormats = [videoStreamFormat]

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
        if timer == nil { startTimer() }
    }

    func stopStream() throws {
        streamingCounter = max(0, streamingCounter - 1)
        if streamingCounter == 0 { stopTimer() }
    }

    private func startTimer() {
        let newTimer = DispatchSource.makeTimerSource(queue: timerQueue)
        newTimer.schedule(deadline: .now(), repeating: .milliseconds(Int(1000 / fps)))
        newTimer.setEventHandler { [weak self] in self?.emitFrame() }
        newTimer.resume()
        timer = newTimer
    }

    private func stopTimer() {
        timer?.cancel()
        timer = nil
    }

    private func emitFrame() {
        guard let pixelBuffer = makeSolidColorPixelBuffer() else { return }
        var formatDesc: CMFormatDescription?
        CMVideoFormatDescriptionCreateForImageBuffer(
            allocator: kCFAllocatorDefault,
            imageBuffer: pixelBuffer,
            formatDescriptionOut: &formatDesc
        )
        guard let desc = formatDesc else { return }

        let hostTime = CMClockGetTime(CMClockGetHostTimeClock())
        var timingInfo = CMSampleTimingInfo(
            duration: CMTime(value: 1, timescale: fps),
            presentationTimeStamp: hostTime,
            decodeTimeStamp: .invalid
        )
        var sampleBuffer: CMSampleBuffer?
        CMSampleBufferCreateReadyWithImageBuffer(
            allocator: kCFAllocatorDefault,
            imageBuffer: pixelBuffer,
            formatDescription: desc,
            sampleTiming: &timingInfo,
            sampleBufferOut: &sampleBuffer
        )
        if let buffer = sampleBuffer {
            // `hostTimeInNanoseconds` expects nanoseconds; `mach_absolute_time()`
            // returns Mach ticks (≈ 41.67 ns per tick on Apple Silicon), so we
            // use the nanosecond-native clock API instead.
            let hostTimeNs = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            stream.send(buffer, discontinuity: [], hostTimeInNanoseconds: hostTimeNs)
        }
    }

    // TODO(phase3): replace per-frame `CVPixelBufferCreate` with a
    // `CVPixelBufferPool` so the Phase 3 pipeline reuses IOSurface-backed
    // buffers instead of allocating 60/sec.
    private func makeSolidColorPixelBuffer() -> CVPixelBuffer? {
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary
        ] as CFDictionary

        var pb: CVPixelBuffer?
        CVPixelBufferCreate(
            kCFAllocatorDefault,
            Int(width),
            Int(height),
            kCVPixelFormatType_32BGRA,
            attrs,
            &pb
        )
        guard let buffer = pb else { return nil }
        CVPixelBufferLockBaseAddress(buffer, [])
        if let base = CVPixelBufferGetBaseAddress(buffer) {
            let count = CVPixelBufferGetBytesPerRow(buffer) * Int(height)
            memset(base, 0x20, count)
        }
        CVPixelBufferUnlockBaseAddress(buffer, [])
        return buffer
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
