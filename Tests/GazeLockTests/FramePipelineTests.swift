import CoreVideo
import XCTest
@testable import GazeLock

final class FramePipelineTests: XCTestCase {
    func testPassesThroughWhenNoFaceDetected() throws {
        let pipeline = try FramePipeline()
        let blank = PixelBufferHelpers.make(width: 320, height: 240, fillByte: 0x40)
        let out = try pipeline.process(pixelBuffer: blank, timestamp: 0.0, intensity: 0.7)
        // No face → returns the same buffer reference
        XCTAssertTrue(out === blank)
    }

    func testSurvivesEmptyIntensity() throws {
        let pipeline = try FramePipeline()
        let blank = PixelBufferHelpers.make(width: 320, height: 240, fillByte: 0x80)
        let out = try pipeline.process(pixelBuffer: blank, timestamp: 0.0, intensity: 0.0)
        XCTAssertTrue(out === blank)
    }
}
