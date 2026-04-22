import XCTest
@testable import GazeLock

final class LandmarkDetectorTests: XCTestCase {
    func testReturnsNilOnBlankFrame() throws {
        let detector = LandmarkDetector()
        let pb = PixelBufferHelpers.make(width: 320, height: 240, fillByte: 0x00)
        let result = try detector.detect(in: pb, timestamp: 0.0)
        XCTAssertNil(result, "Blank frame should produce no face observation")
    }

    func testResetDoesNotCrash() {
        let detector = LandmarkDetector()
        detector.reset()
    }
}
