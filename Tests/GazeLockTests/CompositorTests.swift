import CoreVideo
import XCTest
@testable import GazeLock

final class CompositorTests: XCTestCase {
    func testFeatherMaskReaches1InCenter() {
        // Indirectly test via known identity composite:
        // An ROI fully covering a buffer with a different colour
        // should mostly overwrite, with softened edges.
        let full = PixelBufferHelpers.make(width: 16, height: 16, fillByte: 0x00)
        let roiSize = (w: 16, h: 16)
        let patch = [UInt8](repeating: 0xFF, count: roiSize.w * roiSize.h * 4)

        Compositor.compositeROI(
            full: full,
            eyePatch: patch,
            roiOrigin: (0, 0),
            roiSize: roiSize,
            featherPx: 2.0
        )

        CVPixelBufferLockBaseAddress(full, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(full, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(full) else { return XCTFail() }
        let stride = CVPixelBufferGetBytesPerRow(full)
        // Center pixel should be ~0xFF (alpha ≈ 1)
        let center = base.advanced(by: 8 * stride + 8 * 4).load(as: UInt8.self)
        XCTAssertGreaterThan(center, 0xF0)
        // Corner pixel should be less than center (feathered)
        let corner = base.advanced(by: 0 * stride + 0 * 4).load(as: UInt8.self)
        XCTAssertLessThan(corner, 0xC0)
    }
}
