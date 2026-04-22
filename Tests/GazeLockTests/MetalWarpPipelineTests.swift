import CoreVideo
import XCTest
@testable import GazeLock

final class MetalWarpPipelineTests: XCTestCase {
    func testIdentityFlowReproducesInput() throws {
        let pipeline = try MetalWarpPipeline()
        let width = 64, height = 48
        let src = PixelBufferHelpers.make(width: width, height: height, fillByte: 0xC0)
        let dst = PixelBufferHelpers.make(width: width, height: height, fillByte: 0x00)

        // Identity flow over a 32x24 ROI starting at (0, 0)
        let roiW = 32, roiH = 24
        var data: [Vec2] = []
        for y in 0..<roiH {
            for x in 0..<roiW {
                data.append(Vec2(Double(x), Double(y)))
            }
        }
        let flow = FlowField(width: roiW, height: roiH, data: data)

        try pipeline.apply(source: src, destination: dst, flow: flow, roiOrigin: (0, 0))

        CVPixelBufferLockBaseAddress(dst, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(dst, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(dst) else {
            XCTFail("dst base address nil"); return
        }
        let stride = CVPixelBufferGetBytesPerRow(dst)
        // Check a pixel within the ROI — should be 0xC0 (sampled from src)
        let p = base.advanced(by: 10 * stride + 10 * 4)
        let byte = p.load(as: UInt8.self)
        XCTAssertGreaterThan(byte, 0xB0)  // tolerate BGRA ordering + slight filter rounding
    }
}
