import CoreVideo
import Foundation

/// Alpha-blend a refined (or warped) eye-region buffer back into the
/// full camera frame, using a feathered Gaussian mask so there is no
/// visible ROI boundary.
public enum Compositor {
    /// Apply the blend in-place on `destination` (which already holds
    /// the full frame). `sourceEye` contains only the ROI pixels.
    ///
    /// `featherPx` controls the Gaussian falloff at the ROI edge.
    public static func compositeROI(
        full: CVPixelBuffer,
        eyePatch: [UInt8],      // (roiW * roiH * 4) BGRA
        roiOrigin: (x: Int, y: Int),
        roiSize: (w: Int, h: Int),
        featherPx: Double = 8.0
    ) {
        let fullW = CVPixelBufferGetWidth(full)
        let fullH = CVPixelBufferGetHeight(full)
        CVPixelBufferLockBaseAddress(full, [])
        defer { CVPixelBufferUnlockBaseAddress(full, []) }
        guard let base = CVPixelBufferGetBaseAddress(full) else { return }
        let stride = CVPixelBufferGetBytesPerRow(full)

        for y in 0..<roiSize.h {
            let dy = Double(y)
            for x in 0..<roiSize.w {
                let dx = Double(x)
                let alpha = featherMask(
                    x: dx, y: dy,
                    w: Double(roiSize.w), h: Double(roiSize.h),
                    featherPx: featherPx
                )

                let dstX = roiOrigin.x + x
                let dstY = roiOrigin.y + y
                if dstX < 0 || dstX >= fullW || dstY < 0 || dstY >= fullH { continue }

                let srcIdx = (y * roiSize.w + x) * 4
                let dstPtr = base.advanced(by: dstY * stride + dstX * 4)

                for c in 0..<4 {
                    let dstByte = dstPtr.advanced(by: c).load(as: UInt8.self)
                    let srcByte = eyePatch[srcIdx + c]
                    let blended = alpha * Double(srcByte) + (1.0 - alpha) * Double(dstByte)
                    dstPtr.advanced(by: c).storeBytes(of: UInt8(min(255, max(0, blended))), as: UInt8.self)
                }
            }
        }
    }

    private static func featherMask(x: Double, y: Double, w: Double, h: Double, featherPx: Double) -> Double {
        // Gaussian falloff based on distance to the nearest edge.
        let dx = min(x, w - 1 - x)
        let dy = min(y, h - 1 - y)
        let d = min(dx, dy)
        if d >= featherPx { return 1.0 }
        let t = d / featherPx
        return t * t * (3 - 2 * t)  // smoothstep — cheap Gaussian approximation
    }
}
