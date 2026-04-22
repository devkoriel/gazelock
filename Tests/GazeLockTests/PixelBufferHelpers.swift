import CoreVideo
import Foundation

/// Shared helpers for constructing `CVPixelBuffer`s in tests.
enum PixelBufferHelpers {
    /// Create a solid-colour BGRA buffer of the given size.
    static func make(width: Int, height: Int, fillByte: UInt8 = 0x80) -> CVPixelBuffer {
        let attrs: [String: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true,
        ]
        var buffer: CVPixelBuffer?
        CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &buffer
        )
        guard let pb = buffer else {
            fatalError("Failed to create CVPixelBuffer")
        }
        CVPixelBufferLockBaseAddress(pb, [])
        if let base = CVPixelBufferGetBaseAddress(pb) {
            let bytes = CVPixelBufferGetBytesPerRow(pb) * height
            memset(base, Int32(fillByte), bytes)
        }
        CVPixelBufferUnlockBaseAddress(pb, [])
        return pb
    }
}
