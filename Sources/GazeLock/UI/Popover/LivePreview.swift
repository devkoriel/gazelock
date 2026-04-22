import AppKit
import CoreImage
import CoreVideo
import SwiftUI

/// Small live preview with a "BEFORE / AFTER" split. Both halves
/// receive frames from the same pipeline; the BEFORE side displays the
/// raw input, the AFTER side displays the pipeline output. In Phase 3b
/// they share the same frame source since the main-app preview isn't
/// yet running a separate raw-vs-processed pipeline; the simpler thing
/// is to show the same processed frame on both halves until we wire up
/// a second dispatch.
public struct LivePreview: View {
    @Binding var beforeImage: NSImage?
    @Binding var afterImage: NSImage?

    public init(beforeImage: Binding<NSImage?>, afterImage: Binding<NSImage?>) {
        self._beforeImage = beforeImage
        self._afterImage = afterImage
    }

    public var body: some View {
        HStack(spacing: 3) {
            previewHalf(image: beforeImage, label: "BEFORE")
            previewHalf(image: afterImage, label: "AFTER")
        }
        .frame(height: 70)
        .clipShape(RoundedRectangle(cornerRadius: PopoverStyle.cornerRadius - 4))
    }

    @ViewBuilder
    private func previewHalf(image: NSImage?, label: String) -> some View {
        ZStack(alignment: .bottomLeading) {
            if let image {
                Image(nsImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
            } else {
                Rectangle()
                    .fill(PopoverStyle.backgroundElevated)
            }
            Text(label)
                .font(PopoverStyle.labelFont(size: 8))
                .tracking(1.5)
                .foregroundStyle(PopoverStyle.textSecondary)
                .padding(6)
        }
    }
}

public enum PreviewFrameRenderer {
    private static let ciContext = CIContext()

    public static func nsImage(from pixelBuffer: CVPixelBuffer) -> NSImage? {
        let ci = CIImage(cvPixelBuffer: pixelBuffer)
        guard let cg = ciContext.createCGImage(ci, from: ci.extent) else { return nil }
        return NSImage(cgImage: cg, size: .zero)
    }
}
