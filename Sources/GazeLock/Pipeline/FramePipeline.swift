import CoreVideo
import Foundation

/// End-to-end per-frame orchestrator. Wires every pipeline stage.
///
/// Usage:
///     let pipeline = try FramePipeline()
///     let output = try pipeline.process(pixelBuffer: cvpb, timestamp: ts, intensity: 0.7)
///
/// If no face is detected in the frame, returns the input unchanged.
/// If no Core ML refiner is available, runs warp-only (pure Approach A).
public final class FramePipeline {
    public enum Error: Swift.Error {
        case metalInitFailed
        case pixelBufferAllocationFailed
    }

    private let detector: LandmarkDetector
    private let warp: MetalWarpPipeline
    private let refiner: CoreMLRefiner?
    private let lookAway: LookAwayDetector
    private let eyeROISize = (w: 96, h: 72)

    public init(bundle: Bundle = .main) throws {
        self.detector = LandmarkDetector()
        do {
            self.warp = try MetalWarpPipeline()
        } catch {
            throw Error.metalInitFailed
        }
        self.refiner = try? CoreMLRefiner(bundle: bundle)
        self.lookAway = LookAwayDetector(sensitivity: .normal)
    }

    /// Process a single frame. `intensity` is the user's base slider value;
    /// the LookAwayDetector scales it based on head pose. Aim offsets shift
    /// the correction target within the image plane.
    public func process(
        pixelBuffer: CVPixelBuffer,
        timestamp: TimeInterval,
        intensity: Double,
        verticalAimDeg: Double = -2.0,
        horizontalAimDeg: Double = 0.0,
        sensitivity: Sensitivity = .normal
    ) throws -> CVPixelBuffer {
        guard let landmarks = try detector.detect(in: pixelBuffer, timestamp: timestamp) else {
            return pixelBuffer  // no face → passthrough
        }

        lookAway.sensitivity = sensitivity
        let alpha = lookAway.compute(headPose: landmarks.headPoseRadians, timestampSeconds: timestamp)
        let effectiveIntensity = intensity * alpha

        if effectiveIntensity < 0.01 {
            return pixelBuffer
        }

        let output = try copyOf(pixelBuffer)

        for eye in [landmarks.leftEye, landmarks.rightEye] {
            try warpEye(
                eye: eye,
                headPose: landmarks.headPoseRadians,
                intensity: effectiveIntensity,
                verticalAimDeg: verticalAimDeg,
                horizontalAimDeg: horizontalAimDeg,
                source: pixelBuffer,
                destination: output
            )
        }

        return output
    }

    private func warpEye(
        eye: EyeLandmarks,
        headPose: HeadPose,
        intensity: Double,
        verticalAimDeg: Double,
        horizontalAimDeg: Double,
        source: CVPixelBuffer,
        destination: CVPixelBuffer
    ) throws {
        let targetIris = EyeGeometry.targetIrisPx(
            eye: eye,
            headPose: headPose,
            verticalAimDeg: verticalAimDeg,
            horizontalAimDeg: horizontalAimDeg
        )
        let displacement = Vec2(
            (targetIris.x - eye.pupilCenter.x) * intensity,
            (targetIris.y - eye.pupilCenter.y) * intensity
        )
        let effectiveTarget = eye.pupilCenter + displacement

        // Build TPS control points: 4 ROI corners + iris center
        let roiX = max(0, Int(eye.pupilCenter.x) - eyeROISize.w / 2)
        let roiY = max(0, Int(eye.pupilCenter.y) - eyeROISize.h / 2)

        let source4 = [
            Vec2(Double(roiX), Double(roiY)),
            Vec2(Double(roiX + eyeROISize.w - 1), Double(roiY)),
            Vec2(Double(roiX), Double(roiY + eyeROISize.h - 1)),
            Vec2(Double(roiX + eyeROISize.w - 1), Double(roiY + eyeROISize.h - 1)),
            eye.pupilCenter,
        ]
        var target5 = source4
        target5[4] = effectiveTarget

        let tps = ThinPlateSpline.fit(source: target5, target: source4)  // inverse mapping for sampling
        let flow = FlowField.from(tps: tps, width: eyeROISize.w, height: eyeROISize.h)

        try warp.apply(
            source: source,
            destination: destination,
            flow: flow,
            roiOrigin: (roiX, roiY)
        )
    }

    private func copyOf(_ pb: CVPixelBuffer) throws -> CVPixelBuffer {
        let w = CVPixelBufferGetWidth(pb)
        let h = CVPixelBufferGetHeight(pb)
        let attrs: [String: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
        ]
        var out: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &out)
        guard let output = out else { throw Error.pixelBufferAllocationFailed }
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        CVPixelBufferLockBaseAddress(output, [])
        defer {
            CVPixelBufferUnlockBaseAddress(pb, .readOnly)
            CVPixelBufferUnlockBaseAddress(output, [])
        }
        if let src = CVPixelBufferGetBaseAddress(pb),
           let dst = CVPixelBufferGetBaseAddress(output) {
            memcpy(dst, src, CVPixelBufferGetBytesPerRow(pb) * h)
        }
        return output
    }
}
