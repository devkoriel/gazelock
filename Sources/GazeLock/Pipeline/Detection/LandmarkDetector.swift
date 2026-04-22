import CoreVideo
import Foundation
import Vision

/// Per-eye landmarks detected by Vision for one frame.
public struct EyeLandmarks: Equatable, Sendable {
    public let eyeContour: [Vec2]    // ~8 points around the eye boundary
    public let pupilCenter: Vec2     // Vision's VNFaceLandmarks2D `leftPupil` / `rightPupil`
    public let irisRadiusPx: Double  // estimated from the contour bounding box

    public init(eyeContour: [Vec2], pupilCenter: Vec2, irisRadiusPx: Double) {
        self.eyeContour = eyeContour
        self.pupilCenter = pupilCenter
        self.irisRadiusPx = irisRadiusPx
    }
}

/// Detection result for a single frame.
public struct FaceLandmarksResult: Equatable, Sendable {
    public let imageWidth: Int
    public let imageHeight: Int
    public let faceBoundingBox: CGRect
    public let headPoseRadians: HeadPose   // yaw, pitch, roll
    public let leftEye: EyeLandmarks
    public let rightEye: EyeLandmarks
}

public struct HeadPose: Equatable, Sendable {
    public let yaw: Double
    public let pitch: Double
    public let roll: Double

    public init(yaw: Double, pitch: Double, roll: Double) {
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
    }

    public static let identity = HeadPose(yaw: 0, pitch: 0, roll: 0)
}

/// Wraps `VNDetectFaceLandmarksRequest` with per-eye smoothing.
public final class LandmarkDetector {
    private let leftEyeSmoother: LandmarkSmoother
    private let rightEyeSmoother: LandmarkSmoother
    private let leftPupilKalman: IrisKalman
    private let rightPupilKalman: IrisKalman

    public init(eyeContourPoints: Int = 8) {
        self.leftEyeSmoother = LandmarkSmoother(count: eyeContourPoints)
        self.rightEyeSmoother = LandmarkSmoother(count: eyeContourPoints)
        self.leftPupilKalman = IrisKalman()
        self.rightPupilKalman = IrisKalman()
    }

    /// Run Vision on a pixel buffer; return landmarks or nil if no face.
    ///
    /// `timestamp` drives the temporal smoothing. Use the frame's
    /// presentation timestamp in seconds.
    public func detect(in pixelBuffer: CVPixelBuffer, timestamp: TimeInterval) throws -> FaceLandmarksResult? {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up)
        let request = VNDetectFaceLandmarksRequest()
        try handler.perform([request])

        guard let observation = request.results?.first as? VNFaceObservation,
              let landmarks = observation.landmarks,
              let leftEyeRegion = landmarks.leftEye,
              let rightEyeRegion = landmarks.rightEye,
              let leftPupil = landmarks.leftPupil,
              let rightPupil = landmarks.rightPupil
        else {
            return nil
        }

        let leftContour = Self.pointsInImageSpace(
            region: leftEyeRegion,
            faceBoundingBox: observation.boundingBox,
            imageSize: CGSize(width: width, height: height)
        )
        let rightContour = Self.pointsInImageSpace(
            region: rightEyeRegion,
            faceBoundingBox: observation.boundingBox,
            imageSize: CGSize(width: width, height: height)
        )
        let leftPupilPts = Self.pointsInImageSpace(
            region: leftPupil,
            faceBoundingBox: observation.boundingBox,
            imageSize: CGSize(width: width, height: height)
        )
        let rightPupilPts = Self.pointsInImageSpace(
            region: rightPupil,
            faceBoundingBox: observation.boundingBox,
            imageSize: CGSize(width: width, height: height)
        )

        let smoothedLeftContour = leftEyeSmoother.smooth(leftContour, timestamp: timestamp)
        let smoothedRightContour = rightEyeSmoother.smooth(rightContour, timestamp: timestamp)
        let smoothedLeftPupil = leftPupilKalman.update(
            measurement: Self.centroid(leftPupilPts),
            timestamp: timestamp
        )
        let smoothedRightPupil = rightPupilKalman.update(
            measurement: Self.centroid(rightPupilPts),
            timestamp: timestamp
        )

        return FaceLandmarksResult(
            imageWidth: width,
            imageHeight: height,
            faceBoundingBox: observation.boundingBox,
            headPoseRadians: HeadPose(
                yaw: Double(observation.yaw?.doubleValue ?? 0),
                pitch: Double(observation.pitch?.doubleValue ?? 0),
                roll: Double(observation.roll?.doubleValue ?? 0)
            ),
            leftEye: EyeLandmarks(
                eyeContour: smoothedLeftContour,
                pupilCenter: smoothedLeftPupil,
                irisRadiusPx: Self.irisRadius(from: smoothedLeftContour)
            ),
            rightEye: EyeLandmarks(
                eyeContour: smoothedRightContour,
                pupilCenter: smoothedRightPupil,
                irisRadiusPx: Self.irisRadius(from: smoothedRightContour)
            )
        )
    }

    public func reset() {
        leftEyeSmoother.reset()
        rightEyeSmoother.reset()
        leftPupilKalman.reset()
        rightPupilKalman.reset()
    }

    private static func pointsInImageSpace(
        region: VNFaceLandmarkRegion2D,
        faceBoundingBox: CGRect,
        imageSize: CGSize
    ) -> [Vec2] {
        region.normalizedPoints.map { normalized in
            // normalizedPoints are in the face bounding-box coordinate
            // system (origin lower-left, normalised 0..1). Convert to
            // full-image pixels with y flipped.
            let fx = faceBoundingBox.origin.x + Double(normalized.x) * faceBoundingBox.width
            let fy = faceBoundingBox.origin.y + Double(normalized.y) * faceBoundingBox.height
            let px = fx * imageSize.width
            let py = (1.0 - fy) * imageSize.height
            return Vec2(px, py)
        }
    }

    private static func centroid(_ points: [Vec2]) -> Vec2 {
        guard !points.isEmpty else { return .zero }
        var sx = 0.0
        var sy = 0.0
        for p in points { sx += p.x; sy += p.y }
        let n = Double(points.count)
        return Vec2(sx / n, sy / n)
    }

    private static func irisRadius(from eyeContour: [Vec2]) -> Double {
        // Rough estimate: half the horizontal span of the eye contour,
        // then scale by 0.35 (iris ~ 35% of eye width).
        guard let first = eyeContour.first else { return 10.0 }
        var minX = first.x, maxX = first.x
        for p in eyeContour {
            if p.x < minX { minX = p.x }
            if p.x > maxX { maxX = p.x }
        }
        return max(4.0, (maxX - minX) * 0.35 * 0.5)  // conservative floor
    }
}
