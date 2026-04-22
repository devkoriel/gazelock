import Foundation

/// Per-user multi-monitor calibration. Records head-pose centroids for
/// the camera, the primary screen, and each secondary screen. Runtime
/// detector uses these for precise monitor-switch disengagement. Spec §8.9.
///
/// Stored as yaw/pitch/roll triples (radians) — not as HeadPose — because
/// HeadPose lives in the GazeLock target and this file must compile in
/// both the GazeLock and GazeLockCameraExtension targets (via Sources/Shared).
public struct UserCalibration: Codable, Sendable, Equatable {
    public struct Centroid: Codable, Sendable, Equatable {
        public var yawRad: Double
        public var pitchRad: Double
        public var rollRad: Double
        public init(yawRad: Double, pitchRad: Double, rollRad: Double) {
            self.yawRad = yawRad
            self.pitchRad = pitchRad
            self.rollRad = rollRad
        }
    }

    public var baseProfileId: String
    public var cameraCentroid: Centroid
    public var primaryScreenCentroid: Centroid
    public var secondaryScreenCentroids: [Centroid]
    public var createdAt: Date

    public init(
        baseProfileId: String,
        cameraCentroid: Centroid,
        primaryScreenCentroid: Centroid,
        secondaryScreenCentroids: [Centroid],
        createdAt: Date = Date()
    ) {
        self.baseProfileId = baseProfileId
        self.cameraCentroid = cameraCentroid
        self.primaryScreenCentroid = primaryScreenCentroid
        self.secondaryScreenCentroids = secondaryScreenCentroids
        self.createdAt = createdAt
    }
}
