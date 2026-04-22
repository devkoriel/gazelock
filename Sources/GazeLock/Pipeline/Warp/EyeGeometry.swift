import Foundation

/// Spec §6.3 — project iris rotation onto the image plane.
///
/// Eyeball is modeled as a sphere with radius ≈ 12 mm. Iris sits on
/// the front surface. Given landmarks and head pose, compute the
/// target iris pixel position that would correspond to "looking at
/// camera".
public enum EyeGeometry {
    public static let eyeballRadiusMm: Double = 12.0

    /// Compute the target iris pixel position for "looks-at-camera".
    ///
    /// Mirrors `gazelock_ml.warp.geometry.compute_target_iris_px`.
    public static func targetIrisPx(
        eye: EyeLandmarks,
        headPose: HeadPose
    ) -> Vec2 {
        let cy = cos(headPose.yaw)
        let sy = sin(headPose.yaw)
        let cp = cos(headPose.pitch)
        let sp = sin(headPose.pitch)
        let cr = cos(headPose.roll)
        let sr = sin(headPose.roll)

        // Z-Y-X Euler (intrinsic) rotation matrix: head → camera frame
        // Row-major, 3x3.
        let rot: [[Double]] = [
            [cy * cr - sy * sp * sr, -cy * sr - sy * sp * cr, -sy * cp],
            [cp * sr,                 cp * cr,                 -sp],
            [sy * cr + cy * sp * sr, -sy * sr + cy * sp * cr, cy * cp],
        ]

        // Target gaze = -z in camera frame. In head frame: rot^T @ [0,0,-1].
        // That's the 3rd column of rot^T, negated — equivalently, row 2
        // of rot, negated.
        let camDirHead = (
            x: -rot[2][0],
            y: -rot[2][1],
            z: -rot[2][2]
        )

        // Iris radius in px → mm/px scale (iris diameter ~12 mm → radius 6 mm)
        let pxPerMm = eye.irisRadiusPx / 6.0

        let dispMm = (
            x: eyeballRadiusMm * camDirHead.x,
            y: eyeballRadiusMm * camDirHead.y
        )

        let dxPx = dispMm.x * pxPerMm
        let dyPx = -dispMm.y * pxPerMm  // flip y for image coordinates

        return Vec2(
            eye.pupilCenter.x + dxPx,
            eye.pupilCenter.y + dyPx
        )
    }
}
