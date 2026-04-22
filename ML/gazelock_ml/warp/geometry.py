"""3D eyeball geometry — project iris rotation onto the image plane.

Mirrors design spec §6.3. Eyeball is modeled as a sphere with radius
``EYEBALL_RADIUS_MM`` (~12 mm). Iris sits on the front surface of the
sphere. Given landmarks and head pose, we compute the target iris
pixel position required for the user to "appear to look at camera".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EYEBALL_RADIUS_MM: float = 12.0
# Approximate depth-from-iris to eyeball-center along the face's forward
# axis. Kept as a module constant so the Swift and Python halves agree.


@dataclass(frozen=True)
class EyeLandmarks:
    """Pixel-space landmarks for one eye."""

    # (x, y) pixel coordinates
    outer_corner: tuple[float, float]
    inner_corner: tuple[float, float]
    top_lid: tuple[float, float]
    bottom_lid: tuple[float, float]
    iris_center: tuple[float, float]
    iris_radius_px: float


@dataclass(frozen=True)
class HeadPose:
    """Euler angles in radians, intrinsic."""

    yaw: float  # rotation around vertical axis, + = looking right (user's right)
    pitch: float  # rotation around lateral axis, + = looking up
    roll: float  # rotation around optical axis


def compute_target_iris_px(
    landmarks: EyeLandmarks,
    head_pose: HeadPose,
) -> tuple[float, float]:
    """Return target iris (x, y) in pixels so the user "looks at camera".

    Method:
        1. Estimate eyeball center in 3D (slightly behind iris center).
        2. Compute the 3D direction from eyeball center to the camera
           (always +z in camera coords, so the target gaze vector is
           simply (0, 0, -1) in head-frame after undoing the head pose).
        3. Project: where the iris surface intersects that direction
           gives the new iris front-surface point in 3D.
        4. Project back to the image plane using the iris_radius_px as
           the scale factor (monocular — we avoid a full camera model).
    """
    # Convert head-pose to a rotation matrix (Z-Y-X Euler, intrinsic)
    cy, sy = np.cos(head_pose.yaw), np.sin(head_pose.yaw)
    cp, sp = np.cos(head_pose.pitch), np.sin(head_pose.pitch)
    cr, sr = np.cos(head_pose.roll), np.sin(head_pose.roll)

    # Rotation: head-frame → camera-frame
    rot_head_to_cam = np.array(
        [
            [cy * cr - sy * sp * sr, -cy * sr - sy * sp * cr, -sy * cp],
            [cp * sr, cp * cr, -sp],
            [sy * cr + cy * sp * sr, -sy * sr + cy * sp * cr, cy * cp],
        ],
        dtype=np.float64,
    )

    # Target gaze direction in head frame (eyes-forward = +z in head frame
    # after applying rot_head_to_cam). What we want is for the iris to
    # look directly at the camera, which in camera frame is -Z. In head
    # frame that is rot_head_to_cam.T @ [0, 0, -1].
    cam_dir_head = rot_head_to_cam.T @ np.array([0.0, 0.0, -1.0], dtype=np.float64)

    # The iris_radius_px is our scale for mm→px on the face plane.
    px_per_mm = landmarks.iris_radius_px / 6.0  # iris diameter ~12 mm → radius 6 mm

    # Displacement from eyeball center to target iris position, in mm
    disp_mm = EYEBALL_RADIUS_MM * cam_dir_head  # 3D
    # Project onto image plane (ignore z — orthographic approximation
    # adequate for small 5–15° correction angles)
    dx_px = disp_mm[0] * px_per_mm
    dy_px = -disp_mm[1] * px_per_mm  # flip y for image coordinates

    cx, cy_pix = landmarks.iris_center
    return (cx + dx_px, cy_pix + dy_px)


__all__ = [
    "EYEBALL_RADIUS_MM",
    "EyeLandmarks",
    "HeadPose",
    "compute_target_iris_px",
]
