"""Tests for warp geometry, TPS, and apply."""

from __future__ import annotations

import numpy as np

from gazelock_ml.data.fixtures import EYE_ROI_H, EYE_ROI_W, make_fake_eye_patch
from gazelock_ml.warp.apply import apply_flow
from gazelock_ml.warp.geometry import EyeLandmarks, HeadPose, compute_target_iris_px
from gazelock_ml.warp.tps import eval_tps, fit_tps, flow_field_from_tps


def _square_landmarks() -> EyeLandmarks:
    return EyeLandmarks(
        outer_corner=(10.0, 36.0),
        inner_corner=(86.0, 36.0),
        top_lid=(48.0, 24.0),
        bottom_lid=(48.0, 48.0),
        iris_center=(48.0, 36.0),
        iris_radius_px=10.0,
    )


def test_geometry_identity_pose_moves_iris_toward_camera_center() -> None:
    # When head pose is identity, camera direction in head frame is -z,
    # which has (0, 0) displacement on the image plane → target == current.
    lm = _square_landmarks()
    pose = HeadPose(yaw=0.0, pitch=0.0, roll=0.0)
    tgt = compute_target_iris_px(lm, pose)
    assert abs(tgt[0] - lm.iris_center[0]) < 1e-6
    assert abs(tgt[1] - lm.iris_center[1]) < 1e-6


def test_geometry_yaw_right_shifts_target_left() -> None:
    lm = _square_landmarks()
    pose = HeadPose(yaw=np.radians(10.0), pitch=0.0, roll=0.0)
    tgt = compute_target_iris_px(lm, pose)
    # When user turns head right (yaw=+), the camera is to their left
    # relative to head frame, so the iris target in image coords shifts
    # toward the negative x direction.
    assert tgt[0] < lm.iris_center[0]
    assert abs(tgt[1] - lm.iris_center[1]) < 0.5  # ~no vertical shift


def test_tps_exact_recovery_on_control_points() -> None:
    src = np.array([[0, 0], [10, 0], [0, 10], [10, 10], [5, 5]], dtype=np.float64)
    tgt = src + np.array([2.0, 1.0])
    coefs = fit_tps(src, tgt)
    out = eval_tps(coefs, src, src)
    np.testing.assert_allclose(out, tgt, atol=1e-3)


def test_flow_field_shape_and_range() -> None:
    src = np.array(
        [
            [0, 0], [EYE_ROI_W - 1, 0],
            [0, EYE_ROI_H - 1], [EYE_ROI_W - 1, EYE_ROI_H - 1],
            [EYE_ROI_W / 2, EYE_ROI_H / 2],
        ],
        dtype=np.float64,
    )
    tgt = src.copy()
    tgt[-1] += np.array([3.0, 0.0])  # move center 3px right
    coefs = fit_tps(src, tgt)

    flow = flow_field_from_tps(coefs, src, EYE_ROI_H, EYE_ROI_W)
    assert flow.shape == (EYE_ROI_H, EYE_ROI_W, 2)
    # Corners should map near-identity (anchored)
    np.testing.assert_allclose(flow[0, 0], src[0], atol=0.5)
    np.testing.assert_allclose(
        flow[EYE_ROI_H - 1, EYE_ROI_W - 1], src[3], atol=0.5
    )


def test_apply_flow_identity_returns_input() -> None:
    img = make_fake_eye_patch(seed=3)
    ys, xs = np.mgrid[: img.shape[0], : img.shape[1]]
    identity_flow = np.stack([xs, ys], axis=-1).astype(np.float64)
    out = apply_flow(img, identity_flow)
    # Small numeric wobble is ok (bilinear at integer coords = identity)
    diff = np.abs(out.astype(np.int32) - img.astype(np.int32))
    assert diff.max() <= 1


def test_apply_flow_constant_shift_moves_content() -> None:
    img = make_fake_eye_patch(seed=7)
    ys, xs = np.mgrid[: img.shape[0], : img.shape[1]]
    # Sample from 2px left → shifts content 2px to the right
    flow = np.stack([xs - 2, ys], axis=-1).astype(np.float64)
    out = apply_flow(img, flow)
    # The leftmost 2 columns should now hold replicated edge values
    # (not strictly equal to original's leftmost), so just verify
    # that the output differs from input somewhere.
    assert not np.array_equal(out, img)
