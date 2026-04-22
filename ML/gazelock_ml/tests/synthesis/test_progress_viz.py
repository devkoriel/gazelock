"""Dashboard stats computation tests."""

import json
import time
from pathlib import Path

from gazelock_ml.synthesis.blender.progress_viz import ManifestTail, _fmt_duration


def _write_entry(manifest: Path, frame_id: int, render_seconds: float, rendered_at: int) -> None:
    entry = {
        "frame_id": frame_id,
        "render_seconds": render_seconds,
        "rendered_at": rendered_at,
        "png_path": f"{frame_id:06d}.png",
    }
    with manifest.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")


def test_manifest_tail_reports_zero_until_written(tmp_path: Path) -> None:
    tail = ManifestTail(root=tmp_path, total=100)
    tail.refresh()
    stats = tail.stats()
    assert stats.count == 0
    assert stats.pct == 0.0


def test_manifest_tail_counts_entries(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    now = int(time.time())
    for i in range(5):
        _write_entry(manifest, i, 1.0, now + i)

    tail = ManifestTail(root=tmp_path, total=10)
    tail.refresh()
    stats = tail.stats()
    assert stats.count == 5
    assert stats.total == 10
    assert abs(stats.pct - 0.5) < 1e-9


def test_manifest_tail_rolling_average(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    now = int(time.time())
    for i in range(10):
        _write_entry(manifest, i, 2.0, now + i)

    tail = ManifestTail(root=tmp_path, total=100)
    tail.refresh()
    stats = tail.stats()
    assert abs(stats.avg_seconds - 2.0) < 1e-6


def test_manifest_tail_eta_scales_with_remaining(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    now = int(time.time())
    for i in range(10):
        _write_entry(manifest, i, 3.0, now + i)

    tail = ManifestTail(root=tmp_path, total=100)
    tail.refresh()
    stats = tail.stats()
    assert abs(stats.eta_seconds - 270.0) < 1e-6


def test_manifest_tail_latest_frames_bounded(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    now = int(time.time())
    for i in range(30):
        _write_entry(manifest, i, 1.0, now + i)

    tail = ManifestTail(root=tmp_path, total=100)
    tail.refresh()
    stats = tail.stats()
    assert len(stats.latest_frames) == 20
    assert stats.latest_frames[-1]["frame_id"] == 29


def test_manifest_tail_incremental_read(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    now = int(time.time())
    for i in range(3):
        _write_entry(manifest, i, 1.0, now + i)

    tail = ManifestTail(root=tmp_path, total=100)
    tail.refresh()
    assert tail.stats().count == 3

    for i in range(3, 7):
        _write_entry(manifest, i, 1.0, now + i)
    tail.refresh()
    assert tail.stats().count == 7


def test_fmt_duration_formats() -> None:
    assert _fmt_duration(0) == "—"
    assert _fmt_duration(60) == "1m"
    assert _fmt_duration(3600) == "1h 0m"
    assert _fmt_duration(86400 + 3600 + 120) == "1d 1h 2m"
