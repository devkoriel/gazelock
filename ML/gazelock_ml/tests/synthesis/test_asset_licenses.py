"""Compliance scan: fail if any asset violates CC0 promise."""

import json
from pathlib import Path

import pytest

ASSETS_ROOT = (
    Path(__file__).resolve().parents[2] / "synthesis" / "blender" / "assets"
)
BAD_TOKENS = ["non-commercial", "research only", "cc-by-nc", "all rights reserved"]


def _manifest_paths() -> list[Path]:
    return list(ASSETS_ROOT.rglob("MANIFEST.json"))


@pytest.mark.parametrize(
    "manifest_path",
    _manifest_paths() or [pytest.param(None, marks=pytest.mark.skip(reason="no manifests yet"))],
)
def test_every_manifest_asset_is_cc0(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    for asset in manifest.get("assets", []):
        assert asset["license"] == "CC0", (
            f"{manifest_path}: asset {asset.get('id')} has license "
            f"{asset.get('license')!r} (must be CC0)"
        )


@pytest.mark.parametrize(
    "manifest_path",
    _manifest_paths() or [pytest.param(None, marks=pytest.mark.skip(reason="no manifests yet"))],
)
def test_manifest_license_contract_present(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    assert "license_contract" in manifest


def test_readmes_mention_cc0() -> None:
    readmes = list(ASSETS_ROOT.rglob("README.md"))
    if not readmes:
        pytest.skip("no READMEs yet")
    for readme in readmes:
        text = readme.read_text().lower()
        assert "cc0" in text or "public domain" in text, (
            f"{readme} does not mention CC0"
        )


def test_no_files_carry_forbidden_license_strings() -> None:
    if not ASSETS_ROOT.exists():
        pytest.skip("assets dir does not exist yet")
    for path in ASSETS_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".png", ".jpg", ".hdr", ".glb", ".blend"}:
            continue
        if path.name.startswith("."):
            continue
        try:
            text = path.read_text(errors="ignore").lower()
        except UnicodeDecodeError:
            continue
        for tok in BAD_TOKENS:
            assert tok not in text, f"{path} contains forbidden token {tok!r}"
