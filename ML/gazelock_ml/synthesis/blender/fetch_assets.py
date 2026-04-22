"""Download approved CC0 assets into the local tree.

Reads the MANIFEST.json files, downloads each asset if missing, and
verifies the SHA-256 hash. Idempotent.

Usage:
    uv run python ML/gazelock_ml/synthesis/blender/fetch_assets.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ASSETS_ROOT = Path(__file__).parent / "assets"
MANIFESTS = [
    ASSETS_ROOT / "eyes" / "MANIFEST.json",
    ASSETS_ROOT / "hdri" / "MANIFEST.json",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    print(f"  fetching {url} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, dest.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def fetch_manifest(manifest_path: Path) -> int:
    if not manifest_path.exists():
        print(f"SKIP {manifest_path} (not found)")
        return 0
    manifest = json.loads(manifest_path.read_text())
    errors = 0
    for asset in manifest["assets"]:
        dest = manifest_path.parent / asset["filename"]
        if asset["license"] != "CC0":
            print(f"ERROR: {asset['id']} has non-CC0 license {asset['license']}")
            errors += 1
            continue
        if dest.exists():
            actual = _sha256(dest)
            expected = asset.get("sha256")
            if expected and actual == expected:
                print(f"  OK   {asset['id']} (checksum match)")
                continue
            print(f"  WARN {asset['id']} exists but checksum mismatch; re-downloading")
            dest.unlink()
        try:
            _download(asset["source_url"], dest)
        except Exception as exc:
            print(f"  FAIL {asset['id']}: {exc}")
            errors += 1
            continue
        actual = _sha256(dest)
        if asset.get("sha256") and actual != asset["sha256"]:
            print(f"  FAIL {asset['id']}: checksum mismatch after download")
            errors += 1
    return errors


def main() -> int:
    total_errors = 0
    for m in MANIFESTS:
        print(f"== {m.parent.name} ==")
        total_errors += fetch_manifest(m)
    if total_errors:
        print(f"{total_errors} error(s)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
