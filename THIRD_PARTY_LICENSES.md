# Third-Party Licenses

GazeLock's Phase 2c training data pipeline uses these third-party
datasets and assets. All are MIT or CC0 (commercial-safe).

## Datasets (fetched by user, not committed)

### Microsoft DigiFace-1M
- Source: https://github.com/microsoft/DigiFace1M
- License: MIT
- Used for: Full-face photometric + landmark supervision

### Microsoft FaceSynthetics
- Source: https://github.com/microsoft/FaceSynthetics
- License: MIT
- Used for: Face images with landmark annotations

## Assets (fetched by user via fetch_assets.py)

See the MANIFEST.json files under
`ML/gazelock_ml/synthesis/blender/assets/` for the authoritative list.

All assets must be CC0 per
`ML/gazelock_ml/tests/synthesis/test_asset_licenses.py`.
