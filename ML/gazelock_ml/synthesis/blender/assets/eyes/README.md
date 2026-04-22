# CC0 Eye Assets

All assets in this directory are CC0 / public domain. See `MANIFEST.json`
for sources. Binaries are gitignored — only the manifest + this README
live in git.

## Fetching

    uv run python ML/gazelock_ml/synthesis/blender/fetch_assets.py

The fetcher downloads each asset, verifies the SHA-256, and writes it here.

## Adding new assets

1. Find a CC0-licensed 3D eye model
2. Add an entry to `MANIFEST.json` with URL, author, license link, SHA-256
3. Run `fetch_assets.py` to verify the download
4. Commit only `MANIFEST.json` — never the binary

## Compliance

`test_asset_licenses.py` rejects any non-CC0 entry or file containing
forbidden license strings.
