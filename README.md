# GazeLock

Real-time eye-gaze correction for video calls on macOS.

GazeLock installs a virtual camera that appears as "GazeLock Camera" in any video app (Zoom, Meet, Teams, FaceTime, OBS). When enabled, it redirects your eye gaze so you appear to look directly into the camera, even while you are looking at the screen.

## Status

**v2, pre-alpha.** Under active reconstruction. Not yet functional.

## Requirements

- macOS 15 Sequoia or later
- Apple Silicon (M1 or newer)
- A webcam (built-in or external)

## Build from source

Prerequisites:

```bash
brew install xcodegen swiftlint
export DEVELOPMENT_TEAM=YOUR_APPLE_TEAM_ID
```

Build:

```bash
make build
```

Run tests:

```bash
make test
```

## Privacy

GazeLock processes all video locally on your device. No video or telemetry leaves your machine. Zero network traffic by default. The optional update check (opt-in) polls the GitHub Releases API and nothing else.

## Licensing notes

The application code is MIT-licensed (see `LICENSE`).

Starting in Phase 2, training code uses **UnityEyes** synthetic data (free for academic and non-commercial use). If you plan to redistribute trained weights commercially, you must replace the UnityEyes data pipeline with a permissive alternative.

## Contributing

See `CONTRIBUTING.md`.
