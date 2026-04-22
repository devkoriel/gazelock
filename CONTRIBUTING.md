# Contributing to GazeLock

## Dev setup

1. Install tools: `brew install xcodegen swiftlint create-dmg`
2. Set your Apple Developer Team ID:
   ```bash
   export DEVELOPMENT_TEAM=YOUR_TEAM_ID
   ```
3. Generate the Xcode project: `make generate`
4. Open in Xcode: `open GazeLock.xcodeproj`
5. Build from command line: `make build`
6. Run tests: `make test`

## Code style

- Swift 5.9+ with `SWIFT_STRICT_CONCURRENCY = complete`
- SwiftLint enforced on every PR (`make lint`)
- File size guideline: 200–400 lines typical; 800 max
- One clear responsibility per file
- Immutability by default (`let` over `var` unless mutation is required)

## PR flow

1. Create a branch from `main`
2. Follow TDD where the change is a feature or bug fix: failing test → minimal impl → verify pass → refactor → commit
3. Commits use conventional format: `type(scope): description`
   - Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`, `ops`
4. Run `make verify` before pushing (lint + test + build)
5. Open a PR; CI will run `make verify` on macOS 15 / Apple Silicon
6. Address review comments; squash + merge after approval

## Running ML training locally

(Phase 2 and later.) See `ML/training/README.md` once Phase 2 lands.

## Reference

- Design spec: `docs/superpowers/specs/2026-04-22-gazelock-design.md`
- Architecture: `docs/architecture.md` (maintained, contributor-facing)
- Research references: `docs/reference/`
