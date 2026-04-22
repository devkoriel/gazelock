#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> GazeLock bootstrap"
echo "    Repo root: $REPO_ROOT"
echo

# 1. Homebrew deps
echo "==> Installing Homebrew dependencies"
for tool in xcodegen swiftlint create-dmg; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "    Installing $tool..."
    brew install "$tool"
  else
    echo "    $tool already installed ($("$tool" --version 2>&1 | head -1))"
  fi
done
echo

# 2. Verify DEVELOPMENT_TEAM
if [[ -z "${DEVELOPMENT_TEAM:-}" ]]; then
  echo "!! WARNING: DEVELOPMENT_TEAM is not set."
  echo "   Set it before building:"
  echo "     export DEVELOPMENT_TEAM=YOUR_TEAM_ID"
  echo "   Find your Team ID at https://developer.apple.com/account/#/membership"
  echo
fi

# 3. Generate project
echo "==> Generating Xcode project"
xcodegen generate
echo

# 4. First build
echo "==> Running initial debug build"
make build
echo

echo "==> Bootstrap complete. Next:"
echo "    - Open in Xcode:  open GazeLock.xcodeproj"
echo "    - Run locally:    make run"
echo "    - Run tests:      make test"
echo "    - Run CI parity:  make verify"
