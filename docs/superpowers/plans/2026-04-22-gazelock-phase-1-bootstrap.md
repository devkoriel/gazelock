# GazeLock v2 — Phase 1: Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the fresh GazeLock v2 project skeleton — archived prior prototype, XcodeGen-driven Xcode project, minimal SwiftUI main app, empty Camera Extension bundle, placeholder test, Makefile, CI. End state: `make build` produces a signed debug `GazeLock.app` containing an embedded `GazeLockCameraExtension.systemextension`; `make test` passes; the CI workflow runs green on a new PR.

**Architecture:** Two-bundle Swift build per spec §5.1 — main app (`GazeLock`) and system extension (`GazeLockCameraExtension`). XcodeGen generates the `.xcodeproj` deterministically from `project.yml`. SwiftLint enforces style. Makefile wraps common operations. GitHub Actions verifies every PR on macOS 15 runners.

**Tech Stack:** Swift 5.9+, SwiftUI, AppKit, AVFoundation, CoreMediaIO, XcodeGen 2.x, SwiftLint 0.54+, xcodebuild (Xcode 16+ / macOS 15 SDK), GitHub Actions.

**Design spec:** `docs/superpowers/specs/2026-04-22-gazelock-design.md` — canonical reference for all decisions.

**Phase scope:** Phase 1 of 4. Phases 2–4 will be planned separately after Phase 1 executes:
- **Phase 2** — ML training pipeline (PyTorch, UnityEyes + FFHQ, refiner UNet, Core ML export, quality metrics).
- **Phase 3** — Swift ML inference + UI (Vision landmarks, 1€/Kalman smoothing, analytic warp, Metal shader, Core ML refiner, XPC, popover, main window, onboarding).
- **Phase 4** — Release (blind A/B harness, full tests, signing, notarization, DMG, release workflow).

**Prerequisites on the developer machine:**
- macOS 15+ on Apple Silicon (M1 or newer)
- Xcode 16+ installed (`xcode-select -p` must print Xcode 16+ path)
- Homebrew installed
- Apple Developer account + Developer ID certificate; set `DEVELOPMENT_TEAM` env var to your Team ID (found via `security find-identity -p codesigning -v` or in the Apple Developer portal)
- No running `GazeLock.app` process (quit any prior install first)

---

## File Structure

Files created by Phase 1, relative to the repo root `/Users/koriel/Development/gazelock/`:

| Path | Responsibility |
|---|---|
| `.gitignore` | Ignore build artifacts, generated project, secrets, ML tooling scratch |
| `LICENSE` | MIT license text |
| `README.md` | Project overview, install, build-from-source, privacy, license note |
| `CONTRIBUTING.md` | Dev setup, PR flow, code style |
| `.swiftlint.yml` | Lint rules |
| `.tool-versions` | asdf-style version pins |
| `project.yml` | XcodeGen project definition |
| `Makefile` | Common operations (generate, build, test, lint, clean) |
| `Sources/GazeLock/App/GazeLockApp.swift` | SwiftUI `App` entry point |
| `Sources/GazeLock/App/AppDelegate.swift` | `NSApplicationDelegate` for menu-bar lifecycle |
| `Sources/GazeLock/Resources/Info.plist` | Main app bundle info |
| `Sources/GazeLock/GazeLock.entitlements` | Hardened runtime + camera |
| `Extension/main.swift` | `CMIOExtensionProvider.startService` entry |
| `Extension/CameraExtensionProvider.swift` | Minimal `CMIOExtensionProviderSource` |
| `Extension/CameraExtensionDevice.swift` | Minimal `CMIOExtensionDeviceSource` (static color frames) |
| `Extension/CameraExtensionStream.swift` | Minimal `CMIOExtensionStreamSource` |
| `Extension/Info.plist` | Extension bundle info |
| `Extension/GazeLockCameraExtension.entitlements` | Extension entitlements |
| `Tests/GazeLockTests/PlaceholderTests.swift` | One passing unit test (scaffolding check) |
| `scripts/bootstrap.sh` | One-shot env setup: `brew install` + `xcodegen generate` |
| `.github/workflows/build.yml` | CI: generate + build + test + lint |

Archived to `/Users/koriel/Development/gazelock.archive-2026-01-26/`: everything else from the prior prototype.

Preserved in place: `.git/`, `docs/`, `.superpowers/`.

---

## Task 1: Archive prior prototype

**Files:**
- Move: all items in `/Users/koriel/Development/gazelock/` except `.git`, `docs`, `.superpowers` → `/Users/koriel/Development/gazelock.archive-2026-01-26/`

- [ ] **Step 1: Create archive destination**

Run:
```bash
mkdir -p /Users/koriel/Development/gazelock.archive-2026-01-26
```

Expected: directory created (or already exists; mkdir -p is idempotent).

- [ ] **Step 2: Sanity-list current contents**

Run:
```bash
ls -A /Users/koriel/Development/gazelock/
```

Expected to include: `.claude`, `.git`, `.gitignore`, `.playwright-mcp`, `.serena`, `.superpowers`, `.swiftlint.yml`, `.tool-versions`, `.venv`, `build`, `GazeLock`, `GazeLock.xcodeproj`, `GazeLockCameraExtension`, `Makefile`, `README.md`, `Screenshot 2026-01-26 at 9.41.38 PM.png`, `Tests`, `docs`, `project.yml`, `scripts`.

- [ ] **Step 3: Move everything except .git, docs, .superpowers**

Run:
```bash
cd /Users/koriel/Development/gazelock
for item in $(ls -A | grep -v -E '^(\.git|docs|\.superpowers)$'); do
  mv "$item" /Users/koriel/Development/gazelock.archive-2026-01-26/
done
```

- [ ] **Step 4: Verify working tree is now minimal**

Run:
```bash
ls -A /Users/koriel/Development/gazelock/
```

Expected output exactly:
```
.git
.superpowers
docs
```

- [ ] **Step 5: Verify archive has the moved content**

Run:
```bash
ls /Users/koriel/Development/gazelock.archive-2026-01-26/
```

Expected: non-empty list including `GazeLock/`, `GazeLockCameraExtension/`, `Makefile`, etc.

- [ ] **Step 6: Commit the reset point**

Run:
```bash
cd /Users/koriel/Development/gazelock
git commit --allow-empty -m "ops: archive prior prototype to gazelock.archive-2026-01-26/

Clean reset point for v2 full rebuild per design spec §3.
Prior source, build artifacts, and generated project moved to sibling
directory. Working tree contains .git/, docs/, .superpowers/ only."
```

Expected: commit succeeds; `git log --oneline` shows three commits (spec add, spec fixes, archive).

---

## Task 2: Install dev tools

**Files:** none (uses Homebrew).

- [ ] **Step 1: Install XcodeGen**

Run:
```bash
brew install xcodegen
```

Expected: installs or reports "already installed" and exits 0.

- [ ] **Step 2: Install SwiftLint**

Run:
```bash
brew install swiftlint
```

- [ ] **Step 3: Install create-dmg (used in Phase 4; install now)**

Run:
```bash
brew install create-dmg
```

- [ ] **Step 4: Verify versions**

Run:
```bash
xcodegen --version && swiftlint version && xcodebuild -version | head -1
```

Expected: XcodeGen ≥ 2.38, SwiftLint ≥ 0.54, Xcode ≥ 16.0.

- [ ] **Step 5: Confirm Team ID is set**

Run:
```bash
echo "DEVELOPMENT_TEAM=${DEVELOPMENT_TEAM:-UNSET}"
```

Expected: prints a 10-char Team ID (e.g., `DEVELOPMENT_TEAM=ABCDE12345`). If `UNSET`, set it before proceeding:

```bash
# Add to your shell profile (~/.zshrc):
export DEVELOPMENT_TEAM=YOUR_TEAM_ID_HERE
```

---

## Task 3: Create root meta files

**Files:**
- Create: `.gitignore`
- Create: `LICENSE`
- Create: `README.md`
- Create: `CONTRIBUTING.md`

- [ ] **Step 1: Create .gitignore**

Write `/Users/koriel/Development/gazelock/.gitignore`:

```gitignore
# macOS
.DS_Store
.AppleDouble
.LSOverride
._*

# Xcode — we use XcodeGen; .xcodeproj is generated
*.xcodeproj
*.xcworkspace
build/
DerivedData/
xcuserdata/
*.xcuserstate
*.xcscmblueprint
*.xccheckout
*.moved-aside

# Swift Package Manager
.build/
.swiftpm/
Package.resolved

# Testing
*.xcresult
coverage/

# Python / ML (used in Phase 2)
.venv/
__pycache__/
*.pyc
*.pyo
*.egg-info/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Secrets
.env
.env.*
*.p12
*.mobileprovision
*.provisionprofile

# Distribution
*.dmg
*.pkg

# Visual-companion session files
.superpowers/

# Agent tooling scratch
.serena/
.claude/
.playwright-mcp/

# NOTE: ML weights (.mlpackage) are tracked via Git LFS starting in
# Phase 2 — intentionally NOT in this .gitignore. weights/ is committed.
```

- [ ] **Step 2: Create LICENSE (MIT)**

Write `/Users/koriel/Development/gazelock/LICENSE`:

```
MIT License

Copyright (c) 2026 GazeLock contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 3: Create README.md**

Write `/Users/koriel/Development/gazelock/README.md`:

```markdown
# GazeLock

Real-time eye-gaze correction for video calls on macOS.

GazeLock installs a virtual camera that appears as "GazeLock Camera" in any video app (Zoom, Meet, Teams, FaceTime, OBS). When enabled, it redirects your eye gaze so you appear to look directly into the camera, even while you are looking at the screen.

## Status

**v2, pre-alpha.** Under active reconstruction per the design spec at `docs/superpowers/specs/2026-04-22-gazelock-design.md`. Not yet functional.

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

Starting in Phase 2, training code uses **UnityEyes** synthetic data (free for academic and non-commercial use). GazeLock's distribution is non-commercial (open-source publish + personal use + portfolio), so this is compatible. If you plan to redistribute trained weights commercially, you must replace the UnityEyes data pipeline. See `docs/superpowers/specs/2026-04-22-gazelock-design.md` §7.1.

## Contributing

See `CONTRIBUTING.md`.
```

- [ ] **Step 4: Create CONTRIBUTING.md**

Write `/Users/koriel/Development/gazelock/CONTRIBUTING.md`:

```markdown
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
```

- [ ] **Step 5: Stage and commit root meta files**

Run:
```bash
cd /Users/koriel/Development/gazelock
git add .gitignore LICENSE README.md CONTRIBUTING.md
git commit -m "chore: add root meta files (.gitignore, LICENSE, README, CONTRIBUTING)"
```

---

## Task 4: Create tool configs

**Files:**
- Create: `.swiftlint.yml`
- Create: `.tool-versions`

- [ ] **Step 1: Create .swiftlint.yml**

Write `/Users/koriel/Development/gazelock/.swiftlint.yml`:

```yaml
included:
  - Sources
  - Extension
  - Tests

excluded:
  - .build
  - DerivedData
  - build
  - "*.xcodeproj"

disabled_rules:
  - trailing_whitespace
  - todo
  - force_try  # init-site try! in CameraExtensionStream is acceptable; developer-error only

opt_in_rules:
  - empty_count
  - empty_string
  - explicit_init
  - fatal_error_message
  - first_where
  - force_unwrapping
  - implicitly_unwrapped_optional
  - literal_expression_end_indentation
  - multiline_arguments
  - multiline_function_chains
  - operator_usage_whitespace
  - overridden_super_call
  - redundant_nil_coalescing
  - sorted_first_last
  - unused_import
  - vertical_parameter_alignment_on_call

line_length:
  warning: 120
  error: 160
  ignores_urls: true
  ignores_function_declarations: false
  ignores_comments: true

function_body_length:
  warning: 60
  error: 100

type_body_length:
  warning: 300
  error: 500

file_length:
  warning: 500
  error: 800

identifier_name:
  min_length: 2
  max_length: 50
  excluded:
    - id
    - x
    - y
    - z
```

- [ ] **Step 2: Create .tool-versions**

Write `/Users/koriel/Development/gazelock/.tool-versions`:

```
xcodegen 2.42.0
swiftlint 0.57.0
```

(Pin versions to match CI; adjust to whatever `xcodegen --version` / `swiftlint version` actually print on your machine.)

- [ ] **Step 3: Stage and commit**

Run:
```bash
cd /Users/koriel/Development/gazelock
git add .swiftlint.yml .tool-versions
git commit -m "chore: add SwiftLint config and tool version pins"
```

---

## Task 5: Create XcodeGen project.yml

**Files:**
- Create: `project.yml`

This defines three targets: `GazeLock` (main app), `GazeLockCameraExtension` (system extension bundle embedded in the app), and `GazeLockTests` (unit tests for the app).

- [ ] **Step 1: Write project.yml**

Write `/Users/koriel/Development/gazelock/project.yml`:

```yaml
name: GazeLock

options:
  bundleIdPrefix: com.gazelock
  deploymentTarget:
    macOS: "15.0"
  xcodeVersion: "16.0"
  generateEmptyDirectories: true
  groupSortPosition: top
  createIntermediateGroups: true

settings:
  base:
    MARKETING_VERSION: "0.1.0"
    CURRENT_PROJECT_VERSION: "1"
    SWIFT_VERSION: "5.9"
    MACOSX_DEPLOYMENT_TARGET: "15.0"
    SWIFT_STRICT_CONCURRENCY: complete
    ENABLE_USER_SCRIPT_SANDBOXING: true
    DEVELOPMENT_TEAM: ${DEVELOPMENT_TEAM}
    CODE_SIGN_STYLE: Automatic
    ENABLE_HARDENED_RUNTIME: true

configs:
  Debug:
    SWIFT_OPTIMIZATION_LEVEL: "-Onone"
    ENABLE_TESTABILITY: true
    SWIFT_ACTIVE_COMPILATION_CONDITIONS: DEBUG
  Release:
    SWIFT_OPTIMIZATION_LEVEL: "-O"
    ENABLE_TESTABILITY: false
    SWIFT_ACTIVE_COMPILATION_CONDITIONS: RELEASE

targets:
  GazeLock:
    type: application
    platform: macOS
    sources:
      - path: Sources/GazeLock
    resources:
      - path: Sources/GazeLock/Resources
    info:
      path: Sources/GazeLock/Resources/Info.plist
      properties:
        NSCameraUsageDescription: "GazeLock needs camera access to apply real-time gaze correction."
        NSSystemExtensionUsageDescription: "GazeLock installs a virtual camera extension so video apps can use the corrected feed."
        LSUIElement: true
        CFBundleShortVersionString: "$(MARKETING_VERSION)"
        CFBundleVersion: "$(CURRENT_PROJECT_VERSION)"
    settings:
      base:
        PRODUCT_BUNDLE_IDENTIFIER: com.gazelock.GazeLock
        PRODUCT_NAME: GazeLock
        CODE_SIGN_ENTITLEMENTS: Sources/GazeLock/GazeLock.entitlements
        INFOPLIST_FILE: Sources/GazeLock/Resources/Info.plist
        LD_RUNPATH_SEARCH_PATHS:
          - "@executable_path/../Frameworks"
    dependencies:
      - target: GazeLockCameraExtension
        embed: true
        codeSign: true

  GazeLockCameraExtension:
    type: system-extension
    platform: macOS
    sources:
      - path: Extension
    info:
      path: Extension/Info.plist
      properties:
        CFBundleShortVersionString: "$(MARKETING_VERSION)"
        CFBundleVersion: "$(CURRENT_PROJECT_VERSION)"
        NSExtension:
          NSExtensionPointIdentifier: com.apple.cmio.CMIOExtensionProvider
          NSExtensionPrincipalClass: "$(PRODUCT_MODULE_NAME).CameraExtensionProvider"
    settings:
      base:
        PRODUCT_BUNDLE_IDENTIFIER: com.gazelock.GazeLock.CameraExtension
        PRODUCT_NAME: GazeLockCameraExtension
        CODE_SIGN_ENTITLEMENTS: Extension/GazeLockCameraExtension.entitlements
        INFOPLIST_FILE: Extension/Info.plist
        SYSTEM_EXTENSION_INSTALL_PATH: "$(SYSTEM_EXTENSIONS_FOLDER_PATH)"
        SKIP_INSTALL: true

  GazeLockTests:
    type: bundle.unit-test
    platform: macOS
    sources:
      - path: Tests/GazeLockTests
    dependencies:
      - target: GazeLock
    settings:
      base:
        PRODUCT_BUNDLE_IDENTIFIER: com.gazelock.GazeLockTests
        PRODUCT_NAME: GazeLockTests
        TEST_HOST: "$(BUILT_PRODUCTS_DIR)/GazeLock.app/Contents/MacOS/GazeLock"
        BUNDLE_LOADER: "$(TEST_HOST)"
        GENERATE_INFOPLIST_FILE: true

schemes:
  GazeLock:
    build:
      targets:
        GazeLock: all
        GazeLockCameraExtension: all
    run:
      config: Debug
    test:
      config: Debug
      targets:
        - GazeLockTests
    profile:
      config: Release
    analyze:
      config: Debug
    archive:
      config: Release
```

- [ ] **Step 2: Stage and commit**

Run:
```bash
cd /Users/koriel/Development/gazelock
git add project.yml
git commit -m "chore: add XcodeGen project.yml for macOS 15 / Apple Silicon targets"
```

---

## Task 6: Create main app bundle files

**Files:**
- Create: `Sources/GazeLock/App/GazeLockApp.swift`
- Create: `Sources/GazeLock/App/AppDelegate.swift`
- Create: `Sources/GazeLock/Resources/Info.plist`
- Create: `Sources/GazeLock/GazeLock.entitlements`

The main app is a menu-bar utility (`LSUIElement: true`). For Phase 1 it only shows a menu-bar icon with a single "Quit" menu item — real popover content lands in Phase 3.

- [ ] **Step 1: Create directory structure**

Run:
```bash
cd /Users/koriel/Development/gazelock
mkdir -p Sources/GazeLock/App Sources/GazeLock/Resources
```

- [ ] **Step 2: Write GazeLockApp.swift**

Write `/Users/koriel/Development/gazelock/Sources/GazeLock/App/GazeLockApp.swift`:

```swift
import SwiftUI

@main
struct GazeLockApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        Settings {
            EmptyView()
        }
    }
}
```

- [ ] **Step 3: Write AppDelegate.swift**

Write `/Users/koriel/Development/gazelock/Sources/GazeLock/App/AppDelegate.swift`:

```swift
import AppKit

final class AppDelegate: NSObject, NSApplicationDelegate {
    private var statusItem: NSStatusItem?

    func applicationDidFinishLaunching(_ notification: Notification) {
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        if let button = item.button {
            button.image = NSImage(systemSymbolName: "eye", accessibilityDescription: "GazeLock")
        }

        let menu = NSMenu()
        menu.addItem(NSMenuItem(
            title: "GazeLock (Phase 1 — no correction yet)",
            action: nil,
            keyEquivalent: ""
        ))
        menu.addItem(NSMenuItem.separator())
        menu.addItem(NSMenuItem(
            title: "Quit GazeLock",
            action: #selector(NSApplication.terminate(_:)),
            keyEquivalent: "q"
        ))
        item.menu = menu

        statusItem = item
    }
}
```

- [ ] **Step 4: Write Info.plist**

Write `/Users/koriel/Development/gazelock/Sources/GazeLock/Resources/Info.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>$(EXECUTABLE_NAME)</string>
    <key>CFBundleIdentifier</key>
    <string>$(PRODUCT_BUNDLE_IDENTIFIER)</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>$(PRODUCT_NAME)</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>$(MARKETING_VERSION)</string>
    <key>CFBundleVersion</key>
    <string>$(CURRENT_PROJECT_VERSION)</string>
    <key>LSMinimumSystemVersion</key>
    <string>$(MACOSX_DEPLOYMENT_TARGET)</string>
    <key>LSUIElement</key>
    <true/>
    <key>NSCameraUsageDescription</key>
    <string>GazeLock needs camera access to apply real-time gaze correction.</string>
    <key>NSSystemExtensionUsageDescription</key>
    <string>GazeLock installs a virtual camera extension so video apps can use the corrected feed.</string>
</dict>
</plist>
```

- [ ] **Step 5: Write GazeLock.entitlements**

Write `/Users/koriel/Development/gazelock/Sources/GazeLock/GazeLock.entitlements`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.security.app-sandbox</key>
    <false/>
    <key>com.apple.security.device.camera</key>
    <true/>
    <key>com.apple.developer.system-extension.install</key>
    <true/>
    <key>com.apple.security.cs.allow-jit</key>
    <false/>
    <key>com.apple.security.cs.disable-library-validation</key>
    <false/>
</dict>
</plist>
```

- [ ] **Step 6: Stage and commit**

Run:
```bash
cd /Users/koriel/Development/gazelock
git add Sources/GazeLock
git commit -m "feat(app): add minimal menu-bar app scaffolding

Phase 1 scope: status-bar icon + Quit menu. Popover and main window
land in Phase 3."
```

---

## Task 7: Create Camera Extension bundle files

**Files:**
- Create: `Extension/main.swift`
- Create: `Extension/CameraExtensionProvider.swift`
- Create: `Extension/CameraExtensionDevice.swift`
- Create: `Extension/CameraExtensionStream.swift`
- Create: `Extension/Info.plist`
- Create: `Extension/GazeLockCameraExtension.entitlements`

Minimal CMIOExtension that publishes a single device ("GazeLock Camera") emitting solid-color frames at 60 fps. Real pipeline lands in Phase 3; this task just makes the extension installable and visible in video apps.

- [ ] **Step 1: Create directory**

Run:
```bash
cd /Users/koriel/Development/gazelock
mkdir -p Extension
```

- [ ] **Step 2: Write main.swift**

Write `/Users/koriel/Development/gazelock/Extension/main.swift`:

```swift
import Foundation
import CoreMediaIO

let providerSource = CameraExtensionProvider(
    clientQueue: DispatchQueue(label: "com.gazelock.GazeLock.CameraExtension.clientQueue")
)
CMIOExtensionProvider.startService(provider: providerSource.provider)
CFRunLoopRun()
```

- [ ] **Step 3: Write CameraExtensionProvider.swift**

Write `/Users/koriel/Development/gazelock/Extension/CameraExtensionProvider.swift`:

```swift
import Foundation
import CoreMediaIO

final class CameraExtensionProvider: NSObject, CMIOExtensionProviderSource {
    private(set) var provider: CMIOExtensionProvider!
    private var deviceSource: CameraExtensionDevice!

    init(clientQueue: DispatchQueue) {
        super.init()
        provider = CMIOExtensionProvider(source: self, clientQueue: clientQueue)
        deviceSource = CameraExtensionDevice(
            localizedName: "GazeLock Camera",
            deviceID: UUID()
        )
        do {
            try provider.addDevice(deviceSource.device)
        } catch {
            fatalError("Failed to add GazeLock Camera device: \(error)")
        }
    }

    func connect(to client: CMIOExtensionClient) throws {}

    func disconnect(from client: CMIOExtensionClient) {}

    var availableProperties: Set<CMIOExtensionProperty> {
        [.providerManufacturer]
    }

    func providerProperties(
        forProperties properties: Set<CMIOExtensionProperty>
    ) throws -> CMIOExtensionProviderProperties {
        let state = CMIOExtensionProviderProperties(dictionary: [:])
        if properties.contains(.providerManufacturer) {
            state.manufacturer = "GazeLock contributors"
        }
        return state
    }

    func setProviderProperties(_ providerProperties: CMIOExtensionProviderProperties) throws {}
}
```

- [ ] **Step 4: Write CameraExtensionDevice.swift**

Write `/Users/koriel/Development/gazelock/Extension/CameraExtensionDevice.swift`:

```swift
import Foundation
import CoreMediaIO

final class CameraExtensionDevice: NSObject, CMIOExtensionDeviceSource {
    private(set) var device: CMIOExtensionDevice!
    private var streamSource: CameraExtensionStream!

    init(localizedName: String, deviceID: UUID) {
        super.init()
        device = CMIOExtensionDevice(
            localizedName: localizedName,
            deviceID: deviceID,
            legacyDeviceID: deviceID.uuidString,
            source: self
        )

        streamSource = CameraExtensionStream(localizedName: "\(localizedName) Stream")
        do {
            try device.addStream(streamSource.stream)
        } catch {
            fatalError("Failed to add camera stream: \(error)")
        }
    }

    var availableProperties: Set<CMIOExtensionProperty> {
        [.deviceTransportType, .deviceModel]
    }

    func deviceProperties(
        forProperties properties: Set<CMIOExtensionProperty>
    ) throws -> CMIOExtensionDeviceProperties {
        let state = CMIOExtensionDeviceProperties(dictionary: [:])
        if properties.contains(.deviceTransportType) {
            state.transportType = kIOAudioDeviceTransportTypeVirtual
        }
        if properties.contains(.deviceModel) {
            state.model = "GazeLock Virtual Camera v0.1"
        }
        return state
    }

    func setDeviceProperties(_ deviceProperties: CMIOExtensionDeviceProperties) throws {}
}
```

- [ ] **Step 5: Write CameraExtensionStream.swift**

Write `/Users/koriel/Development/gazelock/Extension/CameraExtensionStream.swift`:

```swift
import Foundation
import CoreMediaIO
import CoreVideo
import CoreMedia

final class CameraExtensionStream: NSObject, CMIOExtensionStreamSource {
    private(set) var stream: CMIOExtensionStream!
    let availableFormats: [CMIOExtensionStreamFormat]

    private var streamingCounter: Int = 0
    private var timer: DispatchSourceTimer?
    private let timerQueue = DispatchQueue(
        label: "com.gazelock.GazeLock.CameraExtension.streamTimer",
        qos: .userInteractive
    )

    private let width: Int32 = 1280
    private let height: Int32 = 720
    private let fps: Int32 = 60

    init(localizedName: String) {
        let formatDescription = try! makeFormatDescription(width: width, height: height)
        let videoStreamFormat = CMIOExtensionStreamFormat(
            formatDescription: formatDescription,
            maxFrameDuration: CMTime(value: 1, timescale: fps),
            minFrameDuration: CMTime(value: 1, timescale: fps),
            validFrameDurations: nil
        )
        self.availableFormats = [videoStreamFormat]

        super.init()

        stream = CMIOExtensionStream(
            localizedName: localizedName,
            streamID: UUID(),
            direction: .source,
            clockType: .hostTime,
            source: self
        )
    }

    var availableProperties: Set<CMIOExtensionProperty> {
        [.streamActiveFormatIndex, .streamFrameDuration]
    }

    func streamProperties(
        forProperties properties: Set<CMIOExtensionProperty>
    ) throws -> CMIOExtensionStreamProperties {
        let state = CMIOExtensionStreamProperties(dictionary: [:])
        if properties.contains(.streamActiveFormatIndex) {
            state.activeFormatIndex = 0
        }
        if properties.contains(.streamFrameDuration) {
            state.frameDuration = CMTime(value: 1, timescale: fps)
        }
        return state
    }

    func setStreamProperties(_ streamProperties: CMIOExtensionStreamProperties) throws {}

    func authorizedToStartStream(for client: CMIOExtensionClient) -> Bool { true }

    func startStream() throws {
        streamingCounter += 1
        if timer == nil { startTimer() }
    }

    func stopStream() throws {
        streamingCounter = max(0, streamingCounter - 1)
        if streamingCounter == 0 { stopTimer() }
    }

    private func startTimer() {
        let newTimer = DispatchSource.makeTimerSource(queue: timerQueue)
        newTimer.schedule(deadline: .now(), repeating: .milliseconds(Int(1000 / fps)))
        newTimer.setEventHandler { [weak self] in self?.emitFrame() }
        newTimer.resume()
        timer = newTimer
    }

    private func stopTimer() {
        timer?.cancel()
        timer = nil
    }

    private func emitFrame() {
        guard let pixelBuffer = makeSolidColorPixelBuffer() else { return }
        var formatDesc: CMFormatDescription?
        CMVideoFormatDescriptionCreateForImageBuffer(
            allocator: kCFAllocatorDefault,
            imageBuffer: pixelBuffer,
            formatDescriptionOut: &formatDesc
        )
        guard let desc = formatDesc else { return }

        let hostTime = CMClockGetTime(CMClockGetHostTimeClock())
        var timingInfo = CMSampleTimingInfo(
            duration: CMTime(value: 1, timescale: fps),
            presentationTimeStamp: hostTime,
            decodeTimeStamp: .invalid
        )
        var sampleBuffer: CMSampleBuffer?
        CMSampleBufferCreateReadyWithImageBuffer(
            allocator: kCFAllocatorDefault,
            imageBuffer: pixelBuffer,
            formatDescription: desc,
            sampleTiming: &timingInfo,
            sampleBufferOut: &sampleBuffer
        )
        if let buffer = sampleBuffer {
            stream.send(buffer, discontinuity: [], hostTimeInNanoseconds: mach_absolute_time())
        }
    }

    private func makeSolidColorPixelBuffer() -> CVPixelBuffer? {
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary
        ] as CFDictionary

        var pb: CVPixelBuffer?
        CVPixelBufferCreate(
            kCFAllocatorDefault,
            Int(width),
            Int(height),
            kCVPixelFormatType_32BGRA,
            attrs,
            &pb
        )
        guard let buffer = pb else { return nil }
        CVPixelBufferLockBaseAddress(buffer, [])
        if let base = CVPixelBufferGetBaseAddress(buffer) {
            let count = CVPixelBufferGetBytesPerRow(buffer) * Int(height)
            memset(base, 0x20, count)
        }
        CVPixelBufferUnlockBaseAddress(buffer, [])
        return buffer
    }
}

private func makeFormatDescription(width: Int32, height: Int32) throws -> CMFormatDescription {
    var desc: CMFormatDescription?
    let status = CMVideoFormatDescriptionCreate(
        allocator: kCFAllocatorDefault,
        codecType: kCVPixelFormatType_32BGRA,
        width: width,
        height: height,
        extensions: nil,
        formatDescriptionOut: &desc
    )
    guard status == noErr, let description = desc else {
        throw NSError(domain: "CameraExtensionStream", code: Int(status))
    }
    return description
}
```

- [ ] **Step 6: Write Extension Info.plist**

Write `/Users/koriel/Development/gazelock/Extension/Info.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>$(EXECUTABLE_NAME)</string>
    <key>CFBundleIdentifier</key>
    <string>$(PRODUCT_BUNDLE_IDENTIFIER)</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>$(PRODUCT_NAME)</string>
    <key>CFBundlePackageType</key>
    <string>SYSX</string>
    <key>CFBundleShortVersionString</key>
    <string>$(MARKETING_VERSION)</string>
    <key>CFBundleVersion</key>
    <string>$(CURRENT_PROJECT_VERSION)</string>
    <key>LSMinimumSystemVersion</key>
    <string>$(MACOSX_DEPLOYMENT_TARGET)</string>
    <key>NSExtension</key>
    <dict>
        <key>NSExtensionPointIdentifier</key>
        <string>com.apple.cmio.CMIOExtensionProvider</string>
        <key>NSExtensionPrincipalClass</key>
        <string>$(PRODUCT_MODULE_NAME).CameraExtensionProvider</string>
    </dict>
</dict>
</plist>
```

- [ ] **Step 7: Write Extension entitlements**

Write `/Users/koriel/Development/gazelock/Extension/GazeLockCameraExtension.entitlements`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.security.app-sandbox</key>
    <true/>
    <key>com.apple.security.device.camera</key>
    <true/>
    <key>com.apple.developer.system-extension.install</key>
    <true/>
</dict>
</plist>
```

- [ ] **Step 8: Stage and commit**

Run:
```bash
cd /Users/koriel/Development/gazelock
git add Extension
git commit -m "feat(extension): add minimal CMIOExtension scaffolding

Phase 1 scope: extension bundle with a single 'GazeLock Camera'
device emitting solid-grey 720p60 frames. Real pipeline (Vision +
warp + refiner) lands in Phase 3."
```

---

## Task 8: Create placeholder unit test

**Files:**
- Create: `Tests/GazeLockTests/PlaceholderTests.swift`

- [ ] **Step 1: Create directory**

Run:
```bash
cd /Users/koriel/Development/gazelock
mkdir -p Tests/GazeLockTests
```

- [ ] **Step 2: Write PlaceholderTests.swift**

Write `/Users/koriel/Development/gazelock/Tests/GazeLockTests/PlaceholderTests.swift`:

```swift
import XCTest
@testable import GazeLock

final class PlaceholderTests: XCTestCase {
    func testBundleLoads() {
        let bundle = Bundle(for: Self.self)
        XCTAssertNotNil(bundle, "Test host bundle must load")
    }

    func testAppTarget_hasNonEmptyName() {
        let name = Bundle.main.infoDictionary?["CFBundleName"] as? String
        XCTAssertFalse((name ?? "").isEmpty, "CFBundleName must be populated")
    }
}
```

- [ ] **Step 3: Stage and commit**

Run:
```bash
cd /Users/koriel/Development/gazelock
git add Tests
git commit -m "test: add placeholder tests to verify test target links"
```

---

## Task 9: Create Makefile

**Files:**
- Create: `Makefile`

- [ ] **Step 1: Write Makefile**

Write `/Users/koriel/Development/gazelock/Makefile`:

```makefile
.PHONY: all generate build build-release test test-unit lint lint-fix clean run verify setup help

PROJECT_NAME = GazeLock
SCHEME = GazeLock
CONFIG_DEBUG = Debug
CONFIG_RELEASE = Release
BUILD_DIR = build
DERIVED_DATA = $(BUILD_DIR)/DerivedData

XCODEBUILD = xcodebuild
XCODEBUILD_FLAGS = -project $(PROJECT_NAME).xcodeproj -scheme $(SCHEME) -derivedDataPath $(DERIVED_DATA)
XCPRETTY ?= xcpretty

all: generate build

generate:
	@echo "==> Generating Xcode project from project.yml"
	@xcodegen generate
	@echo "    Done: $(PROJECT_NAME).xcodeproj"

build: generate
	@echo "==> Building $(PROJECT_NAME) (Debug)"
	$(XCODEBUILD) $(XCODEBUILD_FLAGS) -configuration $(CONFIG_DEBUG) build

build-release: generate
	@echo "==> Building $(PROJECT_NAME) (Release)"
	$(XCODEBUILD) $(XCODEBUILD_FLAGS) -configuration $(CONFIG_RELEASE) build

test: generate
	@echo "==> Running tests"
	$(XCODEBUILD) $(XCODEBUILD_FLAGS) -configuration $(CONFIG_DEBUG) test

test-unit: generate
	@echo "==> Running unit tests"
	$(XCODEBUILD) $(XCODEBUILD_FLAGS) -configuration $(CONFIG_DEBUG) \
		-only-testing:GazeLockTests test

lint:
	@echo "==> Linting with SwiftLint"
	@swiftlint lint --config .swiftlint.yml --quiet

lint-fix:
	@echo "==> Auto-fixing lint issues"
	@swiftlint lint --fix --config .swiftlint.yml --quiet

clean:
	@echo "==> Cleaning build artifacts"
	@rm -rf $(BUILD_DIR)
	@rm -rf $(PROJECT_NAME).xcodeproj
	$(XCODEBUILD) $(XCODEBUILD_FLAGS) clean 2>/dev/null || true

run: build
	@echo "==> Launching $(PROJECT_NAME)"
	@open "$(DERIVED_DATA)/Build/Products/$(CONFIG_DEBUG)/$(PROJECT_NAME).app"

verify: lint test
	@echo "==> verify: lint + test passed"

setup:
	@echo "==> Installing dev tools"
	@which xcodegen > /dev/null || brew install xcodegen
	@which swiftlint > /dev/null || brew install swiftlint
	@which create-dmg > /dev/null || brew install create-dmg
	@echo "    Done. Remember to export DEVELOPMENT_TEAM."

help:
	@echo "GazeLock Makefile — available targets:"
	@echo "  make              -> generate + build (Debug)"
	@echo "  make generate     -> xcodegen generate"
	@echo "  make build        -> Debug build"
	@echo "  make build-release-> Release build"
	@echo "  make test         -> All tests"
	@echo "  make test-unit    -> Unit tests only"
	@echo "  make lint         -> SwiftLint"
	@echo "  make lint-fix     -> SwiftLint auto-fix"
	@echo "  make verify       -> lint + test (CI parity)"
	@echo "  make clean        -> Remove build artifacts + generated project"
	@echo "  make run          -> Build and open the app"
	@echo "  make setup        -> Install dev tools via Homebrew"
```

- [ ] **Step 2: Stage and commit**

Run:
```bash
cd /Users/koriel/Development/gazelock
git add Makefile
git commit -m "chore: add Makefile with generate/build/test/lint/verify targets"
```

---

## Task 10: Generate project and verify first build

**Files:** none created; verifies Tasks 5–9.

- [ ] **Step 1: Generate .xcodeproj**

Run:
```bash
cd /Users/koriel/Development/gazelock
make generate
```

Expected: `GazeLock.xcodeproj/` is created; stdout prints "==> Generating Xcode project from project.yml" and "Done".

- [ ] **Step 2: Verify .xcodeproj is gitignored (should not appear in git status)**

Run:
```bash
git status --short
```

Expected: empty output (no untracked changes — .xcodeproj is ignored via `*.xcodeproj` in .gitignore).

- [ ] **Step 3: Build Debug**

Run:
```bash
make build
```

Expected: xcodebuild output ending with `** BUILD SUCCEEDED **`. Takes ~30–90 seconds on first run.

If the build fails with a code-signing error, confirm `DEVELOPMENT_TEAM` is set and that the team has the `com.apple.developer.system-extension.install` entitlement enabled for the bundle identifier.

- [ ] **Step 4: Verify the app bundle exists and embeds the extension**

Run:
```bash
ls build/DerivedData/Build/Products/Debug/GazeLock.app/Contents/Library/SystemExtensions/
```

Expected: one entry named `com.gazelock.GazeLock.CameraExtension.systemextension`.

- [ ] **Step 5: Run tests**

Run:
```bash
make test
```

Expected: xcodebuild test output ending with `** TEST SUCCEEDED **`. Both placeholder tests pass.

- [ ] **Step 6: Run lint**

Run:
```bash
make lint
```

Expected: empty output (no warnings or errors).

- [ ] **Step 7: Launch the app (manual smoke test)**

Run:
```bash
make run
```

Expected behavior:
1. App launches
2. An eye-icon appears in the menu bar (top-right of screen)
3. Click it → menu with "GazeLock (Phase 1 — no correction yet)" header + "Quit GazeLock" item appears
4. Click "Quit GazeLock" → app exits

If the icon doesn't appear, check Console.app for errors from `GazeLock`.

Kill the app if still running:
```bash
killall GazeLock 2>/dev/null || true
```

- [ ] **Step 8: Commit no-op marker (optional — skip if clean)**

If anything in the working tree changed (it shouldn't — xcodeproj and build/ are gitignored), investigate before committing. Otherwise move on.

---

## Task 11: Add CI workflow

**Files:**
- Create: `.github/workflows/build.yml`

- [ ] **Step 1: Create directory**

Run:
```bash
cd /Users/koriel/Development/gazelock
mkdir -p .github/workflows
```

- [ ] **Step 2: Write build.yml**

Write `/Users/koriel/Development/gazelock/.github/workflows/build.yml`:

```yaml
name: Build

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  build-and-test:
    name: Build & Test (macOS 15, Apple Silicon)
    runs-on: macos-15
    env:
      DEVELOPMENT_TEAM: ${{ secrets.DEVELOPMENT_TEAM }}
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Select Xcode 16
        run: sudo xcode-select -s /Applications/Xcode_16.app

      - name: Install dev tools
        run: |
          brew install xcodegen swiftlint

      - name: Show versions
        run: |
          xcodegen --version
          swiftlint version
          xcodebuild -version

      - name: Lint
        run: make lint

      - name: Generate project
        run: make generate

      - name: Build (Debug)
        run: make build

      - name: Test
        run: make test
```

- [ ] **Step 3: Stage and commit**

Run:
```bash
cd /Users/koriel/Development/gazelock
git add .github/workflows/build.yml
git commit -m "ci: add GitHub Actions build workflow (macOS 15 / Apple Silicon)"
```

- [ ] **Step 4: Verify CI locally (parity check)**

Run:
```bash
cd /Users/koriel/Development/gazelock
make verify
```

Expected: lint + test both succeed, final message `==> verify: lint + test passed`.

---

## Task 12: Add bootstrap script

**Files:**
- Create: `scripts/bootstrap.sh`

- [ ] **Step 1: Create directory**

Run:
```bash
cd /Users/koriel/Development/gazelock
mkdir -p scripts
```

- [ ] **Step 2: Write bootstrap.sh**

Write `/Users/koriel/Development/gazelock/scripts/bootstrap.sh`:

```bash
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
```

- [ ] **Step 3: Mark executable**

Run:
```bash
chmod +x /Users/koriel/Development/gazelock/scripts/bootstrap.sh
```

- [ ] **Step 4: Smoke-test the script**

Run:
```bash
cd /Users/koriel/Development/gazelock
./scripts/bootstrap.sh
```

Expected: all dependencies already present (from Task 2), generate succeeds, debug build succeeds, final "Bootstrap complete" message.

- [ ] **Step 5: Stage and commit**

Run:
```bash
cd /Users/koriel/Development/gazelock
git add scripts/bootstrap.sh
git commit -m "chore(scripts): add bootstrap.sh for one-shot env setup"
```

---

## Task 13: Final verification and phase-1 end state

**Files:** none created; end-state sanity check.

- [ ] **Step 1: Clean + full rebuild from scratch**

Run:
```bash
cd /Users/koriel/Development/gazelock
make clean
make verify
```

Expected: clean succeeds; generate succeeds; lint succeeds; tests succeed. Final message `==> verify: lint + test passed`.

- [ ] **Step 2: Confirm the working tree is clean**

Run:
```bash
git status --short
```

Expected: empty output.

- [ ] **Step 3: Review commit log**

Run:
```bash
git log --oneline
```

Expected: roughly 14 commits covering spec + spec fixes + archive + Phase 1 tasks. Every commit message follows conventional-commit format.

- [ ] **Step 4: Tag the phase-1 end**

Run:
```bash
git tag -a phase-1-bootstrap -m "Phase 1: bootstrap complete

Skeleton in place: XcodeGen + SwiftUI menu-bar app + empty
CameraExtension + Makefile + CI + tests. Ready for Phase 2
(ML training pipeline)."
```

- [ ] **Step 5: Record end-state snapshot in docs**

(Optional — useful for future reference.) This step can be skipped if the `phase-1-bootstrap` tag is considered sufficient.

- [ ] **Step 6: Print next-phase handoff summary**

Manual check — the following should now all be true:

1. `make verify` passes (lint + test).
2. `make build` produces `build/DerivedData/Build/Products/Debug/GazeLock.app`.
3. `GazeLock.app/Contents/Library/SystemExtensions/com.gazelock.GazeLock.CameraExtension.systemextension` exists.
4. `make run` launches a menu-bar app with an eye icon and a Quit menu.
5. `.github/workflows/build.yml` exists; opening a PR against `main` triggers CI.
6. `git log --oneline` shows a clean, well-formatted history.
7. The `phase-1-bootstrap` tag points at the last commit.

---

## Notes for the executing engineer

- **TDD doesn't apply here.** Phase 1 is pure plumbing — no user-facing behavior to test-first. Phases 2 and 3 introduce real features and TDD will be the primary discipline there.
- **If CameraExtension compilation fails on `CMIOExtensionDevice(..., source: self)` or `CMIOExtensionStream(..., source: self)`**, you're on an SDK where those initializers don't accept the source param. Fall back to the KVC workaround — after `super.init()`, call `(device as AnyObject).setValue(self, forKey: "source")` / same for stream. Check the latest Apple sample ("CameraExtensionFilteredTemplate" or "SampleCaptureExtension") for the current shape.
- **CMIOExtensionStreamFormat unavailable API error.** Some macOS minor versions restrict this initializer; if so, construct the format via `CMIOExtensionStreamFormat(formatDescription:maxFrameDuration:minFrameDuration:validFrameDurations:)` exactly as written, and confirm you're on Xcode 16 targeting macOS 15 SDK.
- **System extension installation.** Phase 1 does NOT install the extension into the OS — it only produces a bundle that *contains* it. Actual `OSSystemExtensionRequest` submission happens in Phase 3 (onboarding flow). To verify the bundle is well-formed without installing, use `pluginkit -m -i com.gazelock.GazeLock.CameraExtension` after first install; for now, `ls` the bundle as in Task 10 Step 4.
- **Code signing on first run** — if xcodebuild complains about provisioning profiles for the extension, open Xcode once (`open GazeLock.xcodeproj`), select the GazeLock target → Signing & Capabilities, confirm Team = your Team ID, and let Xcode auto-create the provisioning profile. Close and return to `make`.

---

*End of Phase 1 plan.*
