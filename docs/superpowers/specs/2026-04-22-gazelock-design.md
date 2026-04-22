# GazeLock — Design Specification

**Date:** 2026-04-22
**Status:** Approved by user; pending written-spec review
**Version:** v2 design (full rebuild of prior 2026-01-26 prototype)

---

## 1. Overview

GazeLock is a macOS application that corrects the user's eye gaze in real time during video calls. It installs a virtual camera device that any video app (Zoom, Meet, Teams, FaceTime, OBS) can select. When enabled, it redirects the user's eyes so they appear to be looking directly into the camera, even while the user is actually looking at the screen.

This spec describes the **v2 full rebuild**. A prior prototype (2026-01-25/26) established scaffolding — SwiftUI shell, CoreMediaIO Camera Extension, build system — but its ML pipeline produced visible artifacts: wrong iris geometry, plastic eye texture, per-frame flicker, identity drift. The prior source is archived; only the name is retained.

## 2. Goal and quality bar

**Primary goal:** the user will use GazeLock during live online presentations where the audience must not be able to detect that any correction is happening.

**Quality bar:** undetectable to a human observer on the other end of a video call. Not "plausible", not "mostly OK" — undetectable.

**Acceptance gate:** blind A/B with ≥ 20 volunteers each rating 30 clip pairs. Target **< 55 % correct identification** of corrected vs. raw (50 % = pure chance). No weights version ships that fails this gate.

**Explicit failure modes to preclude** (all observed in the prior prototype):
- Wrong iris geometry or head-pose mismatch
- Plastic / "dead-eye" texture loss
- Per-frame flicker when the head is stationary
- Identity drift — eyes stop looking like the user's
- Visible warping halos or sclera seams at the eye-region boundary

## 3. Scope and constraints

| Axis | Decision |
|---|---|
| Correction angle range | 5–15 °, design headroom to 20 ° |
| User conditions | No glasses; typical office / home lighting |
| Shipping intent | Open-source publish (MIT) + personal use + portfolio showcase; non-commercial |
| macOS deployment target | 15.0 Sequoia + |
| Hardware target | Apple Silicon only (M1 +) |
| Integration model | CoreMediaIO Camera Extension |
| Effort budget | 6–10 weeks |

## 4. Research direction

**Warp + tiny learned refiner** (research option B of three considered).

Alternatives considered and rejected:
- **Pure image-to-image regeneration (UNet / GAN).** This was the architecture of the prior prototype's failures. Regenerates eye pixels from scratch conditioned on target gaze; texture loss, identity drift, flicker, and geometric errors all fall out of the design. **Off the table.**
- **Pure classical warp (no ML).** Strong in the 5–15 ° regime, but exhibits a visible sclera-seam failure mode at the range edge. Retained as a fallback architecture; insufficient as the primary path given the "super super natural" bar.
- **Warp + large learned refiner, or fully self-trained model.** Overkill for this regime and harder to keep identity-safe. Marginal quality gain does not justify the complexity and training effort.

**Chosen design.** Analytic warp (landmarks → flow field → Metal warp) followed by a deliberately-narrow "seam hider" refiner (≈ 120 k-param UNet) that inpaints the disoccluded sclera strip using the original eye as a conditioning signal. The refiner is an **inpainter**, not a **generator** — iris pixels remain the user's original pixels, translated in place, never replaced by hallucinated content. This directly kills all four prior failure modes.

Training data is **permissive-license only**: UnityEyes synthetic (self-rendered, non-commercial license) + FFHQ real eye crops + self-captured calibration footage.

---

## 5. System architecture

### 5.1 Two-process topology (forced by Apple)

Camera extensions must run in their own sandboxed process on modern macOS; there is no single-process virtual-camera option.

```
┌────────────────────────┐   XPC (config)   ┌────────────────────────────────┐
│  GazeLock.app          │◄────────────────►│  GazeLockCameraExtension       │
│  (user-facing)         │                   │  (virtual-camera bundle)       │
│                        │                   │                                │
│  Menu bar + popover    │                   │  CMIOExtensionProvider         │
│  Main window           │                   │  ML pipeline                   │
│  Onboarding            │                   │  CMIOExtensionStream           │
│  ML pipeline (preview) │                   │                                │
└──────────┬─────────────┘                   └──────────┬─────────────────────┘
           │ AVCaptureSession                           │ CMIOExtension I/O
           ▼                                            ▼
┌────────────────────────┐                   ┌────────────────────────────────┐
│  Physical camera       │                   │  Zoom / Meet / Teams / OBS /   │
│  (FaceTime HD, USB)    │                   │  any camera consumer           │
└────────────────────────┘                   └────────────────────────────────┘
```

### 5.2 Control plane (config)

Shared XPC service, app-group identifier. Main app ↔ extension exchange small state: on/off, intensity, selected source camera, active preset. Low-frequency, tiny payloads. Settings changes propagate in < 100 ms (one XPC round trip).

### 5.3 Data plane (frames)

Independent per process. Extension pulls from the physical camera directly; main app (when open) does the same for its preview. Both run the full pipeline locally. No frame sharing, no IPC for frame data. On Apple Silicon the 2× pipeline cost is negligible; when the main app is closed, only the extension runs.

### 5.4 Per-frame pipeline (inside each process)

Budget: 16 ms @ 60 fps; target < 20 ms end-to-end.

```
Raw CVPixelBuffer
    ↓ Apple Vision — face + eye landmarks                        ≈ 4–8 ms
    ↓ Temporal landmark smoother (1€ filter + Kalman on iris)    < 1 ms
    ↓ Analytic warp flow-field (3D eyeball + TPS)                ≈ 1 ms
    ↓ Metal compute shader — warp eye ROI only                   ≈ 2 ms
    ↓ Core ML refiner UNet (Neural Engine)                       ≈ 4–6 ms
    ↓ Metal composite — feathered blend back into frame          ≈ 1 ms
Output CVPixelBuffer
    ├→ Extension:  CMIOExtensionStreamSource → consumer
    └→ Main app:   SwiftUI preview view
```

### 5.5 Topology alternatives rejected

- **Extension publishes frames to main app via shared memory** — adds IPC bandwidth + synchronization complexity for no measurable gain on Apple Silicon.
- **Main app runs pipeline, extension forwards** — extension lifecycle is independent of the main app; breaks when the user quits the app mid-call.
- **Single pipeline instance serving multiple consumers** — Apple's extension sandbox forbids cross-process pipeline sharing.

### 5.6 Architecture invariants

- Extension survives main-app termination.
- Settings changes propagate in < 100 ms.
- Pipeline is fully local; zero network, zero telemetry by default.

---

## 6. ML pipeline

### 6.1 Landmark detection — Apple Vision

`VNDetectFaceLandmarksRequest` per frame. Extracts eye contour (~ 8 points per eye), pupil centers, and face yaw / pitch / roll from `VNFaceObservation`. A **3D eyeball center** is derived from yaw + pupil position using the ~ 12 mm iris-to-eyeball-center anatomical offset (near-constant across adults). This enables geometrically-realistic eyeball rotation rather than flat 2D translation.

Apple Vision is chosen over MediaPipe Iris, dlib, or a custom detector because it ships with the SDK (zero licensing friction, zero binary bloat), runs hardware-accelerated on Apple Silicon, and its frame-level accuracy is sufficient — frame-to-frame jitter is solved separately in 6.2.

### 6.2 Temporal smoothing — 1€ filter + Kalman

Landmark jitter is the primary cause of flicker in any warp-based pipeline. Two-layer smoothing:

- **1€ filter** (Casiez et al. 2012) per landmark component. Adaptive low-pass — heavy smoothing when the landmark is stationary, high cutoff during motion to keep lag minimal.
- **Kalman predictor** on the iris center specifically. Iris moves at saccade speeds (~ 900 °/s) and can outrun the 1€ filter. Kalman predicts through the fast transient and re-locks once motion settles.

Target: stationary-state landmark noise **< 0.3 px RMS at 720p**. Above that, flicker becomes visible.

### 6.3 Analytic warp — 3D eyeball + thin-plate spline

Given eyeball center, current iris position, head pose, and target "looking at camera" vector:

1. Model the eyeball as a sphere; iris as a disc on its front surface.
2. Rotate the iris disc around the eyeball center so its normal points at the camera.
3. Project back to the image plane → target iris position + warped iris boundary.
4. Build a thin-plate spline with:
   - **Anchors** (flow = 0): eyelid-edge landmarks and outer sclera boundary.
   - **Targets**: iris center → projected position; iris-perimeter points displaced accordingly.
5. Rasterize the TPS into a dense 64 × 48 flow field per eye.

The anchor lock preserves iris → sclera → eyelid texture transitions. Disoccluded sclera at 15 ° is < 5 % of the eye region; the refiner handles it.

### 6.4 Metal warp shader

A single compute kernel `iris_warp.metal`. Per-pixel bilinear sample at `(x + flow.x, y + flow.y)`. Runs inside the eye ROI only (~ 6 000 pixels total for both eyes); the rest of the frame byte-copies via `MTLBlitCommandEncoder`. Target < 2 ms for both eyes on M1 +.

### 6.5 Refiner UNet — "seam hider", not "eye generator"

**Critical architectural choice.** The prior prototype's generalist refiner was the source of its failures. Ours is narrow by construction.

- **Input** (6 channels, 96 × 72): warped eye region + **original unwarped eye region as a side channel**.
- **Output** (3 channels, 96 × 72): refined eye region.
- **Job:** resolve disoccluded sclera + any local seam the warp introduced, using the original eye as a direct reference. Inpaint only; do not regenerate.

**Architecture.** Depthwise-separable conv UNet, 3 encoder / decoder levels, 32 → 64 → 128 channels, skip connections at each level. Total parameters ≈ 120 k, ≈ 240 KB at float16.

**Why 120 k suffices.** The model is not being asked to invent an eye; it is being asked to fill the ~ 5 % disoccluded sclera strip and clean up warp-induced seams, with the original eye available as direct conditioning. It is a **conditioned inpainter**, not a generator.

Runtime: ≈ 4 ms per eye on Neural Engine via Core ML.

### 6.6 Compositing and intensity

Refined output → alpha-blend back into the full frame via a **feathered Gaussian mask** around the eye ROI (radial falloff over 8 px at the edge). No hard patch boundary.

**Intensity slider (0–100 %)** scales the flow-field magnitude **pre-warp**. 0 % = no flow = raw passthrough. 100 % = full correction. 50 % = half-way gaze (useful if full correction ever lands uncanny for a specific face).

### 6.7 Pipeline invariants

- **Iris pixels are always the user's original pixels, translated** — never replaced by hallucinated content. Directly kills failure modes B (plastic eye) and D (identity drift).
- **Temporal stability is deterministic for all non-refiner stages.** The refiner is trained with a temporal-consistency loss (§ 7.2). Directly kills failure mode C (flicker).
- **ML touches the eye ROI only** — the rest of the frame is byte-copied, preserving skin, hair, and background untouched.

---

## 7. Training pipeline

### 7.1 Data strategy — self-supervised, permissive-only

| Tier | Source | Size | Role |
|---|---|---|---|
| Primary | UnityEyes synthetic | ~ 100 k pairs | Paired ground truth across gaze angles |
| Domain-adapt | FFHQ eye crops | ~ 50 k patches | Cycle consistency + adversarial realism |
| Validation | Self-captured | ~ 2 k frames | Real-world holdout |

**License note.** UnityEyes is free for academic / non-commercial use. The shipping intent (open-source publish + personal use + portfolio) is non-commercial — fully compatible. This is documented in the repository README. If the project ever pivots to commercial sale, UnityEyes is swapped for a Blender-based synthetic pipeline built in-house (~ 2 weeks additional work).

**Why self-supervised works.** The refiner is not learning "what is an eye"; it is learning "given a warped eye and the unwarped original as reference, restore the fine structure the warp broke". UnityEyes' pair-wise supervision teaches this mapping cleanly because we control both warp parameters and ground truth.

### 7.2 Loss composition

| Term | λ | Purpose |
|---|---|---|
| L1 reconstruction | 1.0 | Pixel-level fidelity to ground truth |
| VGG-16 perceptual | 0.2 | Texture fidelity beyond pixel level |
| Identity preservation | 0.5 | Cosine distance on iris-region features (FFHQ-trained eye encoder); locks iris appearance to the original person |
| Temporal consistency | 0.1 | Near-neighbor synthetic frame pairs; penalizes output difference; stabilizes frame-to-frame |

### 7.3 Training procedure

- Framework: **PyTorch 2.x** for training; **coremltools** for export.
- Optimizer: AdamW, lr = 1e-4, cosine schedule, 50 k steps.
- Batch: 32 pairs per step.
- Mixed precision (fp16 forward, fp32 master weights).
- Hardware: M-series GPU via MPS backend; ≈ 6–12 h per full training run. Fits overnight on the MBP.

### 7.4 Validation

**Training-time monitors** (automated):
- PSNR / SSIM vs. UnityEyes held-out test set (sanity only — not predictive of "undetectable")
- Identity cosine similarity on FFHQ crops — must stay > 0.98
- Frame-to-frame L1 on UnityEyes video sequences (flicker proxy)

**Acceptance gate:** blind human A/B (§ 9.1.3).

### 7.5 Core ML export

- `coremltools.convert(...)` with `compute_units=.all`, `minimum_deployment_target=macOS 15`.
- Precision: **float16 only**. No int8 / int4 quantization for v1 — the refiner is already ≈ 240 KB at fp16; quantization historically introduces eye-region quality regressions non-deterministically.
- Export gate: PyTorch vs. Core ML L∞ output diff < 1e-3 on 1 000 test inputs before the model ships.

### 7.6 Versioning and distribution

- Weights: `gazelock-refiner-v{semver}.mlpackage`.
- Tracked in **Git LFS** in the repo; also published as GitHub Release assets.
- Training script, seed, and data manifest (hash of UnityEyes render params + FFHQ crop file list) committed for end-to-end reproducibility.
- The app embeds the weight version in its bundle. Weight updates ship bundled with app updates via the unified GitHub Releases update channel (§ 9.4) — no separate weights-only update path.

---

## 8. UI and interaction model

### 8.1 App shape — menu-bar utility

GazeLock lives in the menu bar. Clicking the menu-bar icon opens a compact popover. A main window exists but is secondary — opened for initial setup, calibration, and advanced diagnostics. During actual presentations the user operates entirely from the popover (or, ideally, forgets GazeLock exists).

### 8.2 Popover — preview-embedded

```
╭────────────────────────────────╮
│ ● GAZELOCK      12ms · 60fps   │
├────────────────────────────────┤
│ ┌─────┬─────┐                  │
│ │BEFORE│AFTER│  live preview   │
│ └─────┴─────┘                  │
│ CORRECTION          [ON  ▢]    │
│ INTENSITY ▬▬▬▬▬▬▬░░░  70       │
├────────────────────────────────┤
│         OPEN WINDOW →          │
╰────────────────────────────────╯
```

The live before/after preview gives a pre-call confidence check — the user can sanity-verify the correction without opening the main window. Pipeline cost of running while the popover is open is negligible on Apple Silicon.

### 8.3 Visual direction — Precision dark

OLED-black background, SF Pro for text, SF Mono for numerics, single neon-green accent (`#00ff88`), technical uppercase letterspaced labels. Survives any wallpaper, reads the same in every environment, photographs well for portfolio material. No hand-rolled material system required (macOS 15 does not ship native Liquid Glass; Precision Dark hits the quality bar without that complexity).

### 8.4 Main window — four tabs

- **Preview.** Large side-by-side raw / warped / refined frames; toggleable flow-field debug overlay; live per-stage perf metrics (landmark / warp / refiner / total ms).
- **Calibration.** 9-point gaze-target grid — user looks at each point while the app records the camera-to-screen offset for their specific physical setup. One-time, ~ 15 seconds. Fine-tunes the "target gaze" direction per rig.
- **Settings.** Default intensity preset, menu-bar icon style, opt-in update check, "Reveal debug log in Finder".
- **About.** App version, weights version, paper citations (STED, GazeDirector, 1€ filter, …), MIT license, repo link.

### 8.5 Onboarding (first launch)

Four-step sheet sequence, total ~ 60 seconds:

1. **Welcome** — "GazeLock corrects your eye gaze during video calls."
2. **Camera permission** — triggers macOS camera prompt.
3. **Install virtual camera** — triggers system-extension approval flow. May require reboot on some setups.
4. **Calibration** — 9-point gaze grid, 15-second mini-task.

On completion, the popover opens with correction **ON** by default.

### 8.6 Menu-bar icon states

| Glyph | Meaning |
|---|---|
| ⚪ (greyscale) | Installed; correction OFF |
| 🔵 (solid) | Correction ON; no consumer connected |
| 🟢 (pulsing) | Correction ON; actively streaming to a consumer (Zoom / Meet / etc.) |
| 🔴 | Error state (camera missing, extension disabled) → click opens main window |

### 8.7 Keyboard shortcuts

- `⌘⇧G` — toggle correction on/off (global).
- `⌘+` / `⌘-` — intensity ± 10 %.
- Menu-bar left-click → popover; right-click → main window.

---

## 9. Testing and distribution

### 9.1 Testing — three layers

#### 9.1.1 Swift app + extension

- **XCTest / Swift Testing** for unit tests.
- **UI tests** for the 4-step onboarding sheet sequence.
- **Mock** `CMIOExtensionProviderSource` for extension-level tests without a physical camera.
- Target coverage ≥ 80 % for non-UI code.

#### 9.1.2 ML pipeline — deterministic stages

Unit tests with known inputs and expected outputs:
- **1€ filter** output matches Casiez et al. reference implementation on a canonical motion trace.
- **Warp flow-field construction** — given synthetic landmarks and target gaze, TPS flow matches expected field to < 0.1 px.
- **Metal warp shader** — render a test frame, diff pixel-exact against a CPU software reference renderer.
- **Compositing** — feathering curve matches spec.

Golden-image tests: full pipeline on synthetic frames, diff against reference output (< 2 % pixel L1).

#### 9.1.3 Model quality — the acceptance gate

**Automated:**
- PSNR / SSIM vs. UnityEyes test set (sanity; not predictive of "undetectable").
- Identity cosine similarity > 0.98 on FFHQ crops.
- Frame-to-frame L1 on UnityEyes video sequences (flicker proxy).
- Per-stage latency on M1 / M2 / M3 (target < 20 ms total).

**Manual — blind A/B human evaluation.** The gate that decides whether weights ship.
- 30 × 5-second clip pairs: (user, raw, looking directly at camera) vs. (user, corrected, looking at screen).
- Random order.
- 20 + volunteers each rate 30 pairs, picking which is "real".
- **Target < 55 % correct** (50 % = pure chance = undetectable).
- **No weights version ships that fails.**

Harness: static HTML + JS hosted on GitHub Pages; anonymous Firebase write for responses.

### 9.2 Build and release pipeline

**Toolchain.**
- `xcodegen` — `project.yml` → `.xcodeproj` (deterministic).
- `xcodebuild` + `xcpretty`.
- `swiftlint` + `swiftformat` on every PR.
- `create-dmg` for installer.
- `codesign --deep --options runtime` with Developer ID.
- `notarytool submit --wait` + `stapler staple` for Gatekeeper acceptance.

**GitHub Actions** (M-series / macOS 15 runners):
- **`build.yml`** — every PR: xcodegen → build → unit tests → SwiftLint.
- **`release.yml`** — on `v*` tag: build → sign → notarize → staple → DMG → GitHub Release with SHA-256 checksums.
- **`ml-eval.yml`** — manual dispatch: runs full quality eval (PSNR / SSIM / identity / latency) against current weights; publishes a markdown report.

### 9.3 Repository structure

```
gazelock/
├── README.md
├── LICENSE                         (MIT)
├── CONTRIBUTING.md
├── project.yml                     (XcodeGen)
├── Makefile
├── .github/workflows/              (build, release, ml-eval)
├── Sources/GazeLock/               (main app)
├── Extension/                      (system extension bundle)
├── Tests/                          (unit, UI, golden-image)
├── ML/
│   ├── training/                   (PyTorch 2.x training code)
│   ├── data/                       (UnityEyes render params, FFHQ manifest)
│   └── eval/                       (quality metrics + blind-AB harness)
├── weights/                        (Git LFS)
├── docs/
│   ├── design.md                   (contributor-facing current design, maintained)
│   ├── architecture.md
│   ├── reference/                  (paper citations, research journal)
│   └── superpowers/specs/          (dated brainstorm artifacts — this spec lives here)
└── scripts/                        (build, release, train)
```

### 9.4 Distribution

- **GitHub Releases** — sole distribution channel. DMG + SHA-256 checksums.
- **No Mac App Store.** Camera Extension sandboxing + UnityEyes licensing make the MAS path not worth the friction for an open-source / non-commercial project.
- **Update check** — opt-in, polls GitHub Releases API weekly. Notifies in the main window if a newer version exists. No Sparkle framework — keeps the codebase minimal.

### 9.5 Privacy and telemetry

**Zero network traffic by default.** No analytics, no crash reporting, no remote configuration.

- Crash logs stay local; main window → Settings → "Reveal in Finder".
- Update check is opt-in, single endpoint (GitHub Releases API), disable-able.
- Declared explicitly in the README privacy section.

### 9.6 Documentation

- **README.md** — install, usage, system requirements, UnityEyes non-commercial note, building from source, privacy statement.
- **CONTRIBUTING.md** — dev setup, PR flow, code style, running ML training locally.
- **docs/design.md** — this spec (canonical artifact from the design phase).
- **docs/architecture.md** — contributor-facing architecture summary.
- **docs/reference/** — paper citations, research journal of what worked and what did not (doubles as portfolio material).

### 9.7 Release invariants

- Every weights version passes the blind-A/B gate before shipping.
- Every release is notarized and stapled — no Gatekeeper warnings for end users.
- Zero network traffic unless the user explicitly opts in.
- Fully reproducible: clone → `make release` → byte-similar signed DMG (modulo signing nonce).

---

## Appendix A — Decisions log

| Question | Decision | Rejected alternatives |
|---|---|---|
| Prior-attempt failure modes | All of A + B + C + D (geometry, texture, flicker, identity) | — |
| Physical setup | A + B (laptop / single-monitor, 5–15 °) | Dual monitor (large angle); unusual camera placement |
| Shipping intent | Publish (A) + personal (B) + showcase (C) | Commercial sale |
| Conditions to support | No glasses (C) + office lighting (E) | Glasses always / sometimes; studio lighting; low light |
| Salvage decision | C — full rebuild, keep name | Keep scaffold only; fresh project, ported files |
| Integration model | A — CoreMediaIO Camera Extension | Per-app plugins; OBS effect plugin; RTMP / NDI |
| macOS target | macOS 15 Sequoia + | 13 Ventura; 14 Sonoma; 26 |
| Hardware | Apple Silicon only | Intel inclusion |
| Research direction | B — warp + tiny refiner | Pure warp only; fully self-trained model |
| App shape | A — menu-bar utility | Hybrid menu-bar + window; window-first pro tool |
| Popover layout | C — preview-embedded | Ultra-minimal; quick-pro (no preview) |
| Visual direction | B — Precision dark | Apple-native refined; Liquid Glass custom-built |

## Appendix B — Deferred to implementation plan

Items intentionally not nailed down in this spec; owned by the implementation plan:

- UnityEyes render-parameter sweep: camera distance range, head-pose distribution, lighting variation
- FFHQ crop selection criteria (minimum face size, sharpness threshold, ethnicity / age diversity)
- Exact `CMIOExtensionProviderSource` subclass layout and lifecycle hooks
- XPC service protocol — concrete message types and versioning
- Main-window window-management rules (window restoration, multi-display, space assignment)
- Calibration UX details (target spacing, dwell time per target, skip option)
- Blind A/B harness hosting specifics (Firebase project vs. simpler Google Form backend)
- Handling of archived prior source (`gazelock.archive-2026-01-26/`) during rebuild
- Pipeline behavior when landmark detection fails or is low-confidence (pass-through raw frame vs. hold previous corrected frame vs. menu-bar warning state)
- Global-shortcut registration mechanism for `⌘⇧G` (NSEvent global monitor vs. CGEventTap; Accessibility-permission flow on first launch)

## Appendix C — References

- Casiez, G., Roussel, N., Vogel, D. **1€ Filter: A Simple Speed-Based Low-Pass Filter for Noisy Input in Interactive Systems.** CHI 2012.
- Wood, E., Baltrušaitis, T., Morency, L.-P., Robinson, P., Bulling, A. **Learning an appearance-based gaze estimator from one million synthesised images.** ETRA 2016. *(UnityEyes)*
- Wood, E., Baltrušaitis, T., Morency, L.-P., Robinson, P., Bulling, A. **GazeDirector: Fully articulated eye gaze redirection in video.** EUROGRAPHICS 2018.
- Kononenko, D., Lempitsky, V. **Learning to look up: realtime monocular gaze correction using machine learning.** CVPR 2015.
- Zheng, Y., Park, S., Zhang, X., De Mello, S., Hilliges, O. **Self-Learning Transformations for Improving Gaze and Head Redirection.** NeurIPS 2020. *(STED)*
- Park, S., De Mello, S., Molchanov, P., Iqbal, U., Hilliges, O., Kautz, J. **Few-shot Adaptive Gaze Estimation.** ICCV 2019. *(FAZE)*
- Telea, A. **An image inpainting technique based on the fast marching method.** Journal of Graphics Tools 2004.
- Karras, T., Laine, S., Aila, T. **A Style-Based Generator Architecture for Generative Adversarial Networks.** CVPR 2019. *(FFHQ dataset)*
- Bookstein, F. L. **Principal warps: Thin-plate splines and the decomposition of deformations.** IEEE PAMI 1989. *(Thin-plate spline)*

---

*End of specification.*
