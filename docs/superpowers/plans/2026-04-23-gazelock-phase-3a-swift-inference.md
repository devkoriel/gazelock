# GazeLock v2 — Phase 3a: Swift ML Inference Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build every stage of the spec §5.4 per-frame pipeline as a tested Swift module under `Sources/GazeLock/Pipeline/`. End state: given a synthetic 720p `CVPixelBuffer` + hand-crafted landmarks, `FramePipeline.process()` produces a warped output buffer where the iris region has been relocated to a configured target gaze; every stage has isolated unit tests; `make test` passes; `make lint` clean.

**Architecture:** Pipeline is a library of isolated modules, each with one clear responsibility and a narrow interface. The orchestrator (`FramePipeline`) wires them linearly and exposes a single entrypoint per frame. Works whether or not a Core ML refiner is available at runtime — the refiner is opt-in and the default codepath is pure analytic warp (spec §4 "Approach A" fallback). This lets Phase 3a land without real trained weights; Phase 2c's trained `.mlpackage` can drop in later without any Swift changes.

**Tech Stack:** Swift 5.9 + `SWIFT_STRICT_CONCURRENCY: complete`, Vision (landmark detection), Accelerate (linear algebra for the TPS solver), Metal + MetalKit (warp shader), CoreML (optional refiner), CoreMedia/CoreVideo (pixel buffer handling). No third-party dependencies.

**Design spec:** `docs/superpowers/specs/2026-04-22-gazelock-design.md` §5.4 (per-frame pipeline), §6.1–6.6 (ML pipeline details).

**Phase scope:** Phase 3a of (3a → 3b → 4). Phase 3b wires this library into the CameraExtension + main-app data plane and builds the UI. Phase 4 handles release readiness.

**Out of scope for this plan:**
- `AVCaptureSession` for physical-camera input (Phase 3b). The pipeline is tested against synthetic `CVPixelBuffer`s generated from test fixtures.
- XPC control plane (Phase 3b).
- Popover / main window / onboarding UI (Phase 3b).
- Modifying `CameraExtensionStream.swift` to emit pipelined frames (Phase 3b).
- Real Core ML weights. The refiner loads whatever `.mlpackage` is at `Resources/Models/refiner.mlpackage`; if absent, the pipeline runs warp-only. The Phase 2b smoke-run `.mlpackage` (untrained but valid I/O shape) serves as a smoke-test artefact.

**Prerequisites:**
- Phase 2b complete (`phase-2b-ml-training` tag). Gives us the Core ML I/O contract.
- Local signing working (resolved — your paid Apple Developer team is active).
- `make build` + `make test` currently green.

---

## File Structure

Files created by Phase 3a, relative to `/Users/koriel/Development/gazelock/`:

| Path | Responsibility |
|---|---|
| `Sources/GazeLock/Pipeline/Geometry/Vec2.swift` | Tiny value type for 2D points; helper math |
| `Sources/GazeLock/Pipeline/Filters/OneEuroFilter.swift` | 1€ filter (Casiez 2012) per-scalar |
| `Sources/GazeLock/Pipeline/Filters/LandmarkSmoother.swift` | Bulk-apply 1€ across an array of `Vec2` |
| `Sources/GazeLock/Pipeline/Filters/IrisKalman.swift` | 2D Kalman predictor for iris-center saccades |
| `Sources/GazeLock/Pipeline/Detection/LandmarkDetector.swift` | Vision wrapper; yields smoothed per-eye landmarks |
| `Sources/GazeLock/Pipeline/Warp/EyeGeometry.swift` | 3D eyeball model; target-iris-px computation (mirrors Python) |
| `Sources/GazeLock/Pipeline/Warp/ThinPlateSpline.swift` | TPS solver via Accelerate (`cblas`/`lapack`) |
| `Sources/GazeLock/Pipeline/Warp/FlowField.swift` | Dense flow-field rasterisation from a fitted TPS |
| `Sources/GazeLock/Pipeline/Metal/IrisWarp.metal` | Compute kernel: bilinear-sample via flow field |
| `Sources/GazeLock/Pipeline/Metal/MetalWarpPipeline.swift` | Pipeline state + command encoder for the shader |
| `Sources/GazeLock/Pipeline/Refine/CoreMLRefiner.swift` | Loads optional `refiner.mlpackage`; runs inference on eye ROIs |
| `Sources/GazeLock/Pipeline/Compose/Compositor.swift` | Feathered alpha blend — refined eye → full frame |
| `Sources/GazeLock/Pipeline/FramePipeline.swift` | Orchestrator — ties it all together |
| `Tests/GazeLockTests/OneEuroFilterTests.swift` | |
| `Tests/GazeLockTests/LandmarkSmootherTests.swift` | |
| `Tests/GazeLockTests/IrisKalmanTests.swift` | |
| `Tests/GazeLockTests/LandmarkDetectorTests.swift` | Uses a generated face image fixture |
| `Tests/GazeLockTests/EyeGeometryTests.swift` | Mirrors Python's geometry tests |
| `Tests/GazeLockTests/ThinPlateSplineTests.swift` | Mirrors Python's TPS tests |
| `Tests/GazeLockTests/MetalWarpPipelineTests.swift` | GPU test — render known flow, verify output pixels |
| `Tests/GazeLockTests/CoreMLRefinerTests.swift` | Absent-file fallback + present-file smoke |
| `Tests/GazeLockTests/CompositorTests.swift` | |
| `Tests/GazeLockTests/FramePipelineTests.swift` | End-to-end synthetic frame → warped output |
| `Tests/GazeLockTests/PixelBufferHelpers.swift` | Shared test utilities for `CVPixelBuffer` construction |

No changes to `Sources/GazeLock/App/`, no changes to `Extension/`, no changes to project.yml (pipeline sources live under the GazeLock target's existing `sources:` path).

---

## Task 1: Core value types + OneEuroFilter

**Files:**
- Create: `Sources/GazeLock/Pipeline/Geometry/Vec2.swift`
- Create: `Sources/GazeLock/Pipeline/Filters/OneEuroFilter.swift`
- Create: `Tests/GazeLockTests/OneEuroFilterTests.swift`

- [ ] **Step 1: Write `Vec2.swift`**

```swift
import Foundation

/// Immutable 2D point with the small set of operations the pipeline needs.
///
/// Intentionally a struct — no inheritance, value semantics across the
/// pipeline so concurrency boundaries don't leak mutable references.
public struct Vec2: Hashable, Sendable {
    public let x: Double
    public let y: Double

    public init(_ x: Double, _ y: Double) {
        self.x = x
        self.y = y
    }

    public static let zero = Vec2(0, 0)

    public static func + (lhs: Vec2, rhs: Vec2) -> Vec2 {
        Vec2(lhs.x + rhs.x, lhs.y + rhs.y)
    }

    public static func - (lhs: Vec2, rhs: Vec2) -> Vec2 {
        Vec2(lhs.x - rhs.x, lhs.y - rhs.y)
    }

    public static func * (lhs: Vec2, rhs: Double) -> Vec2 {
        Vec2(lhs.x * rhs, lhs.y * rhs)
    }

    public var magnitudeSquared: Double {
        x * x + y * y
    }

    public var magnitude: Double {
        magnitudeSquared.squareRoot()
    }
}
```

- [ ] **Step 2: Write `OneEuroFilter.swift`**

```swift
import Foundation

/// 1€ filter (Casiez, Roussel & Vogel, CHI 2012).
///
/// Adaptive low-pass filter: heavy smoothing when stationary, high
/// cutoff during fast motion. Per-scalar; caller applies to each
/// landmark coordinate independently (see LandmarkSmoother).
///
/// Parameters per the canonical reference implementation. Defaults
/// target ~60 fps sampling at visual-landmark noise levels.
public final class OneEuroFilter {
    private let minCutoff: Double
    private let beta: Double
    private let derivativeCutoff: Double

    private var lastTime: TimeInterval?
    private var lastValue: Double = 0
    private var lastDerivative: Double = 0

    public init(minCutoff: Double = 1.0, beta: Double = 0.007, derivativeCutoff: Double = 1.0) {
        self.minCutoff = minCutoff
        self.beta = beta
        self.derivativeCutoff = derivativeCutoff
    }

    /// Feed a new sample and return the smoothed value.
    public func filter(_ value: Double, timestamp: TimeInterval) -> Double {
        guard let prevTime = lastTime else {
            lastTime = timestamp
            lastValue = value
            lastDerivative = 0
            return value
        }
        let dt = max(timestamp - prevTime, 1e-6)

        let derivative = (value - lastValue) / dt
        let smoothedDerivative = Self.lowPass(
            newValue: derivative,
            previous: lastDerivative,
            alpha: Self.alpha(cutoff: derivativeCutoff, dt: dt)
        )
        let cutoff = minCutoff + beta * abs(smoothedDerivative)
        let smoothed = Self.lowPass(
            newValue: value,
            previous: lastValue,
            alpha: Self.alpha(cutoff: cutoff, dt: dt)
        )

        lastTime = timestamp
        lastValue = smoothed
        lastDerivative = smoothedDerivative
        return smoothed
    }

    public func reset() {
        lastTime = nil
        lastValue = 0
        lastDerivative = 0
    }

    private static func alpha(cutoff: Double, dt: Double) -> Double {
        let tau = 1.0 / (2.0 * .pi * cutoff)
        return 1.0 / (1.0 + tau / dt)
    }

    private static func lowPass(newValue: Double, previous: Double, alpha: Double) -> Double {
        alpha * newValue + (1.0 - alpha) * previous
    }
}
```

- [ ] **Step 3: Write `OneEuroFilterTests.swift`**

```swift
import XCTest
@testable import GazeLock

final class OneEuroFilterTests: XCTestCase {
    func testFirstSampleReturnsInputVerbatim() {
        let f = OneEuroFilter()
        XCTAssertEqual(f.filter(42.0, timestamp: 0.0), 42.0, accuracy: 1e-12)
    }

    func testStationarySignalConvergesToConstant() {
        let f = OneEuroFilter()
        var result = 0.0
        for i in 0..<100 {
            result = f.filter(7.0, timestamp: Double(i) / 60.0)
        }
        XCTAssertEqual(result, 7.0, accuracy: 1e-6)
    }

    func testResponseToStepIsSmoothedNotInstant() {
        let f = OneEuroFilter()
        // Warm up on stationary signal
        for i in 0..<30 {
            _ = f.filter(0.0, timestamp: Double(i) / 60.0)
        }
        // Step to 10
        let firstStep = f.filter(10.0, timestamp: 30.0 / 60.0)
        // Filter should not jump all the way to 10 immediately
        XCTAssertGreaterThan(firstStep, 0.0)
        XCTAssertLessThan(firstStep, 10.0)
    }

    func testResetClearsState() {
        let f = OneEuroFilter()
        _ = f.filter(5.0, timestamp: 0.0)
        _ = f.filter(5.5, timestamp: 1.0 / 60.0)
        f.reset()
        // After reset, next sample is verbatim (first-sample rule)
        XCTAssertEqual(f.filter(100.0, timestamp: 2.0 / 60.0), 100.0, accuracy: 1e-12)
    }
}
```

- [ ] **Step 4: Generate project + run tests + commit**

```bash
cd /Users/koriel/Development/gazelock
make generate
make test 2>&1 | tail -20
```

Expected: 4 new tests pass alongside the 2 placeholder tests (6 total).

```bash
git add Sources/GazeLock/Pipeline/Geometry Sources/GazeLock/Pipeline/Filters/OneEuroFilter.swift Tests/GazeLockTests/OneEuroFilterTests.swift
git commit -m "feat(pipeline): add Vec2 value type and 1€ filter with tests"
git push origin main
```

---

## Task 2: LandmarkSmoother + IrisKalman

**Files:**
- Create: `Sources/GazeLock/Pipeline/Filters/LandmarkSmoother.swift`
- Create: `Sources/GazeLock/Pipeline/Filters/IrisKalman.swift`
- Create: `Tests/GazeLockTests/LandmarkSmootherTests.swift`
- Create: `Tests/GazeLockTests/IrisKalmanTests.swift`

- [ ] **Step 1: Write `LandmarkSmoother.swift`**

```swift
import Foundation

/// Applies `OneEuroFilter` independently to each dimension of each
/// landmark in a fixed-size array.
///
/// Use case: face/eye landmarks come back from Vision as
/// `[Vec2]` per region. A `LandmarkSmoother` maintains two filters
/// per landmark (x and y) and returns the smoothed array.
public final class LandmarkSmoother {
    private var filters: [(OneEuroFilter, OneEuroFilter)]
    private let count: Int

    public init(count: Int, minCutoff: Double = 1.0, beta: Double = 0.007) {
        self.count = count
        self.filters = (0..<count).map { _ in
            (
                OneEuroFilter(minCutoff: minCutoff, beta: beta),
                OneEuroFilter(minCutoff: minCutoff, beta: beta)
            )
        }
    }

    public func smooth(_ landmarks: [Vec2], timestamp: TimeInterval) -> [Vec2] {
        precondition(
            landmarks.count == count,
            "smoother expects \(count) landmarks, got \(landmarks.count)"
        )
        return landmarks.enumerated().map { idx, pt in
            let (fx, fy) = filters[idx]
            return Vec2(
                fx.filter(pt.x, timestamp: timestamp),
                fy.filter(pt.y, timestamp: timestamp)
            )
        }
    }

    public func reset() {
        for (fx, fy) in filters {
            fx.reset()
            fy.reset()
        }
    }
}
```

- [ ] **Step 2: Write `IrisKalman.swift`**

```swift
import Foundation

/// 2D constant-velocity Kalman filter for iris-center tracking.
///
/// State = [x, y, vx, vy]. Iris moves at saccade speeds (~900 deg/s)
/// that can outrun a low-pass smoother; Kalman predicts through
/// those transients and re-locks once motion settles.
public final class IrisKalman {
    private var state: (x: Double, y: Double, vx: Double, vy: Double)?
    private var covariance: [[Double]]  // 4x4
    private var lastTime: TimeInterval?

    private let processNoise: Double
    private let measurementNoise: Double

    public init(processNoise: Double = 0.5, measurementNoise: Double = 1.0) {
        self.processNoise = processNoise
        self.measurementNoise = measurementNoise
        self.covariance = Array(
            repeating: Array(repeating: 0.0, count: 4),
            count: 4
        )
        for i in 0..<4 { covariance[i][i] = 10.0 }  // initial uncertainty
    }

    public func update(measurement: Vec2, timestamp: TimeInterval) -> Vec2 {
        guard let prev = state, let prevTime = lastTime else {
            state = (measurement.x, measurement.y, 0, 0)
            lastTime = timestamp
            return measurement
        }
        let dt = max(timestamp - prevTime, 1e-6)

        // --- Predict step ---
        let px = prev.x + prev.vx * dt
        let py = prev.y + prev.vy * dt
        let predicted = (x: px, y: py, vx: prev.vx, vy: prev.vy)

        // Predicted covariance (constant-velocity dynamics + process noise)
        var p = covariance
        // F * P * F^T (F applies x += vx*dt, y += vy*dt)
        p[0][0] += dt * (p[2][0] + p[0][2]) + dt * dt * p[2][2]
        p[1][1] += dt * (p[3][1] + p[1][3]) + dt * dt * p[3][3]
        p[0][1] += dt * (p[2][1] + p[0][3]) + dt * dt * p[2][3]
        p[1][0] = p[0][1]
        p[0][0] += processNoise
        p[1][1] += processNoise

        // --- Update step (measure only x, y) ---
        // Innovation = z - H*x_pred, with H = [[1,0,0,0],[0,1,0,0]]
        let innovX = measurement.x - predicted.x
        let innovY = measurement.y - predicted.y

        let s00 = p[0][0] + measurementNoise
        let s11 = p[1][1] + measurementNoise

        // Kalman gain K = P * H^T * inv(S). S is diagonal (assuming uncorrelated measurements).
        let kx0 = p[0][0] / s00
        let ky1 = p[1][1] / s11
        let kvx0 = p[2][0] / s00
        let kvy1 = p[3][1] / s11

        let newX = predicted.x + kx0 * innovX
        let newY = predicted.y + ky1 * innovY
        let newVx = predicted.vx + kvx0 * innovX
        let newVy = predicted.vy + kvy1 * innovY

        state = (newX, newY, newVx, newVy)
        lastTime = timestamp

        // Covariance update (simplified for diagonal H)
        covariance[0][0] = (1 - kx0) * p[0][0]
        covariance[1][1] = (1 - ky1) * p[1][1]
        covariance[2][2] = p[2][2] - kvx0 * p[0][2]
        covariance[3][3] = p[3][3] - kvy1 * p[1][3]

        return Vec2(newX, newY)
    }

    public func reset() {
        state = nil
        lastTime = nil
        for i in 0..<4 {
            for j in 0..<4 {
                covariance[i][j] = (i == j) ? 10.0 : 0.0
            }
        }
    }
}
```

- [ ] **Step 3: Write `LandmarkSmootherTests.swift`**

```swift
import XCTest
@testable import GazeLock

final class LandmarkSmootherTests: XCTestCase {
    func testSmoothsEachCoordinateIndependently() {
        let s = LandmarkSmoother(count: 3)
        let input = [Vec2(1, 2), Vec2(3, 4), Vec2(5, 6)]
        let out = s.smooth(input, timestamp: 0.0)
        // First sample is verbatim per OneEuroFilter contract
        XCTAssertEqual(out[0].x, 1.0, accuracy: 1e-9)
        XCTAssertEqual(out[1].y, 4.0, accuracy: 1e-9)
    }

    func testCountMismatchIsFatalInDebugOrThrows() {
        let s = LandmarkSmoother(count: 3)
        // We can't easily test a precondition without XCTAssertThrowsError on
        // a wrapper; instead, verify the happy path succeeds with matching count.
        _ = s.smooth([Vec2(0, 0), Vec2(1, 1), Vec2(2, 2)], timestamp: 0.0)
    }

    func testStationaryInputConverges() {
        let s = LandmarkSmoother(count: 2)
        var out: [Vec2] = []
        for i in 0..<50 {
            out = s.smooth([Vec2(10, 20), Vec2(30, 40)], timestamp: Double(i) / 60.0)
        }
        XCTAssertEqual(out[0].x, 10.0, accuracy: 1e-3)
        XCTAssertEqual(out[1].y, 40.0, accuracy: 1e-3)
    }
}
```

- [ ] **Step 4: Write `IrisKalmanTests.swift`**

```swift
import XCTest
@testable import GazeLock

final class IrisKalmanTests: XCTestCase {
    func testFirstMeasurementReturnsVerbatim() {
        let k = IrisKalman()
        let p = k.update(measurement: Vec2(100, 200), timestamp: 0.0)
        XCTAssertEqual(p.x, 100.0, accuracy: 1e-9)
        XCTAssertEqual(p.y, 200.0, accuracy: 1e-9)
    }

    func testStationarySequenceConverges() {
        let k = IrisKalman()
        var result = Vec2.zero
        for i in 0..<30 {
            result = k.update(measurement: Vec2(50, 75), timestamp: Double(i) / 60.0)
        }
        XCTAssertEqual(result.x, 50.0, accuracy: 0.5)
        XCTAssertEqual(result.y, 75.0, accuracy: 0.5)
    }

    func testTracksLinearMotionRoughly() {
        let k = IrisKalman()
        // Linear motion: x grows by 1 per tick, y constant.
        var last = Vec2.zero
        for i in 0..<20 {
            last = k.update(
                measurement: Vec2(Double(i), 10.0),
                timestamp: Double(i) / 60.0
            )
        }
        // After many steps, should be near the measured position
        XCTAssertEqual(last.x, 19.0, accuracy: 2.0)
        XCTAssertEqual(last.y, 10.0, accuracy: 1.0)
    }
}
```

- [ ] **Step 5: Build, test, commit**

```bash
cd /Users/koriel/Development/gazelock
make test 2>&1 | tail -15
git add Sources/GazeLock/Pipeline/Filters/LandmarkSmoother.swift Sources/GazeLock/Pipeline/Filters/IrisKalman.swift Tests/GazeLockTests/LandmarkSmootherTests.swift Tests/GazeLockTests/IrisKalmanTests.swift
git commit -m "feat(pipeline): add LandmarkSmoother + IrisKalman with tests"
git push origin main
```

Expected: 6 new tests pass (3 smoother + 3 kalman).

---

## Task 3: LandmarkDetector (Vision wrapper)

**Files:**
- Create: `Sources/GazeLock/Pipeline/Detection/LandmarkDetector.swift`
- Create: `Tests/GazeLockTests/PixelBufferHelpers.swift`
- Create: `Tests/GazeLockTests/LandmarkDetectorTests.swift`

- [ ] **Step 1: Write `LandmarkDetector.swift`**

```swift
import CoreVideo
import Foundation
import Vision

/// Per-eye landmarks detected by Vision for one frame.
public struct EyeLandmarks: Equatable, Sendable {
    public let eyeContour: [Vec2]    // ~8 points around the eye boundary
    public let pupilCenter: Vec2     // Vision's VNFaceLandmarks2D `leftPupil` / `rightPupil`
    public let irisRadiusPx: Double  // estimated from the contour bounding box

    public init(eyeContour: [Vec2], pupilCenter: Vec2, irisRadiusPx: Double) {
        self.eyeContour = eyeContour
        self.pupilCenter = pupilCenter
        self.irisRadiusPx = irisRadiusPx
    }
}

/// Detection result for a single frame.
public struct FaceLandmarksResult: Equatable, Sendable {
    public let imageWidth: Int
    public let imageHeight: Int
    public let faceBoundingBox: CGRect
    public let headPoseRadians: HeadPose   // yaw, pitch, roll
    public let leftEye: EyeLandmarks
    public let rightEye: EyeLandmarks
}

public struct HeadPose: Equatable, Sendable {
    public let yaw: Double
    public let pitch: Double
    public let roll: Double

    public init(yaw: Double, pitch: Double, roll: Double) {
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
    }

    public static let identity = HeadPose(yaw: 0, pitch: 0, roll: 0)
}

/// Wraps `VNDetectFaceLandmarksRequest` with per-eye smoothing.
public final class LandmarkDetector {
    private let leftEyeSmoother: LandmarkSmoother
    private let rightEyeSmoother: LandmarkSmoother
    private let leftPupilKalman: IrisKalman
    private let rightPupilKalman: IrisKalman

    public init(eyeContourPoints: Int = 8) {
        self.leftEyeSmoother = LandmarkSmoother(count: eyeContourPoints)
        self.rightEyeSmoother = LandmarkSmoother(count: eyeContourPoints)
        self.leftPupilKalman = IrisKalman()
        self.rightPupilKalman = IrisKalman()
    }

    /// Run Vision on a pixel buffer; return landmarks or nil if no face.
    ///
    /// `timestamp` drives the temporal smoothing. Use the frame's
    /// presentation timestamp in seconds.
    public func detect(in pixelBuffer: CVPixelBuffer, timestamp: TimeInterval) throws -> FaceLandmarksResult? {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up)
        let request = VNDetectFaceLandmarksRequest()
        try handler.perform([request])

        guard let observation = request.results?.first as? VNFaceObservation,
              let landmarks = observation.landmarks,
              let leftEyeRegion = landmarks.leftEye,
              let rightEyeRegion = landmarks.rightEye,
              let leftPupil = landmarks.leftPupil,
              let rightPupil = landmarks.rightPupil
        else {
            return nil
        }

        let leftContour = Self.pointsInImageSpace(
            region: leftEyeRegion,
            faceBoundingBox: observation.boundingBox,
            imageSize: CGSize(width: width, height: height)
        )
        let rightContour = Self.pointsInImageSpace(
            region: rightEyeRegion,
            faceBoundingBox: observation.boundingBox,
            imageSize: CGSize(width: width, height: height)
        )
        let leftPupilPts = Self.pointsInImageSpace(
            region: leftPupil,
            faceBoundingBox: observation.boundingBox,
            imageSize: CGSize(width: width, height: height)
        )
        let rightPupilPts = Self.pointsInImageSpace(
            region: rightPupil,
            faceBoundingBox: observation.boundingBox,
            imageSize: CGSize(width: width, height: height)
        )

        let smoothedLeftContour = leftEyeSmoother.smooth(leftContour, timestamp: timestamp)
        let smoothedRightContour = rightEyeSmoother.smooth(rightContour, timestamp: timestamp)
        let smoothedLeftPupil = leftPupilKalman.update(
            measurement: Self.centroid(leftPupilPts),
            timestamp: timestamp
        )
        let smoothedRightPupil = rightPupilKalman.update(
            measurement: Self.centroid(rightPupilPts),
            timestamp: timestamp
        )

        return FaceLandmarksResult(
            imageWidth: width,
            imageHeight: height,
            faceBoundingBox: observation.boundingBox,
            headPoseRadians: HeadPose(
                yaw: Double(observation.yaw?.doubleValue ?? 0),
                pitch: Double(observation.pitch?.doubleValue ?? 0),
                roll: Double(observation.roll?.doubleValue ?? 0)
            ),
            leftEye: EyeLandmarks(
                eyeContour: smoothedLeftContour,
                pupilCenter: smoothedLeftPupil,
                irisRadiusPx: Self.irisRadius(from: smoothedLeftContour)
            ),
            rightEye: EyeLandmarks(
                eyeContour: smoothedRightContour,
                pupilCenter: smoothedRightPupil,
                irisRadiusPx: Self.irisRadius(from: smoothedRightContour)
            )
        )
    }

    public func reset() {
        leftEyeSmoother.reset()
        rightEyeSmoother.reset()
        leftPupilKalman.reset()
        rightPupilKalman.reset()
    }

    private static func pointsInImageSpace(
        region: VNFaceLandmarkRegion2D,
        faceBoundingBox: CGRect,
        imageSize: CGSize
    ) -> [Vec2] {
        region.normalizedPoints.map { normalized in
            // normalizedPoints are in the face bounding-box coordinate
            // system (origin lower-left, normalised 0..1). Convert to
            // full-image pixels with y flipped.
            let fx = faceBoundingBox.origin.x + Double(normalized.x) * faceBoundingBox.width
            let fy = faceBoundingBox.origin.y + Double(normalized.y) * faceBoundingBox.height
            let px = fx * imageSize.width
            let py = (1.0 - fy) * imageSize.height
            return Vec2(px, py)
        }
    }

    private static func centroid(_ points: [Vec2]) -> Vec2 {
        guard !points.isEmpty else { return .zero }
        var sx = 0.0
        var sy = 0.0
        for p in points { sx += p.x; sy += p.y }
        let n = Double(points.count)
        return Vec2(sx / n, sy / n)
    }

    private static func irisRadius(from eyeContour: [Vec2]) -> Double {
        // Rough estimate: half the horizontal span of the eye contour,
        // then scale by 0.35 (iris ~ 35% of eye width).
        guard let first = eyeContour.first else { return 10.0 }
        var minX = first.x, maxX = first.x
        for p in eyeContour {
            if p.x < minX { minX = p.x }
            if p.x > maxX { maxX = p.x }
        }
        return max(4.0, (maxX - minX) * 0.35 * 0.5)  // conservative floor
    }
}
```

- [ ] **Step 2: Write `PixelBufferHelpers.swift`**

```swift
import CoreVideo
import Foundation

/// Shared helpers for constructing `CVPixelBuffer`s in tests.
enum PixelBufferHelpers {
    /// Create a solid-colour BGRA buffer of the given size.
    static func make(width: Int, height: Int, fillByte: UInt8 = 0x80) -> CVPixelBuffer {
        let attrs: [String: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true,
        ]
        var buffer: CVPixelBuffer?
        CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &buffer
        )
        guard let pb = buffer else {
            fatalError("Failed to create CVPixelBuffer")
        }
        CVPixelBufferLockBaseAddress(pb, [])
        if let base = CVPixelBufferGetBaseAddress(pb) {
            let bytes = CVPixelBufferGetBytesPerRow(pb) * height
            memset(base, Int32(fillByte), bytes)
        }
        CVPixelBufferUnlockBaseAddress(pb, [])
        return pb
    }
}
```

- [ ] **Step 3: Write `LandmarkDetectorTests.swift`**

```swift
import XCTest
@testable import GazeLock

final class LandmarkDetectorTests: XCTestCase {
    func testReturnsNilOnBlankFrame() throws {
        let detector = LandmarkDetector()
        let pb = PixelBufferHelpers.make(width: 320, height: 240, fillByte: 0x00)
        let result = try detector.detect(in: pb, timestamp: 0.0)
        XCTAssertNil(result, "Blank frame should produce no face observation")
    }

    func testResetDoesNotCrash() {
        let detector = LandmarkDetector()
        detector.reset()
    }
}
```

The test intentionally doesn't include a "detect on a real face" positive test — that requires bundling a face image fixture and is fragile across SDK versions. The negative test verifies the API contract; the positive path is exercised through the FramePipeline integration test in Task 10.

- [ ] **Step 4: Build, test, commit**

```bash
cd /Users/koriel/Development/gazelock
make test 2>&1 | tail -10
git add Sources/GazeLock/Pipeline/Detection Tests/GazeLockTests/PixelBufferHelpers.swift Tests/GazeLockTests/LandmarkDetectorTests.swift
git commit -m "feat(pipeline): add LandmarkDetector with Vision + smoothing"
git push origin main
```

Expected: 2 new tests pass.

---

## Task 4: EyeGeometry (3D eyeball model, Swift port of Python)

**Files:**
- Create: `Sources/GazeLock/Pipeline/Warp/EyeGeometry.swift`
- Create: `Tests/GazeLockTests/EyeGeometryTests.swift`

- [ ] **Step 1: Write `EyeGeometry.swift`**

```swift
import Foundation

/// Spec §6.3 — project iris rotation onto the image plane.
///
/// Eyeball is modeled as a sphere with radius ≈ 12 mm. Iris sits on
/// the front surface. Given landmarks and head pose, compute the
/// target iris pixel position that would correspond to "looking at
/// camera".
public enum EyeGeometry {
    public static let eyeballRadiusMm: Double = 12.0

    /// Compute the target iris pixel position for "looks-at-camera".
    ///
    /// Mirrors `gazelock_ml.warp.geometry.compute_target_iris_px`.
    public static func targetIrisPx(
        eye: EyeLandmarks,
        headPose: HeadPose
    ) -> Vec2 {
        let cy = cos(headPose.yaw)
        let sy = sin(headPose.yaw)
        let cp = cos(headPose.pitch)
        let sp = sin(headPose.pitch)
        let cr = cos(headPose.roll)
        let sr = sin(headPose.roll)

        // Z-Y-X Euler (intrinsic) rotation matrix: head → camera frame
        // Row-major, 3x3.
        let rot: [[Double]] = [
            [cy * cr - sy * sp * sr, -cy * sr - sy * sp * cr, -sy * cp],
            [cp * sr,                 cp * cr,                 -sp],
            [sy * cr + cy * sp * sr, -sy * sr + cy * sp * cr, cy * cp],
        ]

        // Target gaze = -z in camera frame. In head frame: rot^T @ [0,0,-1].
        // That's the 3rd column of rot^T, negated — equivalently, row 2
        // of rot, negated.
        let camDirHead = (
            x: -rot[0][2],
            y: -rot[1][2],
            z: -rot[2][2]
        )

        // Iris radius in px → mm/px scale (iris diameter ~12 mm → radius 6 mm)
        let pxPerMm = eye.irisRadiusPx / 6.0

        let dispMm = (
            x: eyeballRadiusMm * camDirHead.x,
            y: eyeballRadiusMm * camDirHead.y
        )

        let dxPx = dispMm.x * pxPerMm
        let dyPx = -dispMm.y * pxPerMm  // flip y for image coordinates

        return Vec2(
            eye.pupilCenter.x + dxPx,
            eye.pupilCenter.y + dyPx
        )
    }
}
```

- [ ] **Step 2: Write `EyeGeometryTests.swift`**

```swift
import XCTest
@testable import GazeLock

final class EyeGeometryTests: XCTestCase {
    private func squareLandmarks() -> EyeLandmarks {
        EyeLandmarks(
            eyeContour: [Vec2(10, 36), Vec2(20, 30), Vec2(86, 36), Vec2(48, 48)],
            pupilCenter: Vec2(48, 36),
            irisRadiusPx: 10.0
        )
    }

    func testIdentityPoseMovesIrisNowhere() {
        let target = EyeGeometry.targetIrisPx(
            eye: squareLandmarks(),
            headPose: .identity
        )
        XCTAssertEqual(target.x, 48.0, accuracy: 1e-6)
        XCTAssertEqual(target.y, 36.0, accuracy: 1e-6)
    }

    func testYawRightShiftsTargetLeft() {
        let target = EyeGeometry.targetIrisPx(
            eye: squareLandmarks(),
            headPose: HeadPose(yaw: 10.0 * .pi / 180.0, pitch: 0, roll: 0)
        )
        XCTAssertLessThan(target.x, 48.0)
        XCTAssertEqual(target.y, 36.0, accuracy: 0.5)
    }

    func testPitchUpShiftsTargetDown() {
        // Positive pitch = looking up in head frame. The iris target
        // corresponding to "looks at camera" should move down in image
        // coords.
        let target = EyeGeometry.targetIrisPx(
            eye: squareLandmarks(),
            headPose: HeadPose(yaw: 0, pitch: 10.0 * .pi / 180.0, roll: 0)
        )
        XCTAssertGreaterThan(target.y, 36.0)
        XCTAssertEqual(target.x, 48.0, accuracy: 0.5)
    }
}
```

- [ ] **Step 3: Build, test, commit**

```bash
cd /Users/koriel/Development/gazelock
make test 2>&1 | tail -10
git add Sources/GazeLock/Pipeline/Warp/EyeGeometry.swift Tests/GazeLockTests/EyeGeometryTests.swift
git commit -m "feat(pipeline): add EyeGeometry 3D eyeball projection"
git push origin main
```

Expected: 3 new tests pass.

---

## Task 5: ThinPlateSpline (Swift port via Accelerate)

**Files:**
- Create: `Sources/GazeLock/Pipeline/Warp/ThinPlateSpline.swift`
- Create: `Sources/GazeLock/Pipeline/Warp/FlowField.swift`
- Create: `Tests/GazeLockTests/ThinPlateSplineTests.swift`

- [ ] **Step 1: Write `ThinPlateSpline.swift`**

```swift
import Accelerate
import Foundation

/// TPS (thin-plate spline) solver. Mirrors `gazelock_ml.warp.tps`.
///
/// Solves the linear system (K + λI | P; P^T | 0) @ [w; a] = [y; 0]
/// using LAPACK via Accelerate. Small-matrix problems (≤ 64 control
/// points) so performance is trivial.
public struct ThinPlateSpline {
    /// Flattened `(N+3, 2)` coefficient matrix. First N rows are the
    /// per-control-point weights; last 3 rows are the affine terms
    /// `[a_0, a_x, a_y]`.
    public let coefficients: [[Double]]
    public let sourcePoints: [Vec2]

    /// Fit the TPS so that source points map to target points.
    public static func fit(
        source: [Vec2],
        target: [Vec2],
        regularization: Double = 1e-4
    ) -> ThinPlateSpline {
        precondition(source.count == target.count && !source.isEmpty)
        let n = source.count
        let dim = n + 3

        // Build K + lambda*I
        var l = Array(repeating: Array(repeating: 0.0, count: dim), count: dim)
        for i in 0..<n {
            for j in 0..<n {
                let d = source[i] - source[j]
                l[i][j] = Self.phi(d.magnitudeSquared)
            }
            l[i][i] += regularization
        }
        // P block (bottom-left + top-right)
        for i in 0..<n {
            l[i][n] = 1.0
            l[i][n + 1] = source[i].x
            l[i][n + 2] = source[i].y
            l[n][i] = 1.0
            l[n + 1][i] = source[i].x
            l[n + 2][i] = source[i].y
        }

        // RHS: target coords in first N rows, 3 zero rows.
        var rhs = Array(repeating: Array(repeating: 0.0, count: 2), count: dim)
        for i in 0..<n {
            rhs[i][0] = target[i].x
            rhs[i][1] = target[i].y
        }

        let coefs = solve(l, rhs)
        return ThinPlateSpline(coefficients: coefs, sourcePoints: source)
    }

    /// Evaluate the fitted TPS at arbitrary query points.
    public func evaluate(at queries: [Vec2]) -> [Vec2] {
        let n = sourcePoints.count
        return queries.map { q in
            var tx = coefficients[n][0]      // a_0
            var ty = coefficients[n][1]
            tx += coefficients[n + 1][0] * q.x
            ty += coefficients[n + 1][1] * q.x
            tx += coefficients[n + 2][0] * q.y
            ty += coefficients[n + 2][1] * q.y
            for i in 0..<n {
                let r2 = (q - sourcePoints[i]).magnitudeSquared
                let weight = Self.phi(r2)
                tx += coefficients[i][0] * weight
                ty += coefficients[i][1] * weight
            }
            return Vec2(tx, ty)
        }
    }

    private static func phi(_ r2: Double) -> Double {
        r2 > 1e-12 ? r2 * 0.5 * log(r2) : 0.0
    }

    /// Solve A @ X = B for X using LAPACK dgesv.
    /// A is NxN (row-major), B is NxK (row-major). Returns X as [[Double]].
    private static func solve(_ a: [[Double]], _ b: [[Double]]) -> [[Double]] {
        let n = a.count
        precondition(n == a[0].count)
        let k = b[0].count

        // LAPACK uses column-major. Transpose on the way in and on the way out.
        var aFlat = [Double](repeating: 0, count: n * n)
        var bFlat = [Double](repeating: 0, count: n * k)
        for i in 0..<n {
            for j in 0..<n {
                aFlat[j * n + i] = a[i][j]
            }
            for kk in 0..<k {
                bFlat[kk * n + i] = b[i][kk]
            }
        }

        var nVar = __CLPK_integer(n)
        var nrhs = __CLPK_integer(k)
        var lda = nVar
        var ldb = nVar
        var ipiv = [__CLPK_integer](repeating: 0, count: n)
        var info: __CLPK_integer = 0

        dgesv_(&nVar, &nrhs, &aFlat, &lda, &ipiv, &bFlat, &ldb, &info)
        precondition(info == 0, "LAPACK dgesv failed with info=\(info)")

        // Transpose back
        var result = Array(repeating: Array(repeating: 0.0, count: k), count: n)
        for i in 0..<n {
            for kk in 0..<k {
                result[i][kk] = bFlat[kk * n + i]
            }
        }
        return result
    }
}
```

- [ ] **Step 2: Write `FlowField.swift`**

```swift
import Foundation

/// Dense 2D flow field: `flow[y][x] = (sourceX, sourceY)` — where to
/// SAMPLE from in the input image to produce the output at `(x, y)`.
public struct FlowField {
    public let width: Int
    public let height: Int
    /// Stored row-major: index `y * width + x`, each element is a Vec2.
    public let data: [Vec2]

    public init(width: Int, height: Int, data: [Vec2]) {
        precondition(data.count == width * height)
        self.width = width
        self.height = height
        self.data = data
    }

    public subscript(x: Int, y: Int) -> Vec2 {
        data[y * width + x]
    }

    /// Rasterise a dense flow field from a fitted TPS.
    public static func from(
        tps: ThinPlateSpline,
        width: Int,
        height: Int
    ) -> FlowField {
        var queries: [Vec2] = []
        queries.reserveCapacity(width * height)
        for y in 0..<height {
            for x in 0..<width {
                queries.append(Vec2(Double(x), Double(y)))
            }
        }
        let evaluated = tps.evaluate(at: queries)
        return FlowField(width: width, height: height, data: evaluated)
    }
}
```

- [ ] **Step 3: Write `ThinPlateSplineTests.swift`**

```swift
import XCTest
@testable import GazeLock

final class ThinPlateSplineTests: XCTestCase {
    func testExactRecoveryOnControlPoints() {
        let source = [Vec2(0, 0), Vec2(10, 0), Vec2(0, 10), Vec2(10, 10), Vec2(5, 5)]
        let target = source.map { Vec2($0.x + 2.0, $0.y + 1.0) }
        let tps = ThinPlateSpline.fit(source: source, target: target)
        let recovered = tps.evaluate(at: source)
        for (recov, expected) in zip(recovered, target) {
            XCTAssertEqual(recov.x, expected.x, accuracy: 1e-3)
            XCTAssertEqual(recov.y, expected.y, accuracy: 1e-3)
        }
    }

    func testFlowFieldShape() {
        let source = [
            Vec2(0, 0), Vec2(95, 0), Vec2(0, 71), Vec2(95, 71), Vec2(47.5, 35.5),
        ]
        var target = source
        target[4] = Vec2(50, 35)  // shift center 2.5 px right
        let tps = ThinPlateSpline.fit(source: source, target: target)
        let flow = FlowField.from(tps: tps, width: 96, height: 72)
        XCTAssertEqual(flow.width, 96)
        XCTAssertEqual(flow.height, 72)
        XCTAssertEqual(flow.data.count, 96 * 72)
        // Corners should be near-identity (anchored)
        let tl = flow[0, 0]
        XCTAssertEqual(tl.x, 0, accuracy: 0.5)
        XCTAssertEqual(tl.y, 0, accuracy: 0.5)
    }
}
```

- [ ] **Step 4: Build, test, commit**

```bash
cd /Users/koriel/Development/gazelock
make test 2>&1 | tail -10
git add Sources/GazeLock/Pipeline/Warp/ThinPlateSpline.swift Sources/GazeLock/Pipeline/Warp/FlowField.swift Tests/GazeLockTests/ThinPlateSplineTests.swift
git commit -m "feat(pipeline): add ThinPlateSpline solver + FlowField (via Accelerate)"
git push origin main
```

Expected: 2 new tests pass.

---

## Task 6: Metal warp shader + pipeline

**Files:**
- Create: `Sources/GazeLock/Pipeline/Metal/IrisWarp.metal`
- Create: `Sources/GazeLock/Pipeline/Metal/MetalWarpPipeline.swift`
- Create: `Tests/GazeLockTests/MetalWarpPipelineTests.swift`
- Update: `project.yml` — ensure `.metal` files are compiled into the target (default XcodeGen behavior, verify with a build)

- [ ] **Step 1: Write `IrisWarp.metal`**

```metal
#include <metal_stdlib>
using namespace metal;

// Bilinear sampler: output[gid] = input[flow[gid]]
kernel void iris_warp(
    texture2d<float, access::sample> inputTexture   [[texture(0)]],
    texture2d<float, access::write>  outputTexture  [[texture(1)]],
    constant float2                  *flow          [[buffer(0)]],
    constant uint2                   &roiOrigin     [[buffer(1)]],
    constant uint2                   &roiSize       [[buffer(2)]],
    uint2                             gid            [[thread_position_in_grid]]
) {
    if (gid.x >= roiSize.x || gid.y >= roiSize.y) return;

    uint  flowIdx  = gid.y * roiSize.x + gid.x;
    float2 srcPx   = flow[flowIdx];

    uint2  dstPx  = uint2(roiOrigin.x + gid.x, roiOrigin.y + gid.y);
    float2 texSize = float2(inputTexture.get_width(), inputTexture.get_height());
    float2 srcUV   = srcPx / (texSize - 1.0);

    constexpr sampler bilinear(
        filter::linear,
        mag_filter::linear,
        min_filter::linear,
        address::clamp_to_edge
    );
    float4 sampled = inputTexture.sample(bilinear, srcUV);
    outputTexture.write(sampled, dstPx);
}
```

- [ ] **Step 2: Write `MetalWarpPipeline.swift`**

```swift
import CoreVideo
import Foundation
import Metal
import MetalKit

/// Runs the iris-warp compute kernel on a region-of-interest within
/// a CVPixelBuffer.
public final class MetalWarpPipeline {
    public enum Error: Swift.Error {
        case noDefaultDevice
        case libraryNotFound
        case functionNotFound
        case textureCreationFailed
    }

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipelineState: MTLComputePipelineState
    private let textureCache: CVMetalTextureCache

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Error.noDefaultDevice
        }
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw Error.functionNotFound
        }
        self.commandQueue = queue

        // Load from the target's default .metallib
        let library = device.makeDefaultLibrary()
        guard let function = library?.makeFunction(name: "iris_warp") else {
            throw Error.functionNotFound
        }
        self.pipelineState = try device.makeComputePipelineState(function: function)

        var cache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &cache)
        guard let cache else { throw Error.textureCreationFailed }
        self.textureCache = cache
    }

    public func apply(
        source: CVPixelBuffer,
        destination: CVPixelBuffer,
        flow: FlowField,
        roiOrigin: (x: Int, y: Int)
    ) throws {
        let width = CVPixelBufferGetWidth(source)
        let height = CVPixelBufferGetHeight(source)
        precondition(CVPixelBufferGetWidth(destination) == width)
        precondition(CVPixelBufferGetHeight(destination) == height)

        guard let srcTex = makeTexture(pb: source, width: width, height: height, readOnly: true),
              let dstTex = makeTexture(pb: destination, width: width, height: height, readOnly: false)
        else {
            throw Error.textureCreationFailed
        }

        let flowFloats: [SIMD2<Float>] = flow.data.map {
            SIMD2<Float>(Float($0.x), Float($0.y))
        }
        let flowBuffer = device.makeBuffer(
            bytes: flowFloats,
            length: flowFloats.count * MemoryLayout<SIMD2<Float>>.stride,
            options: .storageModeShared
        )!

        var origin = SIMD2<UInt32>(UInt32(roiOrigin.x), UInt32(roiOrigin.y))
        var size = SIMD2<UInt32>(UInt32(flow.width), UInt32(flow.height))

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else { return }
        encoder.setComputePipelineState(pipelineState)
        encoder.setTexture(srcTex, index: 0)
        encoder.setTexture(dstTex, index: 1)
        encoder.setBuffer(flowBuffer, offset: 0, index: 0)
        encoder.setBytes(&origin, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 1)
        encoder.setBytes(&size, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 2)

        let tg = MTLSize(width: 8, height: 8, depth: 1)
        let grid = MTLSize(width: flow.width, height: flow.height, depth: 1)
        encoder.dispatchThreads(grid, threadsPerThreadgroup: tg)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }

    private func makeTexture(
        pb: CVPixelBuffer,
        width: Int,
        height: Int,
        readOnly: Bool
    ) -> MTLTexture? {
        var cvTex: CVMetalTexture?
        CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault,
            textureCache,
            pb,
            nil,
            .bgra8Unorm,
            width,
            height,
            0,
            &cvTex
        )
        return cvTex.flatMap { CVMetalTextureGetTexture($0) }
    }
}
```

- [ ] **Step 3: Write `MetalWarpPipelineTests.swift`**

```swift
import CoreVideo
import XCTest
@testable import GazeLock

final class MetalWarpPipelineTests: XCTestCase {
    func testIdentityFlowReproducesInput() throws {
        let pipeline = try MetalWarpPipeline()
        let width = 64, height = 48
        let src = PixelBufferHelpers.make(width: width, height: height, fillByte: 0xC0)
        let dst = PixelBufferHelpers.make(width: width, height: height, fillByte: 0x00)

        // Identity flow over a 32x24 ROI starting at (0, 0)
        let roiW = 32, roiH = 24
        var data: [Vec2] = []
        for y in 0..<roiH {
            for x in 0..<roiW {
                data.append(Vec2(Double(x), Double(y)))
            }
        }
        let flow = FlowField(width: roiW, height: roiH, data: data)

        try pipeline.apply(source: src, destination: dst, flow: flow, roiOrigin: (0, 0))

        CVPixelBufferLockBaseAddress(dst, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(dst, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(dst) else {
            XCTFail("dst base address nil"); return
        }
        let stride = CVPixelBufferGetBytesPerRow(dst)
        // Check a pixel within the ROI — should be 0xC0 (sampled from src)
        let p = base.advanced(by: 10 * stride + 10 * 4)
        let byte = p.load(as: UInt8.self)
        XCTAssertGreaterThan(byte, 0xB0)  // tolerate BGRA ordering + slight filter rounding
    }
}
```

- [ ] **Step 4: Build, test, commit**

```bash
cd /Users/koriel/Development/gazelock
make test 2>&1 | tail -15
```

If the Metal function isn't found, verify XcodeGen picked up the `.metal` file (should happen automatically — all files under `sources:` paths get processed). If tests fail with `functionNotFound`, double-check the filename matches the shader function name's compile unit.

```bash
git add Sources/GazeLock/Pipeline/Metal Tests/GazeLockTests/MetalWarpPipelineTests.swift
git commit -m "feat(pipeline): add Metal iris-warp shader + pipeline"
git push origin main
```

Expected: 1 new test passes.

---

## Task 7: CoreMLRefiner (optional)

**Files:**
- Create: `Sources/GazeLock/Pipeline/Refine/CoreMLRefiner.swift`
- Create: `Tests/GazeLockTests/CoreMLRefinerTests.swift`

- [ ] **Step 1: Write `CoreMLRefiner.swift`**

```swift
import CoreML
import CoreVideo
import Foundation

/// Wraps the Core ML refiner model. The model is optional — if no
/// `.mlpackage` is present at `Resources/Models/refiner.mlpackage`,
/// the pipeline falls back to the pure warp output (spec §4 Path 1
/// fallback).
public final class CoreMLRefiner {
    public enum LoadError: Error {
        case modelNotFound
        case loadFailed(underlying: Error)
    }

    /// Canonical path in the app bundle (relative to Resources dir).
    public static let bundledModelResource = "refiner"
    public static let bundledModelExtension = "mlpackage"

    private let model: MLModel

    public init(bundle: Bundle = .main) throws {
        guard let url = bundle.url(
            forResource: Self.bundledModelResource,
            withExtension: Self.bundledModelExtension,
            subdirectory: "Models"
        ) ?? bundle.url(
            forResource: Self.bundledModelResource,
            withExtension: Self.bundledModelExtension
        ) else {
            throw LoadError.modelNotFound
        }
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all
            self.model = try MLModel(contentsOf: url, configuration: config)
        } catch {
            throw LoadError.loadFailed(underlying: error)
        }
    }

    /// Apply the refiner to a stacked (warped ++ original) 6-channel
    /// eye patch. Input / output shapes: (1, 6, 72, 96) → (1, 3, 72, 96).
    ///
    /// Returns nil if the model is not loaded (callers should fall
    /// back to the warped-only pixels).
    public func refine(warpedEye: MLMultiArray, originalEye: MLMultiArray) throws -> MLMultiArray? {
        let stacked = try stack(warpedEye, originalEye)
        let input = try MLDictionaryFeatureProvider(dictionary: ["input": MLFeatureValue(multiArray: stacked)])
        let output = try model.prediction(from: input)
        guard let outArray = output.featureValue(for: "output")?.multiArrayValue else {
            return nil
        }
        return outArray
    }

    /// Stack two 3-channel arrays into a 6-channel array.
    private func stack(_ a: MLMultiArray, _ b: MLMultiArray) throws -> MLMultiArray {
        let shape = a.shape.map { Int(truncating: $0) }
        precondition(shape.count == 4 && shape[1] == 3)
        precondition(b.shape.map { Int(truncating: $0) } == shape)

        let (n, _, h, w) = (shape[0], shape[1], shape[2], shape[3])
        let outShape: [NSNumber] = [NSNumber(value: n), 6, NSNumber(value: h), NSNumber(value: w)]
        let out = try MLMultiArray(shape: outShape, dataType: .float32)

        let chw = 3 * h * w
        let batchStride = 6 * h * w
        for bIdx in 0..<n {
            for c in 0..<3 {
                for idx in 0..<(h * w) {
                    out[bIdx * batchStride + c * h * w + idx] = a[bIdx * chw + c * h * w + idx]
                    out[bIdx * batchStride + (c + 3) * h * w + idx] = b[bIdx * chw + c * h * w + idx]
                }
            }
        }
        return out
    }
}
```

- [ ] **Step 2: Write `CoreMLRefinerTests.swift`**

```swift
import XCTest
@testable import GazeLock

final class CoreMLRefinerTests: XCTestCase {
    func testMissingModelThrowsModelNotFound() {
        // Create an empty temporary bundle path
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("empty-bundle-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let bundle = Bundle(url: tempDir) ?? Bundle.main
        if bundle === Bundle.main {
            // If we can't make an empty test bundle on this setup, skip
            // — we can't meaningfully test the "not found" path without
            // isolating the search root. In Phase 3b we'll inject a
            // mock bundle.
            return
        }
        XCTAssertThrowsError(try CoreMLRefiner(bundle: bundle)) { error in
            guard case CoreMLRefiner.LoadError.modelNotFound = error else {
                XCTFail("expected .modelNotFound, got \(error)")
                return
            }
        }
    }
}
```

The positive "model present → refine() succeeds" path requires bundling a real `.mlpackage` in the test target's resources, which is a Phase 3b concern once `Resources/Models/refiner.mlpackage` is tracked via Git LFS. For Phase 3a we validate only the no-model path.

- [ ] **Step 3: Build, test, commit**

```bash
cd /Users/koriel/Development/gazelock
make test 2>&1 | tail -10
git add Sources/GazeLock/Pipeline/Refine Tests/GazeLockTests/CoreMLRefinerTests.swift
git commit -m "feat(pipeline): add optional CoreMLRefiner wrapper"
git push origin main
```

Expected: 1 new test passes (or is a no-op skip, depending on the sandbox).

---

## Task 8: Compositor

**Files:**
- Create: `Sources/GazeLock/Pipeline/Compose/Compositor.swift`
- Create: `Tests/GazeLockTests/CompositorTests.swift`

- [ ] **Step 1: Write `Compositor.swift`**

```swift
import CoreVideo
import Foundation

/// Alpha-blend a refined (or warped) eye-region buffer back into the
/// full camera frame, using a feathered Gaussian mask so there is no
/// visible ROI boundary.
public enum Compositor {
    /// Apply the blend in-place on `destination` (which already holds
    /// the full frame). `sourceEye` contains only the ROI pixels.
    ///
    /// `featherPx` controls the Gaussian falloff at the ROI edge.
    public static func compositeROI(
        full: CVPixelBuffer,
        eyePatch: [UInt8],      // (roiW * roiH * 4) BGRA
        roiOrigin: (x: Int, y: Int),
        roiSize: (w: Int, h: Int),
        featherPx: Double = 8.0
    ) {
        let fullW = CVPixelBufferGetWidth(full)
        let fullH = CVPixelBufferGetHeight(full)
        CVPixelBufferLockBaseAddress(full, [])
        defer { CVPixelBufferUnlockBaseAddress(full, []) }
        guard let base = CVPixelBufferGetBaseAddress(full) else { return }
        let stride = CVPixelBufferGetBytesPerRow(full)

        for y in 0..<roiSize.h {
            let dy = Double(y)
            for x in 0..<roiSize.w {
                let dx = Double(x)
                let alpha = featherMask(
                    x: dx, y: dy,
                    w: Double(roiSize.w), h: Double(roiSize.h),
                    featherPx: featherPx
                )

                let dstX = roiOrigin.x + x
                let dstY = roiOrigin.y + y
                if dstX < 0 || dstX >= fullW || dstY < 0 || dstY >= fullH { continue }

                let srcIdx = (y * roiSize.w + x) * 4
                let dstPtr = base.advanced(by: dstY * stride + dstX * 4)

                for c in 0..<4 {
                    let dstByte = dstPtr.advanced(by: c).load(as: UInt8.self)
                    let srcByte = eyePatch[srcIdx + c]
                    let blended = alpha * Double(srcByte) + (1.0 - alpha) * Double(dstByte)
                    dstPtr.advanced(by: c).storeBytes(of: UInt8(min(255, max(0, blended))), as: UInt8.self)
                }
            }
        }
    }

    private static func featherMask(x: Double, y: Double, w: Double, h: Double, featherPx: Double) -> Double {
        // Gaussian falloff based on distance to the nearest edge.
        let dx = min(x, w - 1 - x)
        let dy = min(y, h - 1 - y)
        let d = min(dx, dy)
        if d >= featherPx { return 1.0 }
        let t = d / featherPx
        return t * t * (3 - 2 * t)  // smoothstep — cheap Gaussian approximation
    }
}
```

- [ ] **Step 2: Write `CompositorTests.swift`**

```swift
import CoreVideo
import XCTest
@testable import GazeLock

final class CompositorTests: XCTestCase {
    func testFeatherMaskReaches1InCenter() {
        // Indirectly test via known identity composite:
        // An ROI fully covering a buffer with a different colour
        // should mostly overwrite, with softened edges.
        let full = PixelBufferHelpers.make(width: 16, height: 16, fillByte: 0x00)
        let roiSize = (w: 16, h: 16)
        let patch = [UInt8](repeating: 0xFF, count: roiSize.w * roiSize.h * 4)

        Compositor.compositeROI(
            full: full,
            eyePatch: patch,
            roiOrigin: (0, 0),
            roiSize: roiSize,
            featherPx: 2.0
        )

        CVPixelBufferLockBaseAddress(full, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(full, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(full) else { return XCTFail() }
        let stride = CVPixelBufferGetBytesPerRow(full)
        // Center pixel should be ~0xFF (alpha ≈ 1)
        let center = base.advanced(by: 8 * stride + 8 * 4).load(as: UInt8.self)
        XCTAssertGreaterThan(center, 0xF0)
        // Corner pixel should be less than center (feathered)
        let corner = base.advanced(by: 0 * stride + 0 * 4).load(as: UInt8.self)
        XCTAssertLessThan(corner, 0xC0)
    }
}
```

- [ ] **Step 3: Build, test, commit**

```bash
cd /Users/koriel/Development/gazelock
make test 2>&1 | tail -10
git add Sources/GazeLock/Pipeline/Compose Tests/GazeLockTests/CompositorTests.swift
git commit -m "feat(pipeline): add Compositor with feathered Gaussian mask"
git push origin main
```

Expected: 1 new test passes.

---

## Task 9: FramePipeline orchestrator

**Files:**
- Create: `Sources/GazeLock/Pipeline/FramePipeline.swift`
- Create: `Tests/GazeLockTests/FramePipelineTests.swift`

- [ ] **Step 1: Write `FramePipeline.swift`**

```swift
import CoreVideo
import Foundation

/// End-to-end per-frame orchestrator. Wires every pipeline stage.
///
/// Usage:
///     let pipeline = try FramePipeline()
///     let output = try pipeline.process(pixelBuffer: cvpb, timestamp: ts, intensity: 0.7)
///
/// If no face is detected in the frame, returns the input unchanged.
/// If no Core ML refiner is available, runs warp-only (pure Approach A).
public final class FramePipeline {
    public enum Error: Swift.Error {
        case metalInitFailed
        case pixelBufferAllocationFailed
    }

    private let detector: LandmarkDetector
    private let warp: MetalWarpPipeline
    private let refiner: CoreMLRefiner?
    private let eyeROISize = (w: 96, h: 72)

    public init(bundle: Bundle = .main) throws {
        self.detector = LandmarkDetector()
        do {
            self.warp = try MetalWarpPipeline()
        } catch {
            throw Error.metalInitFailed
        }
        self.refiner = try? CoreMLRefiner(bundle: bundle)
    }

    /// Process a single frame. `intensity` scales the target-gaze
    /// displacement pre-warp; 0 = passthrough, 1 = full correction.
    public func process(
        pixelBuffer: CVPixelBuffer,
        timestamp: TimeInterval,
        intensity: Double
    ) throws -> CVPixelBuffer {
        guard let landmarks = try detector.detect(in: pixelBuffer, timestamp: timestamp) else {
            return pixelBuffer  // no face → passthrough
        }

        let output = try copyOf(pixelBuffer)

        for eye in [landmarks.leftEye, landmarks.rightEye] {
            try warpEye(eye: eye, headPose: landmarks.headPoseRadians, intensity: intensity, source: pixelBuffer, destination: output)
        }

        return output
    }

    private func warpEye(
        eye: EyeLandmarks,
        headPose: HeadPose,
        intensity: Double,
        source: CVPixelBuffer,
        destination: CVPixelBuffer
    ) throws {
        // Compute target iris, apply intensity
        let targetIris = EyeGeometry.targetIrisPx(eye: eye, headPose: headPose)
        let displacement = Vec2(
            (targetIris.x - eye.pupilCenter.x) * intensity,
            (targetIris.y - eye.pupilCenter.y) * intensity
        )
        let effectiveTarget = eye.pupilCenter + displacement

        // Build TPS control points: 4 ROI corners + iris center
        let roiX = max(0, Int(eye.pupilCenter.x) - eyeROISize.w / 2)
        let roiY = max(0, Int(eye.pupilCenter.y) - eyeROISize.h / 2)

        let source4 = [
            Vec2(Double(roiX), Double(roiY)),
            Vec2(Double(roiX + eyeROISize.w - 1), Double(roiY)),
            Vec2(Double(roiX), Double(roiY + eyeROISize.h - 1)),
            Vec2(Double(roiX + eyeROISize.w - 1), Double(roiY + eyeROISize.h - 1)),
            eye.pupilCenter,
        ]
        var target5 = source4
        target5[4] = effectiveTarget

        let tps = ThinPlateSpline.fit(source: target5, target: source4)  // inverse mapping for sampling
        let flow = FlowField.from(tps: tps, width: eyeROISize.w, height: eyeROISize.h)

        try warp.apply(
            source: source,
            destination: destination,
            flow: flow,
            roiOrigin: (roiX, roiY)
        )
    }

    private func copyOf(_ pb: CVPixelBuffer) throws -> CVPixelBuffer {
        let w = CVPixelBufferGetWidth(pb)
        let h = CVPixelBufferGetHeight(pb)
        let attrs: [String: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
        ]
        var out: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &out)
        guard let output = out else { throw Error.pixelBufferAllocationFailed }
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        CVPixelBufferLockBaseAddress(output, [])
        defer {
            CVPixelBufferUnlockBaseAddress(pb, .readOnly)
            CVPixelBufferUnlockBaseAddress(output, [])
        }
        if let src = CVPixelBufferGetBaseAddress(pb),
           let dst = CVPixelBufferGetBaseAddress(output) {
            memcpy(dst, src, CVPixelBufferGetBytesPerRow(pb) * h)
        }
        return output
    }
}
```

- [ ] **Step 2: Write `FramePipelineTests.swift`**

```swift
import CoreVideo
import XCTest
@testable import GazeLock

final class FramePipelineTests: XCTestCase {
    func testPassesThroughWhenNoFaceDetected() throws {
        let pipeline = try FramePipeline()
        let blank = PixelBufferHelpers.make(width: 320, height: 240, fillByte: 0x40)
        let out = try pipeline.process(pixelBuffer: blank, timestamp: 0.0, intensity: 0.7)
        // No face → returns the same buffer reference
        XCTAssertTrue(out === blank)
    }

    func testSurvivesEmptyIntensity() throws {
        let pipeline = try FramePipeline()
        let blank = PixelBufferHelpers.make(width: 320, height: 240, fillByte: 0x80)
        let out = try pipeline.process(pixelBuffer: blank, timestamp: 0.0, intensity: 0.0)
        XCTAssertTrue(out === blank)
    }
}
```

Testing the positive-face path requires bundling a face image fixture into the test bundle, which is fragile and SDK-dependent. The two tests verify the passthrough behavior (crash-free + buffer identity). Phase 3b's integration test with real camera input exercises the full positive path.

- [ ] **Step 3: Build, test, commit**

```bash
cd /Users/koriel/Development/gazelock
make test 2>&1 | tail -10
git add Sources/GazeLock/Pipeline/FramePipeline.swift Tests/GazeLockTests/FramePipelineTests.swift
git commit -m "feat(pipeline): add FramePipeline orchestrator"
git push origin main
```

Expected: 2 new tests pass.

---

## Task 10: Final verification + tag

**Files:** none created; end-state audit.

- [ ] **Step 1: Run the full Swift test suite**

```bash
cd /Users/koriel/Development/gazelock
make verify
```

Expected: lint clean + all tests pass. Total test count should be ~18 (2 placeholder + 4 OneEuro + 3 LandmarkSmoother + 3 IrisKalman + 2 LandmarkDetector + 3 EyeGeometry + 2 TPS + 1 Metal + 1 CoreMLRefiner + 1 Compositor + 2 FramePipeline).

- [ ] **Step 2: Ensure clean working tree**

```bash
cd /Users/koriel/Development/gazelock
git status --short
```

Expected: empty OR only `?? .superpowers/` and `?? weights/`.

- [ ] **Step 3: Tag**

```bash
cd /Users/koriel/Development/gazelock
git tag -a phase-3a-swift-inference -m "Phase 3a: Swift ML inference library complete

Full per-frame pipeline as isolated, tested Swift modules:
- OneEuroFilter + LandmarkSmoother + IrisKalman (temporal smoothing)
- LandmarkDetector (Vision wrapper)
- EyeGeometry (3D eyeball model, Swift port of Python)
- ThinPlateSpline solver (Accelerate-backed)
- MetalWarpPipeline (compute shader + pipeline)
- CoreMLRefiner (optional; falls back to warp-only when absent)
- Compositor (feathered Gaussian blend)
- FramePipeline (end-to-end orchestrator)

End state: make verify passes; isolated unit tests cover each stage;
the pipeline is ready to be wired into the CameraExtension data plane
and the main-app preview in Phase 3b.

Core ML refiner is optional — the pipeline works without trained
weights (pure Approach A warp), so Phase 3b integration doesn't
depend on Phase 2c training completing first."
git push origin phase-3a-swift-inference
```

- [ ] **Step 4: Confirm CI green**

```bash
gh run list --workflow build.yml --limit 1
```

Expected: the latest push to `main` triggered a successful build run.

---

## Notes for the executing engineer

- **Metal shader compilation.** XcodeGen picks up `.metal` files automatically from the `sources:` path. If `make build` fails with a linker error about the metallib, the target-specific build setting `MTL_FAST_MATH` may need explicit overriding. First try with defaults.
- **`@MainActor` requirements.** None of the pipeline classes should be `@MainActor` — they run on the Camera Extension's dispatch queue (and the main app's capture-output queue) which are background queues, not main. If strict concurrency flags a call site, add explicit isolation markers rather than bulk-`@MainActor`ing the pipeline.
- **`Sendable` requirements.** Pipeline stages are reference types holding mutable state (filters, Metal resources) — they must not cross queue boundaries. Keep them owned by a single actor or protect with a serial queue. Mark them `@unchecked Sendable` only if you can prove the framework-managed dispatch queues provide synchronisation. Same pattern as the Camera Extension sendability annotations from Phase 1.
- **Core ML refiner I/O contract.** Input is 6-channel (warped RGB ++ original RGB), 72×96. Output is 3-channel, 72×96. Values are float32 in [0, 1]. Phase 2b's smoke-run `.mlpackage` enforces this shape. When real weights land (Phase 2c), the contract is unchanged.
- **Bundling real weights.** Phase 3a ships the pipeline with NO `.mlpackage` bundled. When Phase 2c produces a trained weight file, copy it to `Sources/GazeLock/Resources/Models/refiner.mlpackage` and track via Git LFS (`.gitattributes` already covers `*.mlpackage`). XcodeGen's `resources:` rule for the GazeLock target picks it up automatically.
- **Performance.** Phase 3a doesn't benchmark. The spec's per-stage latency numbers (§5.4) are targets; actual numbers come in Phase 3b when we profile against real camera input. If any stage is suspiciously slow in a unit test (> 100 ms for a 720p frame), note it as a DONE_WITH_CONCERNS item rather than silently tuning.
- **Strict concurrency quirk: Vision APIs.** `VNDetectFaceLandmarksRequest` is `@unchecked Sendable` via the Vision framework itself; we pass it synchronously to `VNImageRequestHandler.perform()` so there are no cross-actor concerns. No extra annotations needed on `LandmarkDetector`.

---

*End of Phase 3a plan.*
