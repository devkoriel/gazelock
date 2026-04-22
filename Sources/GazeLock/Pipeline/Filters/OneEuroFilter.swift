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
