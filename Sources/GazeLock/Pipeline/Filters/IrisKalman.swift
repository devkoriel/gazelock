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
