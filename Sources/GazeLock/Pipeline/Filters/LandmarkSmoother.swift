import Foundation

/// Applies `OneEuroFilter` independently to each dimension of each
/// landmark.
///
/// Vision's face-landmark regions return a variable number of points
/// depending on the constellation the detector picks (typically 6-11 per
/// eye). We adapt the filter pool on each call — growing when Vision
/// returns more points, keeping existing filter state for the indices
/// it has seen before.
public final class LandmarkSmoother {
    private var filters: [(OneEuroFilter, OneEuroFilter)]
    private let minCutoff: Double
    private let beta: Double

    public init(count: Int = 0, minCutoff: Double = 1.0, beta: Double = 0.007) {
        self.minCutoff = minCutoff
        self.beta = beta
        self.filters = (0..<count).map { _ in
            (
                OneEuroFilter(minCutoff: minCutoff, beta: beta),
                OneEuroFilter(minCutoff: minCutoff, beta: beta)
            )
        }
    }

    public func smooth(_ landmarks: [Vec2], timestamp: TimeInterval) -> [Vec2] {
        // Grow the filter pool if Vision gave us more points this frame.
        while filters.count < landmarks.count {
            filters.append((
                OneEuroFilter(minCutoff: minCutoff, beta: beta),
                OneEuroFilter(minCutoff: minCutoff, beta: beta)
            ))
        }
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
