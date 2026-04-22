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
