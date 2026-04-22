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
