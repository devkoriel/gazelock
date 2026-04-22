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
