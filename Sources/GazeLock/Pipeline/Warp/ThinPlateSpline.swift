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
