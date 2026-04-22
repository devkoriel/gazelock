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
