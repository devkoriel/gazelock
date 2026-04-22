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
