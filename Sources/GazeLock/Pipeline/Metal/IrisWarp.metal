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
