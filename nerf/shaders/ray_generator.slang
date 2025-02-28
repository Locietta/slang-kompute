// nerf/shaders/ray_generator.slang
struct CameraParams {
    float4x4 c2w;       // Camera to world transform
    float4x4 intrinsic; // Camera intrinsic matrix (padded to mat4)
    float2 resolution;  // Image resolution (width, height)
    float2 padding;     // Padding to ensure alignment
};

[[vk::push_constant]]
CameraParams camera;

[[vk::binding(0, 0)]]
RWStructuredBuffer<float3> ray_origins;
[[vk::binding(1, 0)]]
RWStructuredBuffer<float3> ray_directions;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 DTid: SV_DispatchThreadID) {
    uint x = DTid.x;
    uint y = DTid.y;

    if (x >= uint(camera.resolution.x) || y >= uint(camera.resolution.y)) {
        return;
    }

    uint pixel_idx = y * uint(camera.resolution.x) + x;

    // Get camera intrinsics
    float fx = camera.intrinsic[0][0];
    float fy = camera.intrinsic[1][1];
    float cx = camera.intrinsic[0][2];
    float cy = camera.intrinsic[1][2];

    // Generate ray direction in camera space
    float3 dir;
    dir.x = (float(x) - cx) / fx;
    dir.y = -(float(y) - cy) / fy; // Negative to match PyTorch convention
    dir.z = -1.0;                  // Looking in the negative z direction

    // Transform ray to world space
    float4 world_dir = mul(camera.c2w, float4(dir, 0.0));
    float4 world_origin = mul(camera.c2w, float4(0.0, 0.0, 0.0, 1.0));

    // Normalize direction
    float3 ray_dir = normalize(world_dir.xyz);

    // Store rays
    ray_origins[pixel_idx] = world_origin.xyz;
    ray_directions[pixel_idx] = ray_dir;
}
