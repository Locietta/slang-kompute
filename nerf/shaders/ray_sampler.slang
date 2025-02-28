// nerf/shaders/ray_sampler.slang
struct SamplingParams {
    float near;       // Near plane distance
    float far;        // Far plane distance
    uint num_samples; // Number of samples per ray
};

[[vk::push_constant]]
SamplingParams params;

[SpecializationConstant]
const uint perturb = 1;

[[vk::binding(0, 0)]]
StructuredBuffer<float> ray_origins;
[[vk::binding(1, 0)]]
StructuredBuffer<float> ray_directions;
[[vk::binding(2, 0)]]
RWStructuredBuffer<float> sample_positions;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float> sample_directions;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> sample_z_vals;
[[vk::binding(5, 0)]]
StructuredBuffer<float> randoms;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 DTid: SV_DispatchThreadID) {
    uint ray_idx = DTid.x;

    if (ray_idx >= params.num_samples) return;

    // Get ray origin and direction
    float3 origin = float3(ray_origins[ray_idx * 3 + 0], ray_origins[ray_idx * 3 + 1], ray_origins[ray_idx * 3 + 2]);
    float3 direction = float3(ray_directions[ray_idx * 3 + 0], ray_directions[ray_idx * 3 + 1], ray_directions[ray_idx * 3 + 2]);

    // Generate t values along ray (stratified sampling)
    float t_vals[64]; // Assuming max samples is 64

    // Uniform samples
    for (uint i = 0; i < params.num_samples; i++) {
        t_vals[i] = float(i) / float(params.num_samples - 1);
    }

    // Apply perturbation if needed
    if (bool(perturb)) {
        for (uint i = 0; i < params.num_samples; i++) {
            t_vals[i] = randoms[ray_idx * params.num_samples + i];
        }
    }

    // Map t values to near-far range
    for (uint i = 0; i < params.num_samples; i++) {
        float z = params.near * (1.0 - t_vals[i]) + params.far * t_vals[i];
        uint sample_idx = ray_idx * params.num_samples + i;

        // Calculate and store sample position
        float3 pos = origin + z * direction;
        sample_positions[3 * sample_idx + 0] = pos.x;
        sample_positions[3 * sample_idx + 1] = pos.y;
        sample_positions[3 * sample_idx + 2] = pos.z;

        // Store direction and z value
        sample_directions[3 * sample_idx + 0] = direction.x;
        sample_directions[3 * sample_idx + 1] = direction.y;
        sample_directions[3 * sample_idx + 2] = direction.z;
        sample_z_vals[sample_idx] = z;
    }
}
