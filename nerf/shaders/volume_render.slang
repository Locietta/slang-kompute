// nerf/shaders/volume_render.slang
struct RenderParams {
    uint batch_size;
    uint n_samples;
    float noise_std;
    uint white_bkgd;
};

[[vk::push_constant]]
RenderParams params;

[[vk::binding(0, 0)]]
StructuredBuffer<float> raw; // [batch_size, n_samples, 4]

[[vk::binding(1, 0)]]
StructuredBuffer<float> z_vals; // [batch_size, n_samples]

[[vk::binding(2, 0)]]
StructuredBuffer<float> rays_d; // [batch_size, 3]

[[vk::binding(3, 0)]]
StructuredBuffer<float> noise; // [batch_size, n_samples]

[[vk::binding(4, 0)]]
RWStructuredBuffer<float> weights; // [batch_size, n_samples]

[[vk::binding(5, 0)]]
RWStructuredBuffer<float> rgb_map; // [batch_size, 3]

[[vk::binding(6, 0)]]
RWStructuredBuffer<float> disp_map; // [batch_size]

[[vk::binding(7, 0)]]
RWStructuredBuffer<float> acc_map; // [batch_size]

[[vk::binding(8, 0)]]
RWStructuredBuffer<float> depth_map; // [batch_size]

// ReLU activation
float relu(float x) {
    return max(0.0f, x);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 DTid: SV_DispatchThreadID) {
    uint ray_idx = DTid.x;

    if (ray_idx >= params.batch_size) return;

    // Step 1: Compute distances between samples
    float dists[64]; // Maximum samples per ray (adjust if needed)

    // Get ray direction norm
    float ray_norm = sqrt(
        rays_d[ray_idx * 3 + 0] * rays_d[ray_idx * 3 + 0] +
        rays_d[ray_idx * 3 + 1] * rays_d[ray_idx * 3 + 1] +
        rays_d[ray_idx * 3 + 2] * rays_d[ray_idx * 3 + 2]);

    for (uint s = 0; s < params.n_samples; s++) {
        float z_dist = 0.0f;

        if (s < params.n_samples - 1) {
            z_dist = z_vals[ray_idx * params.n_samples + s + 1] -
                     z_vals[ray_idx * params.n_samples + s];
        } else {
            // For the last sample, use a very large value
            z_dist = 1e10f;
        }

        dists[s] = z_dist * ray_norm;
    }

    // Step 2: Compute alpha values and weights
    float T = 1.0f;          // Accumulated transmittance
    float weight_sum = 0.0f; // For normalization
    float depth_sum = 0.0f;
    float rgb[3] = { 0.0f, 0.0f, 0.0f };

    for (uint s = 0; s < params.n_samples; s++) {
        // Get raw values for this sample
        uint raw_base_idx = (ray_idx * params.n_samples + s) * 4;

        // Get density and add noise if needed
        float sigma = raw[raw_base_idx + 3]; // Density channel

        if (params.noise_std > 0.0f) {
            sigma += noise[ray_idx * params.n_samples + s];
        }

        // Apply activation (ReLU)
        sigma = relu(sigma);

        // Convert to alpha using distance
        float alpha = 1.0f - exp(-sigma * dists[s]);

        // Calculate weight for this sample
        float weight = T * alpha;
        weights[ray_idx * params.n_samples + s] = weight;

        // Update transmittance for next iteration
        T *= (1.0f - alpha);

        // Accumulate weighted RGB
        rgb[0] += weight * raw[raw_base_idx + 0];
        rgb[1] += weight * raw[raw_base_idx + 1];
        rgb[2] += weight * raw[raw_base_idx + 2];

        // Accumulate weighted depth
        depth_sum += weight * z_vals[ray_idx * params.n_samples + s];

        // Accumulate weight for final alpha
        weight_sum += weight;
    }

    // Step 3: Compute final outputs
    // If white background, add contribution from background
    if (bool(params.white_bkgd) && weight_sum < 1.0f) {
        rgb[0] += (1.0f - weight_sum);
        rgb[1] += (1.0f - weight_sum);
        rgb[2] += (1.0f - weight_sum);
    }

    // Store results
    rgb_map[ray_idx * 3 + 0] = rgb[0];
    rgb_map[ray_idx * 3 + 1] = rgb[1];
    rgb_map[ray_idx * 3 + 2] = rgb[2];

    // Calculate disparity (inverse depth)
    float eps = 1e-10f;
    float inv_eps = 1.0f / eps;
    float disp = 0.0f;

    if (weight_sum > 0.0f) {
        disp = 1.0f / max(eps, depth_sum / weight_sum);
    } else {
        disp = inv_eps;
    }

    disp_map[ray_idx] = disp;
    acc_map[ray_idx] = weight_sum;
    depth_map[ray_idx] = depth_sum;
}
