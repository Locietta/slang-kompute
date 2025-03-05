// nerf/shaders/volume_render_backward.slang
struct RenderParams {
    uint batch_size;
    uint n_samples;
    float noise_std;
    uint white_bkgd;
};

[[vk::push_constant]]
RenderParams params;

// Input tensors (same as forward pass)
[[vk::binding(0, 0)]]
StructuredBuffer<float> raw; // [batch_size, n_samples, 4]

[[vk::binding(1, 0)]]
StructuredBuffer<float> z_vals; // [batch_size, n_samples]

[[vk::binding(2, 0)]]
StructuredBuffer<float> rays_d; // [batch_size, 3]

[[vk::binding(3, 0)]]
StructuredBuffer<float> weights; // [batch_size, n_samples]

// Gradient input and outputs
[[vk::binding(4, 0)]]
StructuredBuffer<float> grad_rgb; // [batch_size, 3] - Gradient of loss w.r.t RGB

[[vk::binding(5, 0)]]
StructuredBuffer<float> grad_disp; // [batch_size] - Gradient of loss w.r.t disparity (optional)

[[vk::binding(6, 0)]]
StructuredBuffer<float> grad_acc; // [batch_size] - Gradient of loss w.r.t accumulation (optional)

[[vk::binding(7, 0)]]
RWStructuredBuffer<float> grad_raw; // [batch_size, n_samples, 4] - Gradient w.r.t raw output

// Helper function for ReLU derivative
float d_relu(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 DTid: SV_DispatchThreadID) {
    uint ray_idx = DTid.x;

    if (ray_idx >= params.batch_size) return;

    // Get ray direction norm
    float ray_norm = sqrt(
        rays_d[ray_idx * 3 + 0] * rays_d[ray_idx * 3 + 0] +
        rays_d[ray_idx * 3 + 1] * rays_d[ray_idx * 3 + 1] +
        rays_d[ray_idx * 3 + 2] * rays_d[ray_idx * 3 + 2]);

    // Compute distances between samples (same as forward pass)
    float dists[64]; // Maximum samples per ray
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

    // Compute accumulated transmittance T at each step
    float T_vals[64]; // Store transmittance at each step
    T_vals[0] = 1.0f; // Start with full transmittance

    for (uint s = 0; s < params.n_samples - 1; s++) {
        // Extract raw density and compute alpha
        uint raw_base_idx = (ray_idx * params.n_samples + s) * 4;
        float sigma = raw[raw_base_idx + 3]; // Density

        // Apply activation (ReLU)
        sigma = max(0.0f, sigma);

        // Compute alpha
        float alpha = 1.0f - exp(-sigma * dists[s]);

        // Update transmittance for next step
        T_vals[s + 1] = T_vals[s] * (1.0f - alpha);
    }

    // Get incoming gradients for ray
    float incoming_grad_rgb[3];
    incoming_grad_rgb[0] = grad_rgb[ray_idx * 3 + 0];
    incoming_grad_rgb[1] = grad_rgb[ray_idx * 3 + 1];
    incoming_grad_rgb[2] = grad_rgb[ray_idx * 3 + 2];

    float incoming_grad_acc = grad_acc[ray_idx];
    float incoming_grad_disp = grad_disp[ray_idx];

    // Collect weights and compute weight sum
    float weights_sum = 0.0f;
    float depth_sum = 0.0f;

    for (uint s = 0; s < params.n_samples; s++) {
        float weight = weights[ray_idx * params.n_samples + s];
        weights_sum += weight;
        depth_sum += weight * z_vals[ray_idx * params.n_samples + s];
    }

    // Backpropagate gradients through each sample
    for (uint s = 0; s < params.n_samples; s++) {
        uint raw_base_idx = (ray_idx * params.n_samples + s) * 4;
        float weight = weights[ray_idx * params.n_samples + s];
        float depth_val = z_vals[ray_idx * params.n_samples + s];

        // Extract sigma and apply activation
        float sigma = raw[raw_base_idx + 3];
        bool active = sigma > 0.0f; // For ReLU derivative
        sigma = max(0.0f, sigma);

        // Compute alpha
        float alpha = 1.0f - exp(-sigma * dists[s]);

        // ------ Backpropagate through RGB ------
        // dL/drgb_raw = dL/drgb * weight
        grad_raw[raw_base_idx + 0] = incoming_grad_rgb[0] * weight;
        grad_raw[raw_base_idx + 1] = incoming_grad_rgb[1] * weight;
        grad_raw[raw_base_idx + 2] = incoming_grad_rgb[2] * weight;

        // ------ Backpropagate through density ------
        // Init gradient for sigma
        float grad_sigma = 0.0f;

        // RGB gradient contribution
        // Each density affects RGB output through weight
        float rgb_contribution =
            incoming_grad_rgb[0] * raw[raw_base_idx + 0] +
            incoming_grad_rgb[1] * raw[raw_base_idx + 1] +
            incoming_grad_rgb[2] * raw[raw_base_idx + 2];

        // Handle white background contribution if enabled
        if (bool(params.white_bkgd)) {
            rgb_contribution -= incoming_grad_rgb[0] + incoming_grad_rgb[1] + incoming_grad_rgb[2];
        }

        // Compute gradient through transmittance
        float grad_weight = rgb_contribution + incoming_grad_acc;

        // Handle disparity gradient contribution
        if (weights_sum > 0.0f) {
            float disp = 1.0f / max(1e-10f, depth_sum / weights_sum);
            float depth_avg = depth_sum / weights_sum;

            // ddisp/dweight = -depth/weight_sum² + z/weight_sum if depth_i = z
            float ddisp_dweight = -depth_avg / (weights_sum * weights_sum) +
                                  depth_val / weights_sum;

            // Convert to weight_i gradient through disparity
            grad_weight += incoming_grad_disp * ddisp_dweight * disp * disp;
        }

        // Gradient of weight w.r.t alpha: dweight/dalpha = T
        float dweight_dalpha = T_vals[s];

        // Gradient of alpha w.r.t sigma: dalpha/dsigma = (1-alpha) * dist
        float dalpha_dsigma = (1.0f - alpha) * dists[s];

        // Backpropagate from weight to sigma
        grad_sigma += grad_weight * dweight_dalpha * dalpha_dsigma;

        // Backpropagate through all subsequent weights (changes in transmittance)
        // Each alpha affects all subsequent weights through cumprod
        for (uint j = s + 1; j < params.n_samples; j++) {
            uint next_raw_idx = (ray_idx * params.n_samples + j) * 4;
            float next_weight = weights[ray_idx * params.n_samples + j];
            float next_depth = z_vals[ray_idx * params.n_samples + j];

            // Compute gradient contribution for this later sample
            float later_rgb_contribution =
                incoming_grad_rgb[0] * raw[next_raw_idx + 0] +
                incoming_grad_rgb[1] * raw[next_raw_idx + 1] +
                incoming_grad_rgb[2] * raw[next_raw_idx + 2];

            float later_grad_weight = later_rgb_contribution + incoming_grad_acc;

            // Handle disparity gradient
            if (weights_sum > 0.0f) {
                float disp = 1.0f / max(1e-10f, depth_sum / weights_sum);
                float depth_avg = depth_sum / weights_sum;

                float ddisp_dweight = -depth_avg / (weights_sum * weights_sum) +
                                      next_depth / weights_sum;

                later_grad_weight += incoming_grad_disp * ddisp_dweight * disp * disp;
            }

            // Effect on later weight: dw_j/dalpha_i = -T_j / (1-alpha_i)
            // Note: we divide by (1-alpha) and multiply by derivative of alpha w.r.t sigma
            float dw_dalpha = -next_weight / (1.0f - alpha);

            // Gradient contribution
            grad_sigma += later_grad_weight * dw_dalpha * dalpha_dsigma;
        }

        // Apply ReLU derivative and store gradient for density
        grad_raw[raw_base_idx + 3] = grad_sigma * d_relu(raw[raw_base_idx + 3]);
    }
}
