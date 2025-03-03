// nerf/shaders/nerf_feature_extraction.slang
struct FeatureParams {
    uint batch_size;
    uint feature_dim;
};

[[vk::push_constant]]
FeatureParams params;

[[vk::binding(0, 0)]]
StructuredBuffer<float> hidden_features; // Last layer output

[[vk::binding(1, 0)]]
RWStructuredBuffer<float> features; // Extracted features

[[vk::binding(2, 0)]]
RWStructuredBuffer<float> alpha; // Density output

[[vk::binding(3, 0)]]
StructuredBuffer<float> feature_weights;

[[vk::binding(4, 0)]]
StructuredBuffer<float> feature_bias;

[[vk::binding(5, 0)]]
StructuredBuffer<float> alpha_weights;

[[vk::binding(6, 0)]]
StructuredBuffer<float> alpha_bias;

// ReLU activation function
float relu(float x) {
    return max(0.0f, x);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 DTid: SV_DispatchThreadID) {
    uint batch_idx = DTid.x;

    if (batch_idx >= params.batch_size) return;

    // Compute base indices
    uint input_base = batch_idx * params.feature_dim;

    // Extract features - linear layer with ReLU
    for (uint feat_idx = 0; feat_idx < params.feature_dim; feat_idx++) {
        float sum = feature_bias[feat_idx];

        for (uint in_idx = 0; in_idx < params.feature_dim; in_idx++) {
            uint weight_idx = feat_idx * params.feature_dim + in_idx;
            sum += hidden_features[input_base + in_idx] * feature_weights[weight_idx];
        }

        // Store extracted feature with ReLU
        features[batch_idx * params.feature_dim + feat_idx] = relu(sum);
    }

    // Compute alpha (density) - single output
    float alpha_sum = alpha_bias[0];

    for (uint in_idx = 0; in_idx < params.feature_dim; in_idx++) {
        alpha_sum += hidden_features[input_base + in_idx] * alpha_weights[in_idx];
    }

    // Store alpha (no activation yet - will be applied during volume rendering)
    alpha[batch_idx] = alpha_sum;
}
