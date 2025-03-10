// nerf/shaders/nerf_feature_backward.slang
struct FeatureParams {
    uint batch_size;
    uint feature_dim;
};

[[vk::push_constant]]
FeatureParams params;

// Input gradients and activations
[[vk::binding(0, 0)]]
StructuredBuffer<float> grad_features; // Gradient from view branch
[[vk::binding(1, 0)]]
StructuredBuffer<float> grad_output; // Original output gradient (for density)
[[vk::binding(2, 0)]]
StructuredBuffer<float> hidden_features; // Last layer features

// Output gradients
[[vk::binding(3, 0)]]
RWStructuredBuffer<float> grad_feature_weights; // Gradient for feature weights
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> grad_feature_bias; // Gradient for feature bias
[[vk::binding(5, 0)]]
RWStructuredBuffer<float> grad_alpha_weights; // Gradient for alpha weights
[[vk::binding(6, 0)]]
RWStructuredBuffer<float> grad_alpha_bias; // Gradient for alpha bias
[[vk::binding(7, 0)]]
RWStructuredBuffer<float> grad_hidden; // Gradient for MLP output

// Original weights and biases
[[vk::binding(8, 0)]]
StructuredBuffer<float> feature_weights; // Original feature weights
[[vk::binding(9, 0)]]
StructuredBuffer<float> alpha_weights; // Original alpha weights
[[vk::binding(10, 0)]]
StructuredBuffer<float> feature_bias; // Original feature bias

// ReLU derivative
float relu_derivative(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 DTid: SV_DispatchThreadID) {
    uint batch_idx = DTid.x;
    if (batch_idx >= params.batch_size) return;

    // Compute base indices
    uint feature_base = batch_idx * params.feature_dim;
    uint hidden_base = batch_idx * params.feature_dim;
    uint output_base = batch_idx * 4; // RGBA output (4 channels)

    // 1. Backpropagate through density (alpha) output
    float grad_alpha = grad_output[output_base + 3]; // Alpha is the 4th component

    // Update alpha bias gradient
    grad_alpha_bias[0] += grad_alpha;

    // Update alpha weights and propagate to hidden features
    for (uint in_idx = 0; in_idx < params.feature_dim; in_idx++) {
        // Gradient for alpha weights
        grad_alpha_weights[in_idx] += grad_alpha * hidden_features[hidden_base + in_idx];

        // Propagate gradient to hidden features (will be accumulated below)
        grad_hidden[hidden_base + in_idx] += grad_alpha * alpha_weights[in_idx];
    }

    // 2. Backpropagate through feature extraction
    for (uint feat_idx = 0; feat_idx < params.feature_dim; feat_idx++) {
        // Get gradient from view branch
        float grad_feat = grad_features[feature_base + feat_idx];

        // Compute pre-activation value
        float pre_activation = feature_bias[feat_idx];
        for (uint in_idx = 0; in_idx < params.feature_dim; in_idx++) {
            uint weight_idx = feat_idx * params.feature_dim + in_idx;
            pre_activation += hidden_features[hidden_base + in_idx] * feature_weights[weight_idx];
        }

        // Apply ReLU derivative
        float grad = grad_feat * relu_derivative(pre_activation);

        // Update feature bias gradient
        grad_feature_bias[feat_idx] += grad;

        // Update feature weights and propagate to hidden features
        for (uint in_idx = 0; in_idx < params.feature_dim; in_idx++) {
            uint weight_idx = feat_idx * params.feature_dim + in_idx;

            // Gradient for feature weights
            grad_feature_weights[weight_idx] += grad * hidden_features[hidden_base + in_idx];

            // Propagate gradient to hidden features (accumulate with alpha gradient)
            grad_hidden[hidden_base + in_idx] += grad * feature_weights[weight_idx];
        }
    }
}
