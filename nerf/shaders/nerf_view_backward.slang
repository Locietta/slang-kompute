// nerf/shaders/nerf_view_backward.slang
struct ViewParams {
    uint batch_size;
    uint feature_dim;
    uint view_dim;
    uint rgb_dim;
};

[[vk::push_constant]]
ViewParams params;

// Input gradients and activations
[[vk::binding(0, 0)]]
StructuredBuffer<float> grad_output; // Gradient from output (RGB+density)
[[vk::binding(1, 0)]]
StructuredBuffer<float> features; // Extracted features
[[vk::binding(2, 0)]]
StructuredBuffer<float> view_dirs; // View directions

// Output gradients
[[vk::binding(3, 0)]]
RWStructuredBuffer<float> grad_features; // Gradient for features
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> grad_view_weights; // Gradient for view weights
[[vk::binding(5, 0)]]
RWStructuredBuffer<float> grad_view_bias; // Gradient for view bias
[[vk::binding(6, 0)]]
RWStructuredBuffer<float> grad_rgb_weights; // Gradient for RGB weights
[[vk::binding(7, 0)]]
RWStructuredBuffer<float> grad_rgb_bias; // Gradient for RGB bias

// Original weights
[[vk::binding(8, 0)]]
StructuredBuffer<float> view_weights; // Original view weights
[[vk::binding(9, 0)]]
StructuredBuffer<float> rgb_weights; // Original RGB weights
[[vk::binding(10, 0)]]
StructuredBuffer<float> view_bias; // Original view bias
[[vk::binding(11, 0)]]
StructuredBuffer<float> rgb_bias; // Original RGB bias
[[vk::binding(12, 0)]]
RWStructuredBuffer<float> grad_alpha; // Gradient for density (alpha)

// Derivative of sigmoid
float sigmoid_derivative(float x) {
    float s = 1.0f / (1.0f + exp(-x));
    return s * (1.0f - s);
}

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
    uint view_base = batch_idx * params.view_dim;
    uint output_base = batch_idx * (params.rgb_dim + 1); // RGB + density

    // Hidden layer size (typically half of feature_dim)
    uint hidden_dim = params.feature_dim / 2;

    // Store hidden layer activations
    float hidden[256];      // Maximum size for hidden layer
    float hidden_grad[256]; // Gradients for hidden layer

    // Initialize hidden gradients to zero
    for (uint h_idx = 0; h_idx < hidden_dim; h_idx++) {
        hidden_grad[h_idx] = 0.0f;
    }

    // 1. Compute hidden layer values (needed for backprop)
    for (uint h_idx = 0; h_idx < hidden_dim; h_idx++) {
        float sum = view_bias[h_idx]; // Include bias term as in forward pass

        // Process features
        for (uint f_idx = 0; f_idx < params.feature_dim; f_idx++) {
            uint weight_idx = h_idx * (params.feature_dim + params.view_dim) + f_idx;
            sum += features[feature_base + f_idx] * view_weights[weight_idx];
        }

        // Process view directions
        for (uint v_idx = 0; v_idx < params.view_dim; v_idx++) {
            uint weight_idx = h_idx * (params.feature_dim + params.view_dim) + params.feature_dim + v_idx;
            sum += view_dirs[view_base + v_idx] * view_weights[weight_idx];
        }

        // Apply ReLU
        hidden[h_idx] = max(0.0f, sum);
    }

    // 2. Backpropagate through RGB output layer
    for (uint rgb_idx = 0; rgb_idx < params.rgb_dim; rgb_idx++) {
        // Get gradient from output
        float grad_out = grad_output[output_base + rgb_idx];

        // Compute pre-activation value
        float pre_activation = rgb_bias[rgb_idx]; // Use original bias, not gradient buffer
        for (uint h_idx = 0; h_idx < hidden_dim; h_idx++) {
            uint weight_idx = rgb_idx * hidden_dim + h_idx;
            pre_activation += hidden[h_idx] * rgb_weights[weight_idx];
        }

        // Apply sigmoid derivative
        float grad = grad_out * sigmoid_derivative(pre_activation);

        // Update RGB bias gradient
        grad_rgb_bias[rgb_idx] += grad;

        // Update RGB weights and propagate to hidden layer
        for (uint h_idx = 0; h_idx < hidden_dim; h_idx++) {
            uint weight_idx = rgb_idx * hidden_dim + h_idx;

            // Gradient for RGB weights
            grad_rgb_weights[weight_idx] += grad * hidden[h_idx];

            // Propagate gradient to hidden layer
            hidden_grad[h_idx] += grad * rgb_weights[weight_idx];
        }
    }

    // Pass through the density gradient directly
    grad_alpha[batch_idx] = grad_output[output_base + params.rgb_dim];

    // 3. Backpropagate through hidden layer
    for (uint h_idx = 0; h_idx < hidden_dim; h_idx++) {
        // Compute pre-activation value
        float pre_activation = view_bias[h_idx]; // Include bias term as in forward pass

        // Process features
        for (uint f_idx = 0; f_idx < params.feature_dim; f_idx++) {
            uint weight_idx = h_idx * (params.feature_dim + params.view_dim) + f_idx;
            pre_activation += features[feature_base + f_idx] * view_weights[weight_idx];
        }

        // Process view directions
        for (uint v_idx = 0; v_idx < params.view_dim; v_idx++) {
            uint weight_idx = h_idx * (params.feature_dim + params.view_dim) + params.feature_dim + v_idx;
            pre_activation += view_dirs[view_base + v_idx] * view_weights[weight_idx];
        }

        // Apply ReLU derivative
        float grad = hidden_grad[h_idx] * relu_derivative(pre_activation);

        // Update view bias gradient
        grad_view_bias[h_idx] += grad;

        // Update view weights and propagate to inputs
        for (uint f_idx = 0; f_idx < params.feature_dim; f_idx++) {
            uint weight_idx = h_idx * (params.feature_dim + params.view_dim) + f_idx;

            // Gradient for view weights
            grad_view_weights[weight_idx] += grad * features[feature_base + f_idx];

            // Propagate gradient to features
            grad_features[feature_base + f_idx] += grad * view_weights[weight_idx];
        }

        for (uint v_idx = 0; v_idx < params.view_dim; v_idx++) {
            uint weight_idx = h_idx * (params.feature_dim + params.view_dim) + params.feature_dim + v_idx;

            // Gradient for view weights (view direction part)
            grad_view_weights[weight_idx] += grad * view_dirs[view_base + v_idx];

            // We don't need to propagate to view_dirs as they're input
        }
    }
}