// nerf/shaders/nerf_view_dependent.slang
struct ViewParams {
    uint batch_size;
    uint feature_dim;
    uint view_dim;
    uint rgb_dim;
};

[[vk::push_constant]]
ViewParams params;

[[vk::binding(0, 0)]]
StructuredBuffer<float> features; // Extracted features

[[vk::binding(1, 0)]]
StructuredBuffer<float> view_dirs; // Encoded view directions

[[vk::binding(2, 0)]]
StructuredBuffer<float> alpha; // Density from feature extraction

[[vk::binding(3, 0)]]
RWStructuredBuffer<float> output; // Final RGB+alpha output

[[vk::binding(4, 0)]]
StructuredBuffer<float> view_weights;

[[vk::binding(5, 0)]]
StructuredBuffer<float> view_bias;

[[vk::binding(6, 0)]]
StructuredBuffer<float> rgb_weights;

[[vk::binding(7, 0)]]
StructuredBuffer<float> rgb_bias;

// ReLU activation function
float relu(float x) {
    return max(0.0f, x);
}

// Sigmoid activation for RGB
float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 DTid: SV_DispatchThreadID) {
    uint batch_idx = DTid.x;

    if (batch_idx >= params.batch_size) return;

    // Concatenate features and view directions by computing dot products directly
    uint hidden_dim = params.feature_dim / 2; // Hidden layer is typically half size

    // Process through the view MLP layer
    float hidden[256]; // Maximum size for hidden layer

    // Compute hidden layer (features + view_dirs -> hidden)
    for (uint h_idx = 0; h_idx < hidden_dim; h_idx++) {
        float sum = view_bias[h_idx];

        // Process features
        for (uint f_idx = 0; f_idx < params.feature_dim; f_idx++) {
            uint weight_idx = h_idx * (params.feature_dim + params.view_dim) + f_idx;
            sum += features[batch_idx * params.feature_dim + f_idx] * view_weights[weight_idx];
        }

        // Process view directions
        for (uint v_idx = 0; v_idx < params.view_dim; v_idx++) {
            uint weight_idx = h_idx * (params.feature_dim + params.view_dim) + params.feature_dim + v_idx;
            sum += view_dirs[batch_idx * params.view_dim + v_idx] * view_weights[weight_idx];
        }

        // Store with ReLU
        hidden[h_idx] = relu(sum);
    }

    // Compute RGB output
    for (uint rgb_idx = 0; rgb_idx < params.rgb_dim; rgb_idx++) {
        float sum = rgb_bias[rgb_idx];

        for (uint h_idx = 0; h_idx < hidden_dim; h_idx++) {
            uint weight_idx = rgb_idx * hidden_dim + h_idx;
            sum += hidden[h_idx] * rgb_weights[weight_idx];
        }

        // Store with sigmoid activation for RGB
        output[batch_idx * (params.rgb_dim + 1) + rgb_idx] = sigmoid(sum);
    }

    // Add alpha to output
    output[batch_idx * (params.rgb_dim + 1) + params.rgb_dim] = alpha[batch_idx];
}
