struct LayerParams {
    uint input_dim;
    uint output_dim;
    uint batch_size;
    uint use_skip_connection;
};

[[vk::push_constant]]
LayerParams params;

// Input gradients and activations
[[vk::binding(0, 0)]]
StructuredBuffer<float> grad_output; // Gradient from next layer
[[vk::binding(1, 0)]]
StructuredBuffer<float> input; // Layer input
[[vk::binding(2, 0)]]
StructuredBuffer<float> output; // Layer output
[[vk::binding(3, 0)]]
StructuredBuffer<float> weights; // Layer weights
[[vk::binding(4, 0)]]
StructuredBuffer<float> bias; // Layer bias

// Output gradients
[[vk::binding(5, 0)]]
RWStructuredBuffer<float> grad_input; // Gradient for previous layer
[[vk::binding(6, 0)]]
RWStructuredBuffer<float> grad_weights; // Gradient for weights
[[vk::binding(7, 0)]]
RWStructuredBuffer<float> grad_bias; // Gradient for bias

// Optional skip connection
[[vk::binding(8, 0)]]
StructuredBuffer<float> skip_input; // Skip connection input
[[vk::binding(9, 0)]]
RWStructuredBuffer<float> grad_skip; // Gradient for skip input

// ReLU derivative
float relu_derivative(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 DTid: SV_DispatchThreadID) {
    uint batch_idx = DTid.x;
    if (batch_idx >= params.batch_size) return;

    // Base indices for this batch
    uint input_base = batch_idx * params.input_dim;
    uint output_base = batch_idx * params.output_dim;

    // Clear gradients for this batch
    for (uint in_idx = 0; in_idx < params.input_dim; in_idx++) {
        grad_input[input_base + in_idx] = 0.0f;
    }

    // Compute gradients
    for (uint out_idx = 0; out_idx < params.output_dim; out_idx++) {
        float output_val = output[output_base + out_idx];
        float grad_out = grad_output[output_base + out_idx];

        // Apply ReLU derivative
        float grad = grad_out * relu_derivative(output_val);

        // Accumulate bias gradients
        grad_bias[out_idx] += grad;

        // Compute gradients for weights and inputs
        for (uint in_idx = 0; in_idx < params.input_dim; in_idx++) {
            uint weight_idx = out_idx * params.input_dim + in_idx;

            // Gradient for weights
            float input_val = input[input_base + in_idx];
            grad_weights[weight_idx] += grad * input_val;

            // Gradient for inputs
            grad_input[input_base + in_idx] += grad * weights[weight_idx];
        }

        // Handle skip connection if present
        if (bool(params.use_skip_connection)) {
            uint skip_dim = params.input_dim; // Assuming skip input has same dimension as regular input

            for (uint skip_idx = 0; skip_idx < skip_dim; skip_idx++) {
                uint weight_idx = out_idx * (params.input_dim + skip_dim) + params.input_dim + skip_idx;
                grad_skip[batch_idx * skip_dim + skip_idx] += grad * weights[weight_idx];
            }
        }
    }
}
