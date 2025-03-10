// nerf/shaders/nerf_network_layer.slang
struct LayerParams {
    uint input_dim;
    uint output_dim;
    uint batch_size;
    uint use_skip_connection;
};

[[vk::push_constant]]
LayerParams params;

[[vk::binding(0, 0)]]
StructuredBuffer<float> input;

[[vk::binding(1, 0)]]
RWStructuredBuffer<float> output;

[[vk::binding(2, 0)]]
StructuredBuffer<float> weights;

[[vk::binding(3, 0)]]
StructuredBuffer<float> bias;

// Skip connection input (optional)
[[vk::binding(4, 0)]]
StructuredBuffer<float> skip_input;

// ReLU activation function
float relu(float x) {
    return max(0.0f, x);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 DTid: SV_DispatchThreadID) {
    uint batch_idx = DTid.x;

    if (batch_idx >= params.batch_size) return;

    // Compute the base indices
    uint input_base = batch_idx * params.input_dim;
    uint output_base = batch_idx * params.output_dim;

    // For each output neuron
    for (uint out_idx = 0; out_idx < params.output_dim; out_idx++) {
        float sum = bias[out_idx];

        // Process the regular input
        for (uint in_idx = 0; in_idx < params.input_dim; in_idx++) {
            uint weight_idx = out_idx * params.input_dim + in_idx;
            sum += input[input_base + in_idx] * weights[weight_idx];
        }

        // Add skip connection if needed
        if (bool(params.use_skip_connection)) {
            uint skip_dim = params.input_dim; // Assuming skip input has same dimension as regular input
            for (uint skip_idx = 0; skip_idx < skip_dim; skip_idx++) {
                uint weight_idx = out_idx * (params.input_dim + skip_dim) + params.input_dim + skip_idx;
                sum += skip_input[batch_idx * skip_dim + skip_idx] * weights[weight_idx];
            }
        }

        // Apply ReLU activation and store the result
        output[output_base + out_idx] = relu(sum);
    }
}
