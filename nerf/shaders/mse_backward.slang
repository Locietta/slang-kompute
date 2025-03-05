// nerf/shaders/mse_backward.slang
[[vk::binding(0, 0)]]
StructuredBuffer<float> predicted;

[[vk::binding(1, 0)]]
StructuredBuffer<float> target;

[[vk::binding(2, 0)]]
RWStructuredBuffer<float> grad_output;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 DTid: SV_DispatchThreadID) {
    uint idx = DTid.x;
    uint num_elements = grad_output.getCount();

    if (idx < num_elements) {
        // MSE derivative: 2 * (predicted - target) / N
        float diff = predicted[idx] - target[idx];
        grad_output[idx] = 2.0f * diff / num_elements;
    }
}
