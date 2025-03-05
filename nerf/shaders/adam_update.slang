// nerf/shaders/adam_update.slang
[[vk::constant_id(0)]]
uint param_size;

struct PushConstantData {
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    float beta1_t;
    float beta2_t;
};

[[vk::push_constant]]
PushConstantData params;

[[vk::binding(0, 0)]]
RWStructuredBuffer<float> parameter;

[[vk::binding(1, 0)]]
StructuredBuffer<float> gradient;

[[vk::binding(2, 0)]]
RWStructuredBuffer<float> exp_avg;

[[vk::binding(3, 0)]]
RWStructuredBuffer<float> exp_avg_sq;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 DTid: SV_DispatchThreadID) {
    uint idx = DTid.x;

    if (idx < param_size) {
        // Get current values
        float param = parameter[idx];
        float grad = gradient[idx];
        float m = exp_avg[idx];
        float v = exp_avg_sq[idx];

        // Update biased first moment estimate
        m = params.beta1 * m + (1.0f - params.beta1) * grad;

        // Update biased second raw moment estimate
        v = params.beta2 * v + (1.0f - params.beta2) * grad * grad;

        // Store updated moments
        exp_avg[idx] = m;
        exp_avg_sq[idx] = v;

        // Compute bias-corrected estimates
        float m_corrected = m / (1.0f - params.beta1_t);
        float v_corrected = v / (1.0f - params.beta2_t);

        // Update parameters
        param = param - params.learning_rate * m_corrected / (sqrt(v_corrected) + params.epsilon);

        // Write updated parameter
        parameter[idx] = param;
    }
}
