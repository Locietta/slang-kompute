// nerf/shaders/positional_encoder.slang
// Specialization constants
[[vk::constant_id(0)]]
const uint num_freqs = 10;

[[vk::constant_id(1)]]
const uint include_input = 1; // Use uint instead of bool for specialization constant

[[vk::binding(0, 0)]]
StructuredBuffer<float> frequency_bands;

[[vk::binding(1, 0)]]
StructuredBuffer<float> inputs;

[[vk::binding(2, 0)]]
RWStructuredBuffer<float> encoded;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 DTid: SV_DispatchThreadID) {
    uint vector_idx = DTid.x;

    // Number of vectors to process
    uint num_vectors = uint(inputs.getCount()) / 3;

    if (vector_idx >= num_vectors) return;

    // Read input vector (assuming 3D position or direction)
    float3 input_vec = float3(
        inputs[vector_idx * 3 + 0],
        inputs[vector_idx * 3 + 1],
        inputs[vector_idx * 3 + 2]);

    // Calculate base offset for this vector in the output
    uint output_stride = uint(encoded.getCount()) / num_vectors;
    uint base_offset = vector_idx * output_stride;
    uint offset = 0;

    // First copy the input if needed
    if (bool(include_input)) {
        encoded[base_offset + 0] = input_vec.x;
        encoded[base_offset + 1] = input_vec.y;
        encoded[base_offset + 2] = input_vec.z;
        offset = 3;
    }

    // Now add sin/cos of each frequency
    for (uint freq_idx = 0; freq_idx < num_freqs; freq_idx++) {
        float freq = frequency_bands[freq_idx];

        // For each input dimension
        for (uint dim = 0; dim < 3; dim++) {
            float x = input_vec[dim] * freq;

            // Calculate output position for this (freq, dim) pair
            uint pos = base_offset + offset;

            // Write sin and cos values
            encoded[pos] = sin(x);
            encoded[pos + 1] = cos(x);

            offset += 2;
        }
    }
}
