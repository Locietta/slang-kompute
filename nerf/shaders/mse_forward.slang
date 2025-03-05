// nerf/shaders/mse_forward.slang

[[vk::binding(0, 0)]]
StructuredBuffer<float> predicted;

[[vk::binding(1, 0)]]
StructuredBuffer<float> target;

[[vk::binding(2, 0)]]
RWStructuredBuffer<float> loss_value;

groupshared float partial_sums[64];

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 DTid: SV_DispatchThreadID, uint3 GTid: SV_GroupThreadID, uint3 Gid: SV_GroupID) {
    uint idx = DTid.x;
    uint local_idx = GTid.x;
    uint num_elements = predicted.getCount();

    float squared_error = 0.0f;
    if (idx < num_elements) {
        float diff = predicted[idx] - target[idx];
        squared_error = diff * diff;
    }

    partial_sums[local_idx] = squared_error;
    GroupMemoryBarrierWithGroupSync(); // wait until everything is transfered from device memory to shared memory

    for (uint stride = 32; stride > 0; stride >>= 1) {
        if (local_idx < stride) {
            partial_sums[local_idx] += partial_sums[local_idx + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (local_idx == 0) {
        /// NOTE: loss_value[0] should be zero inited on CPU side
        InterlockedAdd(loss_value[0], partial_sums[0] / num_elements);
    }
}
