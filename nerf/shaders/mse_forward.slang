// nerf/shaders/mse_forward.slang

#define THREAD_GROUP_SIZE 256
#define LANE_COUNT 32
#define WAVE_PER_GROUP (THREAD_GROUP_SIZE / LANE_COUNT)

[[vk::binding(0, 0)]]
StructuredBuffer<float> predicted;

[[vk::binding(1, 0)]]
StructuredBuffer<float> target;

[[vk::binding(2, 0)]]
RWStructuredBuffer<float> loss_value;

groupshared float partial_sums[WAVE_PER_GROUP];

[shader("compute")]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 DTid: SV_DispatchThreadID, uint3 GTid: SV_GroupThreadID) {
    uint idx = DTid.x;
    uint local_idx = GTid.x;
    uint num_elements = predicted.getCount();

    float error = (idx < num_elements) ? (predicted[idx] - target[idx]) : 0.0f;
    float squared_error = error * error;

    float waveReduction = WaveActiveSum(squared_error);

    if (WaveIsFirstLane()) {
        partial_sums[local_idx / LANE_COUNT] = waveReduction;
    }

    // Synchronize all threads in the group to ensure all wave reductions are visible
    GroupMemoryBarrierWithGroupSync();

    if (local_idx == 0) {
        float sum = partial_sums[0];
        [unroll]
        for (uint i = 1; i < WAVE_PER_GROUP; ++i) {
            sum += partial_sums[i];
        }

        /// NOTE: loss_value[0] should be zero inited on CPU side
        InterlockedAdd(loss_value[0], sum / num_elements);
    }
}
