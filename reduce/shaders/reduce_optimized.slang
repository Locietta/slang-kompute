// reduce/shaders/reduce_optimized.slang

#include "util/common.slang"

#define LANE_COUNT 32

groupshared float partial_sums[THREAD_GROUP_SIZE];

[shader("compute")]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 DTid: SV_DispatchThreadID, uint3 GTid: SV_GroupThreadID, uint3 Gid: SV_GroupID) {
    uint local_idx = GTid.x;
    uint idx = Gid.x * THREAD_GROUP_SIZE * 2 + local_idx;
    uint num_elements = source.getCount();
    partial_sums[local_idx] = idx < num_elements ? source[idx] : 0.0f;
    partial_sums[local_idx] += idx + THREAD_GROUP_SIZE < num_elements ? source[idx + THREAD_GROUP_SIZE] : 0.0f;

    GroupMemoryBarrierWithGroupSync(); // wait until everything is transfered from device memory to shared memory

    [ForceUnroll]
    for (uint offset = THREAD_GROUP_SIZE / 2; offset > LANE_COUNT; offset >>= 1) {
        if (local_idx < offset) {
            partial_sums[local_idx] += partial_sums[local_idx + offset];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // use wave intrinsics to reduce the last warp
    static_assert(THREAD_GROUP_SIZE >= LANE_COUNT * 2, "Reduce group size must be at least 2 * LANE_COUNT");
    float temp = partial_sums[local_idx] + partial_sums[local_idx + LANE_COUNT];
    float total_sum = WaveActiveSum(temp);
    if (local_idx == 0) {
        /// NOTE: result[0] should be zero inited before this pass
        InterlockedAdd(result[0], total_sum);
    }
}
