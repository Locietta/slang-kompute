// Define the number of threads per group - typically a power of 2

#include "util/common.slang"

// Shared memory for the workgroup
groupshared float warp_level_sums[WAVE_PER_GROUP];

[shader("compute")]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 DTid: SV_DispatchThreadID, uint3 GTid: SV_GroupThreadID, uint3 Gid: SV_GroupID) {
    uint local_idx = GTid.x;
    uint idx = Gid.x * THREAD_GROUP_SIZE * 2 + local_idx;
    uint num_elements = source.getCount();

    float sum = idx < num_elements ? source[idx] : 0.0f;
    sum += idx + THREAD_GROUP_SIZE < num_elements ? source[idx + THREAD_GROUP_SIZE] : 0.0f;

    uint lane_id = WaveGetLaneIndex();
    uint warp_id = local_idx / WARP_SIZE;

    sum = WaveActiveSum(sum);
    if (WaveIsFirstLane()) warp_level_sums[warp_id] = sum;

    // Synchronize all threads in the group to ensure all wave reductions are visible
    GroupMemoryBarrierWithGroupSync();

    sum = (local_idx < WAVE_PER_GROUP) ? warp_level_sums[lane_id] : 0.0f;
    if (warp_id == 0) sum = WaveActiveSum(sum); // Final reduce using first warp
    if (local_idx == 0) result[Gid.x] = sum; // Write the result for the group
}
