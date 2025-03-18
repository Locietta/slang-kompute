// reduce/shaders/reduce.slang

#include "util/common.slang"

groupshared float partial_sums[THREAD_GROUP_SIZE];

[shader("compute")]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 DTid: SV_DispatchThreadID, uint3 GTid: SV_GroupThreadID, uint3 Gid: SV_GroupID) {
    uint idx = DTid.x;
    uint local_idx = GTid.x;
    uint num_elements = source.getCount();
    partial_sums[local_idx] = idx < num_elements ? source[idx] : 0.0f;

    GroupMemoryBarrierWithGroupSync(); // wait until everything is transfered from device memory to shared memory

    for (uint offset = THREAD_GROUP_SIZE / 2; offset > 0; offset >>= 1) {
        if (local_idx < offset) {
            partial_sums[local_idx] += partial_sums[local_idx + offset];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (local_idx == 0) {
        /// NOTE: result[0] should be zero inited on CPU side
        InterlockedAdd(result[0], partial_sums[0]);
    }
}
