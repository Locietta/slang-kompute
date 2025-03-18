// Define the number of threads per group - typically a power of 2

#include "util/common.slang"

#define LANE_COUNT 32
#define WAVE_PER_GROUP (THREAD_GROUP_SIZE / LANE_COUNT)

// Shared memory for the workgroup
groupshared float shared[WAVE_PER_GROUP];

[shader("compute")]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 DTid: SV_DispatchThreadID, uint3 GTid: SV_GroupThreadID) {
    uint groupIndex = GTid.x;
    uint globalIndex = DTid.x;

    float value = source[globalIndex];

    // This leverages hardware wave operations for the first stage of reduction
    float waveReduction = WaveActiveSum(value);

    // Only the first lane in each wave writes to shared memory
    // This dramatically reduces shared memory operations
    if (WaveIsFirstLane()) {
        shared[groupIndex / LANE_COUNT] = waveReduction;
    }

    // Synchronize all threads in the group to ensure all wave reductions are visible
    GroupMemoryBarrierWithGroupSync();

    // Only the first thread of the first wave does the final reduction
    if (groupIndex == 0) {
        float sum = shared[0];
        [unroll]
        for (uint i = 1; i < WAVE_PER_GROUP; ++i) {
            sum += shared[i];
        }

        // Write the result for this workgroup to the output buffer
        /// NOTE: result[0] should be zero inited on CPU side
        InterlockedAdd(result[0], sum);
    }
}
