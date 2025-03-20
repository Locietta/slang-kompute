// reduce/shaders/reduce_optimized.slang

#include "util/common.slang"

float WaveShuffleDown(float value, uint offset) {
    return spirv_asm {
        OpCapability GroupNonUniform;
        OpCapability GroupNonUniformShuffleRelative;
        OpExtension "SPV_KHR_shader_subgroup";
        
        result: $$float = OpGroupNonUniformShuffleDown Subgroup $value $offset
    };
}

float WarpReduce<let group_size : uint>(float sum) {
    if (group_size >= 32) sum += WaveShuffleDown(sum, 16);
    if (group_size >= 16) sum += WaveShuffleDown(sum, 8);
    if (group_size >= 8) sum += WaveShuffleDown(sum, 4);
    if (group_size >= 4) sum += WaveShuffleDown(sum, 2);
    if (group_size >= 2) sum += WaveShuffleDown(sum, 1);
    return sum;
}

groupshared float warp_level_sums[WAVE_PER_GROUP];

[shader("compute")]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 DTid: SV_DispatchThreadID, uint3 GTid: SV_GroupThreadID, uint3 Gid: SV_GroupID) {
    uint local_idx = GTid.x;
    uint num_elements = source.getCount();
    uint idx = Gid.x * THREAD_GROUP_SIZE * 2 + local_idx;

    float sum = idx < num_elements ? source[idx] : 0.0f;
    sum += idx + THREAD_GROUP_SIZE < num_elements ? source[idx + THREAD_GROUP_SIZE] : 0.0f;

    uint lane_id = WaveGetLaneIndex();
    uint warp_id = local_idx / WARP_SIZE;

    sum = WarpReduce<WARP_SIZE>(sum);
    if (WaveIsFirstLane()) warp_level_sums[warp_id] = sum;
    // wait until everything is transferred from device memory to shared memory
    GroupMemoryBarrierWithGroupSync();

    sum = (local_idx < WAVE_PER_GROUP) ? warp_level_sums[lane_id] : 0.0f;

    if (warp_id == 0) sum = WarpReduce<WAVE_PER_GROUP>(sum); // Final reduce using first warp

    if (local_idx == 0) result[Gid.x] = sum;
}
