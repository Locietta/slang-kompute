// nerf/shaders/mse_forward.slang

[[vk::binding(0, 0)]]
StructuredBuffer<float> source;

[[vk::binding(1, 0)]]
RWStructuredBuffer<float> result;

groupshared float partial_sums[64];

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 DTid: SV_DispatchThreadID, uint3 GTid: SV_GroupThreadID, uint3 Gid: SV_GroupID) {
    uint idx = DTid.x;
    uint local_idx = GTid.x;
    uint num_elements = source.getCount();
    partial_sums[local_idx] = idx < num_elements ? source[idx] : 0.0f;

    GroupMemoryBarrierWithGroupSync(); // wait until everything is transfered from device memory to shared memory

    for (uint offset = 32; offset > 0; offset >>= 1) {
        if (local_idx < offset) {
            partial_sums[local_idx] += partial_sums[local_idx + offset];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    /// DEBUG: WARP unroll not working for me?

    // // WARP count: AMD - 64 , NVIDIA - 32
    // // manually unrolled: don't need group sync in the same WARP
    // if (local_idx < 32) partial_sums[local_idx] += partial_sums[local_idx + 32];
    // if (local_idx < 16) partial_sums[local_idx] += partial_sums[local_idx + 16];
    // if (local_idx < 8) partial_sums[local_idx] += partial_sums[local_idx + 8];
    // if (local_idx < 4) partial_sums[local_idx] += partial_sums[local_idx + 4];
    // if (local_idx < 2) partial_sums[local_idx] += partial_sums[local_idx + 2];
    // if (local_idx < 1) partial_sums[local_idx] += partial_sums[local_idx + 1];

    // GroupMemoryBarrier();

    if (local_idx == 0) {
        /// NOTE: result[0] should be zero inited on CPU side
        InterlockedAdd(result[0], partial_sums[0]);
    }
}
