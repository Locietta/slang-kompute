
#pragma once
#define THREAD_GROUP_SIZE 256
#define WARP_SIZE 32
#define WAVE_PER_GROUP (THREAD_GROUP_SIZE / WARP_SIZE)

[[vk::binding(0, 0)]]
StructuredBuffer<float> source;

[[vk::binding(1, 0)]]
RWStructuredBuffer<float> result;
