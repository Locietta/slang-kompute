#pragma once

#define THREAD_GROUP_SIZE 256

[[vk::binding(0, 0)]]
StructuredBuffer<float> source;

[[vk::binding(1, 0)]]
RWStructuredBuffer<float> result;