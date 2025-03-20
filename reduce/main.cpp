#include <kompute/Kompute.hpp>
#include <random>
#include <vector>
#include <concepts>
#include <fmt/core.h>

#include "utils.hpp"

// Initialize Kompute with atomic float extension
const static std::vector<std::string> k_extensions = {"VK_EXT_shader_atomic_float"};
const static std::vector<uint32_t> k_family_queue_indices{};
static kp::Manager manager = kp::Manager(0, k_family_queue_indices, k_extensions);
const static float k_period = manager.getDeviceProperties().limits.timestampPeriod;

// Measure execution time with warm-up and averaging
template <std::invocable F>
double measure_time(F &&func, int warmup_runs = 10, int measurement_runs = 30) {
    // Warm-up runs
    for (int i = 0; i < warmup_runs; ++i) {
        func();
    }

    // Measurement runs
    std::vector<float> times(measurement_runs);
    for (int i = 0; i < measurement_runs; ++i) {
        times[i] = func();
    }

    // Calculate average time
    float sum = 0.0f;
    for (const auto time : times) {
        sum += time;
    }
    return sum / measurement_runs;
}

const static std::vector<uint32_t> k_spirv_naive = [] {
    constexpr auto data = bytes_to_words(
#include "reduce.spv.h"
    );
    return std::vector<uint32_t>(data.begin(), data.end());
}();

const static std::vector<uint32_t> k_spirv_wave = [] {
    constexpr auto data = bytes_to_words(
#include "reduce_wave.spv.h"
    );
    return std::vector<uint32_t>(data.begin(), data.end());
}();

const static std::vector<uint32_t> k_spirv_optimized = [] {
    constexpr auto data = bytes_to_words(
#include "reduce_optimized.spv.h"
    );
    return std::vector<uint32_t>(data.begin(), data.end());
}();

constexpr auto k_thread_group_size = 256u;
std::random_device rd;
std::mt19937 gen(42);

constexpr uint32_t calc_reduce_times(uint32_t length, uint32_t group_size) noexcept {
    if (length <= 1) return 0;

    const uint32_t a_msb_pos = 32u - std::countl_zero(length - 1);
    const uint32_t b_msb_pos = 32u - std::countl_zero(group_size - 1);

    return divide_and_round_up(a_msb_pos, b_msb_pos);
}

/// ret: time_consumed
float recursive_reduce(kp::Manager &mgr, std::vector<float> const &data, std::vector<uint32_t> const &spirv, uint32_t group_size, float &output) {
    // group_size must be a power of 2 and greater than 1
    assert((group_size & (group_size - 1)) == 0 && group_size > 1);

    const uint32_t length = data.size();
    if (length == 0) {
        output = 0.0f;
        return 0.0f;
    } else if (length == 1) {
        output = data[0];
        return 0.0f;
    }
    const auto reduce_times = calc_reduce_times(length, group_size);
    std::vector<std::shared_ptr<kp::TensorT<float>>> tensors;
    tensors.reserve(reduce_times + 1); // +1 for the final result tensor
    tensors.push_back(mgr.tensorT(data));
    mgr.sequence()->eval<kp::OpSyncDevice>({tensors[0]}); // upload data to GPU

    std::vector<std::shared_ptr<kp::Algorithm>> algorithms;
    for (auto i = 0u, l = length; i < reduce_times; ++i) {
        const auto dispatch_count = divide_and_round_up(l, group_size);
        tensors.push_back(mgr.tensorT<float>(dispatch_count));
        const auto algo = mgr.algorithm({tensors[i], tensors[i + 1]}, spirv, kp::Workgroup({dispatch_count, 1, 1}));
        algorithms.push_back(algo);
        l = divide_and_round_up(l, group_size);
    }

    // Record the algorithms
    const auto seq = mgr.sequence(0, reduce_times + 1);
    for (auto i = 0u; i < reduce_times; ++i) {
        seq->record<kp::OpAlgoDispatch>(algorithms[i]);
    }
    seq->eval();

    const auto timestamps = seq->getTimestamps();
    double time_consumed = 0.0;
    for (auto i = 0uz; i < reduce_times; ++i) {
        double dispatch_time = (timestamps[i + 1] - timestamps[i]) * k_period / 1000000.0f;
        time_consumed += dispatch_time;
    }

    seq->eval<kp::OpSyncLocal>({tensors[reduce_times]});
    output = tensors[reduce_times]->data()[0];
    return time_consumed;
}

int main() {

    // Test size (32M)
    constexpr uint32_t size = 1u << 25; // 32M

    fmt::println("Testing with size: {}", size);

    // Generate random input data
    std::vector<float> input_data(size);
    double cpu_sum = 0.0;
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (auto &val : input_data) {
        val = dis(gen);
        cpu_sum += val;
    }

    float result = 0.0f;

    const float time_consumed = measure_time([&] {
        return recursive_reduce(manager, input_data, k_spirv_optimized, k_thread_group_size * 2, result);
    });

    fmt::println("CPU sum: {}", cpu_sum);
    fmt::println("GPU sum: {}", result);
    fmt::println("Time consumed: {} ms", time_consumed);
    fmt::println("Relative error: {}", std::abs(cpu_sum - result) / cpu_sum);

    return 0;
}