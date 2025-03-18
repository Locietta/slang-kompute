#include <kompute/Kompute.hpp>
#include <random>
#include <vector>
#include <chrono>
#include <concepts>
#include <fmt/core.h>

#include "utils.hpp"
#include "OpClear.h"

static float period = 0.0f;

// Measure execution time with warm-up and averaging
template <std::invocable F, std::invocable F2>
double measure_time(F &&func, F2 &&cleaner, int warmup_runs = 10, int measurement_runs = 30) {
    // Warm-up runs
    for (int i = 0; i < warmup_runs; ++i) {
        func();
    }

    // Measurement runs
    std::vector<double> times(measurement_runs);
    for (int i = 0; i < measurement_runs; ++i) {
        cleaner();
        auto time_stampes = func();
        times[i] = static_cast<float>(time_stampes[1] - time_stampes[0]) * period / 1000000.0f;
    }

    // Calculate average time
    double sum = 0.0;
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

int main() {
    // Initialize Kompute with atomic float extension
    const std::vector<std::string> extensions = {"VK_EXT_shader_atomic_float"};
    const std::vector<uint32_t> family_queue_indices;

    kp::Manager manager = kp::Manager(0, family_queue_indices, extensions);

    period = manager.getDeviceProperties().limits.timestampPeriod;

    // Test size (32M)
    constexpr uint32_t size = 1u << 25; // 32M

    fmt::println("Testing with size: {}", size);

    // std::this_thread::sleep_for(std::chrono::seconds(5));
    // system("pause");

    // Generate random input data
    std::vector<float> input_data(size);
    float cpu_sum = 0.0f;
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (auto &val : input_data) {
        val = dis(gen);
        cpu_sum += val;
    }

    // Create Kompute tensors
    auto tensor_input = manager.tensorT(input_data);
    auto tensor_output = manager.tensorT<float>(1);           // Single float for result
    auto tensor_output_optimized = manager.tensorT<float>(1); // Single float for result
    auto tensor_output_wave = manager.tensorT<float>(1);      // Single float for result

    // Create algorithm
    const auto dispatch_count = divide_and_round_up(size, k_thread_group_size);
    const auto dispatch_count_optimized = divide_and_round_up(size, k_thread_group_size * 2);

    auto naive_algo = manager.algorithm({tensor_input, tensor_output}, k_spirv_naive, kp::Workgroup({dispatch_count, 1, 1}));
    auto optimized_algo = manager.algorithm({tensor_input, tensor_output_optimized}, k_spirv_optimized, kp::Workgroup({dispatch_count_optimized, 1, 1}));
    auto wave_algo = manager.algorithm({tensor_input, tensor_output_wave}, k_spirv_wave, kp::Workgroup({dispatch_count, 1, 1}));

    manager.sequence()->eval<kp::OpSyncDevice>({tensor_input});

    // Measure execution time
    double naive_time = measure_time([&] { return manager.sequence(0, 2)->eval<kp::OpAlgoDispatch>(naive_algo)->getTimestamps(); },
                                     [&] { manager.sequence()->eval<kp::OpClear>({tensor_output}); });
    double wave_time = measure_time([&] { return manager.sequence(0, 2)->eval<kp::OpAlgoDispatch>(wave_algo)->getTimestamps(); },
                                    [&] { manager.sequence()->eval<kp::OpClear>({tensor_output_wave}); });
    double optimized_time = measure_time([&] { return manager.sequence(0, 2)->eval<kp::OpAlgoDispatch>(optimized_algo)->getTimestamps(); },
                                         [&] { manager.sequence()->eval<kp::OpClear>({tensor_output_optimized}); });

    // Sync the data from GPU to CPU
    manager.sequence()
        ->record<kp::OpSyncLocal>({tensor_output})
        ->record<kp::OpSyncLocal>({tensor_output_optimized})
        ->record<kp::OpSyncLocal>({tensor_output_wave})
        ->eval();

    // Get result and verify
    float naive_sum = tensor_output->data()[0];
    float optimized_sum = tensor_output_optimized->data()[0];
    float wave_sum = tensor_output_wave->data()[0];

    float naive_relative_error = std::abs(cpu_sum - naive_sum) / cpu_sum;
    float optimized_relative_error = std::abs(cpu_sum - optimized_sum) / cpu_sum;
    float wave_relative_error = std::abs(cpu_sum - wave_sum) / cpu_sum;

    fmt::println("CPU sum: {}", cpu_sum);
    fmt::println("Naive sum: {}", naive_sum);
    fmt::println("Naive Relative error: {}", naive_relative_error);
    fmt::println("Optimized sum: {}", optimized_sum);
    fmt::println("Optimized Relative error: {}", optimized_relative_error);
    fmt::println("Wave sum: {}", wave_sum);
    fmt::println("Wave relative error: {}", wave_relative_error);

    fmt::println("Naive execution time: {:.4f} ms", naive_time);
    fmt::println("Optimized execution time: {:.4f} ms", optimized_time);
    fmt::println("Wave execution time: {:.4f} ms", wave_time);
    // system("pause");

    return 0;
}