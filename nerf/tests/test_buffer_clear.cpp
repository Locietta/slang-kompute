// nerf/tests/test_buffer_clear.cpp
#include <kompute/Kompute.hpp>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <random>
#include "../OpClear.h"

#define TEST_ASSERT(condition, message)                                                         \
    do {                                                                                        \
        if (!(condition)) {                                                                     \
            fmt::print(stderr, "Assertion failed: {} at {}:{}\n", message, __FILE__, __LINE__); \
            return 1;                                                                           \
        }                                                                                       \
    } while (0)

int main() {
    try {
        kp::Manager mgr;
        fmt::print("Initialized Kompute manager with device: {}\n",
                   std::string_view(mgr.getDeviceProperties().deviceName));

        /// Test 1: clear buffer with OpClear

        // generate random data for the buffer
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0f, 1.0f);

        const size_t buffer_size = 1024;
        std::vector<float> initial_data(buffer_size);
        for (size_t i = 0; i < buffer_size; ++i) {
            initial_data[i] = dis(gen);
        }

        fmt::println("Initial data: {}", initial_data);

        auto tensor = mgr.tensorT(initial_data);

        mgr.sequence()->record<kp::OpClear>({tensor}, 1.0f)->record<kp::OpSyncLocal>({tensor})->eval();

        // check if all values in the buffer are now 1.0f
        auto result = tensor->vector();
        fmt::println("Result data: {}", result);
        for (size_t i = 0; i < buffer_size; ++i) {
            TEST_ASSERT(result[i] == 1.0f, "Value at index " + std::to_string(i) + " is not 1.0f");
        }

        /// Test 2: clear multiple tensors at once
        // tests/test_buffer_clear.cpp
        const size_t num_tensors = 4;
        std::vector<std::shared_ptr<kp::Memory>> tensors;
        for (size_t i = 0; i < num_tensors; ++i) {
            std::vector<float> initial_data(buffer_size);
            for (size_t j = 0; j < buffer_size; ++j) {
                initial_data[j] = dis(gen);
            }
            tensors.push_back(mgr.tensorT(initial_data));
        }
        mgr.sequence()->record<kp::OpClear>(tensors, 2.0f)->record<kp::OpSyncLocal>(tensors)->eval();
        for (size_t i = 0; i < num_tensors; ++i) {
            auto result = std::dynamic_pointer_cast<kp::TensorT<float>>(tensors[i])->vector();
            for (size_t j = 0; j < buffer_size; ++j) {
                TEST_ASSERT(result[j] == 2.0f, "Value at index " + std::to_string(j) + " is not 2.0f");
            }
        }

        /// Test 3: clear multiple tensors with different sizes, record one by one
        const size_t num_tensors_diff_sizes = 3;
        std::vector<std::shared_ptr<kp::Memory>> tensors_diff_sizes;
        std::vector<size_t> sizes = {16, 32, 64};
        for (size_t i = 0; i < num_tensors_diff_sizes; ++i) {
            std::vector<float> initial_data(sizes[i]);
            for (size_t j = 0; j < sizes[i]; ++j) {
                initial_data[j] = dis(gen);
            }
            tensors_diff_sizes.push_back(mgr.tensorT(initial_data));
        }
        auto seq = mgr.sequence();
        for (size_t i = 0; i < num_tensors_diff_sizes; ++i) {
            seq->record<kp::OpClear>({tensors_diff_sizes[i]}, 3.0f);
        }
        seq->record<kp::OpSyncLocal>(tensors_diff_sizes)->eval();
        for (size_t i = 0; i < num_tensors_diff_sizes; ++i) {
            auto result = std::dynamic_pointer_cast<kp::TensorT<float>>(tensors_diff_sizes[i])->vector();
            for (size_t j = 0; j < sizes[i]; ++j) {
                TEST_ASSERT(result[j] == 3.0f, "Value at index " + std::to_string(j) + " is not 3.0f");
            }
        }

    } catch (const std::exception &e) {
        fmt::print(stderr, "Exception caught: {}\n", e.what());
        return 1;
    }
}