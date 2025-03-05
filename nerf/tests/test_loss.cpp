// nerf/tests/test_loss.cpp
#include <kompute/Kompute.hpp>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <random>
#include <cmath>
#include "../loss.h"

#define TEST_ASSERT(condition, message)                                                         \
    do {                                                                                        \
        if (!(condition)) {                                                                     \
            fmt::print(stderr, "Assertion failed: {} at {}:{}\n", message, __FILE__, __LINE__); \
            return 1;                                                                           \
        }                                                                                       \
    } while (0)

bool float_equals(float a, float b, float epsilon = 1e-5f) {
    return std::abs(a - b) / a < epsilon;
}

int main() {
    try {
        kp::Manager mgr(0, {}, { "VK_EXT_shader_atomic_float" });
        fmt::print("Initialized Kompute manager with device: {}\n",
                   std::string_view(mgr.getDeviceProperties().deviceName));

        // Create MSE loss
        nerf::MSELoss loss(mgr);

        /// Test 1: Simple MSE loss calculation
        // Create two tensors with exact values
        std::vector<float> predicted_data = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> target_data = {1.5f, 2.5f, 3.5f, 4.5f};

        auto predicted = mgr.tensorT(predicted_data);
        auto target = mgr.tensorT(target_data);

        // Calculate expected loss manually
        float expected_loss = 0.0f;
        for (size_t i = 0; i < predicted_data.size(); i++) {
            float diff = predicted_data[i] - target_data[i];
            expected_loss += diff * diff;
        }
        expected_loss /= predicted_data.size();

        // Compute loss
        float computed_loss = loss.compute(predicted, target);
        fmt::println("Expected MSE loss: {}", expected_loss);
        fmt::println("Computed MSE loss: {}", computed_loss);
        TEST_ASSERT(float_equals(computed_loss, expected_loss),
                    "Computed loss doesn't match expected value");

        /// Test 2: MSE loss backward pass
        // Create gradient output tensor
        auto grad_output = mgr.tensorT<float>(predicted_data.size());

        // Call backward
        loss.backward(predicted, target, grad_output);

        // Sync gradient to CPU
        mgr.sequence()->record<kp::OpSyncLocal>({grad_output})->eval();

        // Calculate expected gradients manually
        std::vector<float> expected_grads(predicted_data.size());
        for (size_t i = 0; i < predicted_data.size(); i++) {
            expected_grads[i] = 2.0f * (predicted_data[i] - target_data[i]) / predicted_data.size();
        }

        // Check gradients
        auto grad_data = grad_output->vector();
        fmt::println("Expected gradients: {}", expected_grads);
        fmt::println("Computed gradients: {}", grad_data);

        for (size_t i = 0; i < predicted_data.size(); i++) {
            TEST_ASSERT(float_equals(grad_data[i], expected_grads[i]),
                        "Gradient at index " + std::to_string(i) + " doesn't match expected value");
        }

        /// Test 3: MSE loss with larger random tensors
        const size_t large_size = 1024 * 64;

        // Generate random data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-10.0f, 10.0f);

        std::vector<float> large_predicted(large_size);
        std::vector<float> large_target(large_size);
        for (size_t i = 0; i < large_size; i++) {
            large_predicted[i] = dis(gen);
            large_target[i] = dis(gen);
        }

        // fmt::println("large_predicted: {}", large_predicted);
        // fmt::println("large_target: {}", large_target);

        auto large_pred_tensor = mgr.tensorT(large_predicted);
        auto large_target_tensor = mgr.tensorT(large_target);

        // Calculate expected loss manually
        float large_expected_loss = 0.0f;
        for (size_t i = 0; i < large_size; i++) {
            float diff = large_predicted[i] - large_target[i];
            large_expected_loss += (diff * diff);
        }
        large_expected_loss /= large_size;

        // Compute loss
        float large_computed_loss = loss.compute(large_pred_tensor, large_target_tensor);
        fmt::println("Large tensor - Expected loss: {}", large_expected_loss);
        fmt::println("Large tensor - Computed loss: {}", large_computed_loss);
        TEST_ASSERT(float_equals(large_computed_loss, large_expected_loss),
                    "Large tensor loss doesn't match expected value");

        // Test gradient for large tensors
        auto large_grad = mgr.tensorT<float>(large_size);
        loss.backward(large_pred_tensor, large_target_tensor, large_grad);

        // Sync and check a few random gradients
        mgr.sequence()->record<kp::OpSyncLocal>({large_grad})->eval();
        auto large_grad_data = large_grad->vector();

        for (size_t i = 0; i < 10; i++) {
            size_t idx = gen() % large_size;
            float expected_grad = 2.0f * (large_predicted[idx] - large_target[idx]) / large_size;
            TEST_ASSERT(float_equals(large_grad_data[idx], expected_grad, 1e-5f),
                        "Large tensor gradient at random index doesn't match expected value");
        }

        fmt::println("All MSE loss tests passed!");
        return 0;
    } catch (const std::exception &e) {
        fmt::print(stderr, "Exception caught: {}\n", e.what());
        return 1;
    }
}