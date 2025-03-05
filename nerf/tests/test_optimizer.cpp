// nerf/tests/test_optimizer.cpp
#include <kompute/Kompute.hpp>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <random>
#include <cmath>
#include "../optimizer.h"

#define TEST_ASSERT(condition, message)                                                         \
    do {                                                                                        \
        if (!(condition)) {                                                                     \
            fmt::print(stderr, "Assertion failed: {} at {}:{}\n", message, __FILE__, __LINE__); \
            return 1;                                                                           \
        }                                                                                       \
    } while (0)

bool float_equals(float a, float b, float epsilon = 1e-5f) {
    return std::abs(a - b) < epsilon;
}

int main() {
    try {
        kp::Manager mgr;
        fmt::print("Initialized Kompute manager with device: {}\n",
                   std::string_view(mgr.getDeviceProperties().deviceName));

        /// Test 1: Basic Adam optimizer functionality
        // Create parameters and gradients
        std::vector<float> param_data = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> grad_data = {0.1f, 0.2f, 0.3f, 0.4f};

        auto param_tensor = mgr.tensorT(param_data);
        auto grad_tensor = mgr.tensorT(grad_data);

        // Create Adam optimizer with known parameters for easy testing
        nerf::AdamParams adam_params;
        adam_params.learning_rate = 0.1f;
        adam_params.beta1 = 0.9f;
        adam_params.beta2 = 0.999f;
        adam_params.epsilon = 1e-8f;
        
        nerf::Adam optimizer(mgr, adam_params);

        // Add parameter with gradient
        optimizer.add_parameter_with_gradient(param_tensor, grad_tensor);

        // Step the optimizer once
        optimizer.step();

        // Calculate expected values after one step
        std::vector<float> expected_params(param_data.size());
        std::vector<float> expected_m(param_data.size());
        std::vector<float> expected_v(param_data.size());
        
        for (size_t i = 0; i < param_data.size(); i++) {
            // First moment: m = beta1 * m + (1 - beta1) * g
            expected_m[i] = (1 - adam_params.beta1) * grad_data[i];
            
            // Second moment: v = beta2 * v + (1 - beta2) * g^2
            expected_v[i] = (1 - adam_params.beta2) * grad_data[i] * grad_data[i];
            
            // Bias correction
            float m_corrected = expected_m[i] / (1 - adam_params.beta1);
            float v_corrected = expected_v[i] / (1 - adam_params.beta2);
            
            // Update parameter
            expected_params[i] = param_data[i] - adam_params.learning_rate * m_corrected / 
                                (std::sqrt(v_corrected) + adam_params.epsilon);
        }

        // Sync updated params to CPU
        mgr.sequence()->record<kp::OpSyncLocal>({param_tensor})->eval();
        auto updated_params = param_tensor->vector();

        fmt::println("Initial parameters: {}", param_data);
        fmt::println("Expected parameters after step: {}", expected_params);
        fmt::println("Actual parameters after step: {}", updated_params);

        // Verify parameter updates
        for (size_t i = 0; i < param_data.size(); i++) {
            TEST_ASSERT(float_equals(updated_params[i], expected_params[i]),
                        "Parameter at index " + std::to_string(i) + " doesn't match expected value");
        }

        /// Test 2: Multiple steps
        // Step several more times and verify behavior
        for (int step = 0; step < 5; step++) {
            // Update gradients (simulating a new backward pass)
            std::vector<float> new_grads(param_data.size());
            for (size_t i = 0; i < param_data.size(); i++) {
                new_grads[i] = updated_params[i] * 0.1f; // Simple gradient rule for testing
            }
            
            // Upload new gradients
            mgr.sequence()->record<kp::OpSyncLocal>({grad_tensor})->eval();
            std::memcpy(grad_tensor->data(), new_grads.data(), new_grads.size() * sizeof(float));
            mgr.sequence()->record<kp::OpSyncDevice>({grad_tensor})->eval();
            
            // Step the optimizer
            optimizer.step();
            
            // Sync parameters back
            mgr.sequence()->record<kp::OpSyncLocal>({param_tensor})->eval();
            updated_params = param_tensor->vector();
            
            fmt::println("Parameters after step {}: {}", step + 2, updated_params);
        }

        /// Test 3: Multiple parameters
        // Create several parameter tensors with different sizes
        std::vector<size_t> sizes = {4, 8, 16};
        std::vector<std::shared_ptr<kp::TensorT<float>>> params;
        
        // Create new optimizer
        nerf::Adam multi_optimizer(mgr, adam_params);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1.0f, 1.0f);
        
        // Add parameters
        for (size_t size : sizes) {
            std::vector<float> data(size);
            for (size_t i = 0; i < size; i++) {
                data[i] = dis(gen);
            }
            auto tensor = mgr.tensorT(data);
            multi_optimizer.add_parameter(tensor);
            params.push_back(tensor);
        }
        
        // Verify all parameters are initialized
        TEST_ASSERT(params.size() == sizes.size(), "Not all parameters were added");
        
        // Step the optimizer with all parameters
        multi_optimizer.step();
        
        // Reset optimizer state
        multi_optimizer.reset();
        
        // Verify we can add more parameters after reset
        std::vector<float> extra_param = {0.1f, 0.2f, 0.3f};
        auto extra_tensor = mgr.tensorT(extra_param);
        multi_optimizer.add_parameter(extra_tensor);
        
        fmt::println("All optimizer tests passed!");
        return 0;
    } catch (const std::exception &e) {
        fmt::print(stderr, "Exception caught: {}\n", e.what());
        return 1;
    }
}