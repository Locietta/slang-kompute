// nerf/tests/test_volume_renderer.cpp
#include <glm/glm.hpp>
#include <fmt/core.h>
#include <kompute/Kompute.hpp>
#include "../volume_renderer.h"

// Simple test assertion macro
#define TEST_ASSERT(condition, message)                                                         \
    do {                                                                                        \
        if (!(condition)) {                                                                     \
            fmt::print(stderr, "Assertion failed: {} at {}:{}\n", message, __FILE__, __LINE__); \
            return 1;                                                                           \
        }                                                                                       \
    } while (0)

// Helper to check if values are close
bool float_equal(float a, float b, float tolerance = 1e-6f) {
    return std::abs(a - b) / a < tolerance;
}

int main() {
    try {
        // Initialize Kompute
        kp::Manager mgr;
        fmt::print("Initialized Kompute manager with device: {}\n",
                   std::string_view(mgr.getDeviceProperties().deviceName));

        // Create volume renderer with consolidated shader
        nerf::VolumeRenderingParams params;
        params.raw_noise_std = 0.0f; // No noise for deterministic testing
        params.white_background = true;

        nerf::VolumeRenderer renderer(mgr, params);

        // Create test data
        uint32_t batch_size = 2; // 2 rays
        uint32_t n_samples = 4;  // 4 samples per ray

        // Create sample positions along rays (z values)
        std::vector<float> z_vals = {
            1.0f, 2.0f, 3.0f, 4.0f, // Ray 1
            1.0f, 2.0f, 3.0f, 4.0f  // Ray 2
        };

        // Create ray directions (unit vectors)
        std::vector<float> rays_d = {
            0.0f, 0.0f, 1.0f, // Ray 1
            0.0f, 0.0f, 1.0f  // Ray 2
        };

        // Create raw network output (RGBA)
        // Ray 1: Opaque red point at sample 1, semi-transparent blue at sample 3
        // Ray 2: Transparent, then opaque green at sample 2
        std::vector<float> raw = {
            // Ray 1
            1.0f, 0.0f, 0.0f, 5.0f, // Sample 1: Red, high density
            0.0f, 0.0f, 0.0f, 0.0f, // Sample 2: Transparent
            0.0f, 0.0f, 1.0f, 1.0f, // Sample 3: Blue, medium density
            0.0f, 0.0f, 0.0f, 0.0f, // Sample 4: Transparent

            // Ray 2
            0.0f, 0.0f, 0.0f, 0.0f, // Sample 1: Transparent
            0.0f, 1.0f, 0.0f, 5.0f, // Sample 2: Green, high density
            0.0f, 0.0f, 0.0f, 0.0f, // Sample 3: Transparent
            0.0f, 0.0f, 0.0f, 0.0f  // Sample 4: Transparent
        };

        // Create tensors
        auto z_vals_tensor = mgr.tensorT(z_vals);
        auto rays_d_tensor = mgr.tensorT(rays_d);
        auto raw_tensor = mgr.tensorT(raw);

        mgr.sequence()->eval<kp::OpSyncDevice>(
            {
                z_vals_tensor,
                rays_d_tensor,
                raw_tensor,
            });

        //======= FORWARD PASS TEST =======//

        // Render
        renderer.render(raw_tensor, z_vals_tensor, rays_d_tensor);

        // Synchronize outputs
        renderer.sync_outputs();

        // Get outputs
        auto rgb = renderer.get_rgb();
        auto disp = renderer.get_disparity();
        auto acc = renderer.get_accumulated();
        auto depth = renderer.get_depth();
        auto weights = renderer.get_weights();

        // Print results
        fmt::print("Rendering results with consolidated shader:\n");

        fmt::print("RGB values:\n");
        for (uint32_t i = 0; i < batch_size; i++) {
            fmt::print("  Ray {}: [{:.4f}, {:.4f}, {:.4f}]\n",
                       i,
                       rgb->data()[i * 3 + 0],
                       rgb->data()[i * 3 + 1],
                       rgb->data()[i * 3 + 2]);
        }

        fmt::print("Disparity values:\n");
        for (uint32_t i = 0; i < batch_size; i++) {
            fmt::print("  Ray {}: {:.4f}\n", i, disp->data()[i]);
        }

        fmt::print("Accumulated opacity:\n");
        for (uint32_t i = 0; i < batch_size; i++) {
            fmt::print("  Ray {}: {:.4f}\n", i, acc->data()[i]);
        }

        fmt::print("Depth values:\n");
        for (uint32_t i = 0; i < batch_size; i++) {
            fmt::print("  Ray {}: {:.4f}\n", i, depth->data()[i]);
        }

        fmt::print("Sample weights (first ray):\n");
        for (uint32_t i = 0; i < n_samples; i++) {
            fmt::print("  Sample {}: {:.4f}\n", i, weights->data()[i]);
        }

        // Verify expected results
        // For Ray 1: Mostly red with some blue contribution
        // For Ray 2: Mostly green

        // Verify that outputs are in reasonable ranges
        for (uint32_t i = 0; i < batch_size * 3; i++) {
            TEST_ASSERT(rgb->data()[i] >= 0.0f && rgb->data()[i] <= 1.0f,
                        "RGB values should be in range [0, 1]");
        }

        for (uint32_t i = 0; i < batch_size; i++) {
            TEST_ASSERT(acc->data()[i] >= 0.0f && acc->data()[i] <= 1.0f,
                        "Accumulated opacity should be in range [0, 1]");

            TEST_ASSERT(depth->data()[i] >= z_vals[0] && depth->data()[i] <= z_vals[n_samples - 1],
                        "Depth values should be in range of z_vals");
        }

        //======= BACKWARD PASS TEST =======//

        fmt::print("\n===== Testing Backward Pass =====\n");

        // Create gradients with respect to RGB outputs
        // Gradient for Ray 1: [1.0, 0.0, 0.0] - Increase red channel
        // Gradient for Ray 2: [0.0, 1.0, 0.0] - Increase green channel
        std::vector<float> grad_rgb_data = {
            1.0f, 0.0f, 0.0f, // Ray 1: Increase red
            0.0f, 1.0f, 0.0f  // Ray 2: Increase green
        };

        // Optional gradients for disparity and accumulation
        std::vector<float> grad_disp_data = {0.1f, 0.1f}; // Small gradient for disparity
        std::vector<float> grad_acc_data = {0.0f, 0.0f};  // No gradient for accumulation

        // Create gradient tensors
        auto grad_rgb_tensor = mgr.tensorT(grad_rgb_data);
        auto grad_disp_tensor = mgr.tensorT(grad_disp_data);
        auto grad_acc_tensor = mgr.tensorT(grad_acc_data);

        mgr.sequence()->eval<kp::OpSyncDevice>(
            {
                grad_rgb_tensor,
                grad_disp_tensor,
                grad_acc_tensor,
            });

        // Run backward pass
        renderer.backward(grad_rgb_tensor, grad_disp_tensor, grad_acc_tensor);

        // Get gradients with respect to raw network output
        auto grad_raw = renderer.get_grad_raw();

        // Sync gradients to CPU
        mgr.sequence()->record<kp::OpSyncLocal>({grad_raw})->eval();

        // Print gradients
        fmt::print("Gradients with respect to raw network output:\n");

        for (uint32_t ray = 0; ray < batch_size; ray++) {
            fmt::print("Ray {}:\n", ray);
            for (uint32_t s = 0; s < n_samples; s++) {
                uint32_t base_idx = (ray * n_samples + s) * 4;
                fmt::print("  Sample {}: [{:.6f}, {:.6f}, {:.6f}, {:.6f}] (r,g,b,sigma)\n",
                           s,
                           grad_raw->data()[base_idx + 0],
                           grad_raw->data()[base_idx + 1],
                           grad_raw->data()[base_idx + 2],
                           grad_raw->data()[base_idx + 3]);
            }
        }

        //======= Test specific gradient patterns =======//

        // 1. Test that red gradients propagate correctly to Ray 1
        // - The red sample (sample 0) should have positive gradient for red channel
        // - Other samples should have smaller or zero gradients
        TEST_ASSERT(grad_raw->data()[0] > 0.0f,
                    "Red gradient for Ray 1, Sample 0 should be positive");

        // 2. Test that green gradients propagate correctly to Ray 2
        // - The green sample (sample 1) should have positive gradient for green channel
        uint32_t ray2_sample1_green_idx = (1 * n_samples + 1) * 4 + 1;
        TEST_ASSERT(grad_raw->data()[ray2_sample1_green_idx] > 0.0f,
                    "Green gradient for Ray 2, Sample 1 should be positive");

        // 3. Test that density gradients are non-zero for samples that contribute to final color
        uint32_t ray1_sample0_density_idx = (0 * n_samples + 0) * 4 + 3;
        uint32_t ray2_sample1_density_idx = (1 * n_samples + 1) * 4 + 3;
        TEST_ASSERT(grad_raw->data()[ray1_sample0_density_idx] != 0.0f,
                    "Density gradient for Ray 1, Sample 0 should be non-zero");
        TEST_ASSERT(grad_raw->data()[ray2_sample1_density_idx] != 0.0f,
                    "Density gradient for Ray 2, Sample 1 should be non-zero");

        //======= Finite difference verification =======//
        fmt::print("\n===== Finite Difference Verification =====\n");
        fmt::print("This verifies gradient correctness by comparing analytical gradients\n");
        fmt::print("with numerical gradients calculated via small perturbations.\n");

        // Choose a component to verify
        uint32_t test_ray = 0;
        uint32_t test_sample = 0;
        uint32_t test_channel = 0; // Red channel
        float epsilon = -1e-4f;

        // Index into the raw tensor for the chosen component
        uint32_t test_idx = (test_ray * n_samples + test_sample) * 4 + test_channel;

        // Store original RGB for this ray
        std::vector<float> original_rgb_values = {
            rgb->data()[test_ray * 3 + 0],
            rgb->data()[test_ray * 3 + 1],
            rgb->data()[test_ray * 3 + 2]};

        fmt::print("Original RGB for Ray {}: [{:.6f}, {:.6f}, {:.6f}]\n",
                   test_ray,
                   original_rgb_values[0],
                   original_rgb_values[1],
                   original_rgb_values[2]);

        // Create a perturbed raw tensor (add epsilon to the chosen component)
        std::vector<float> perturbed_raw = raw;
        perturbed_raw[test_idx] += epsilon;

        fmt::print("Perturbing Ray {}, Sample {}, Channel {} by epsilon={:.6f}\n",
                   test_ray, test_sample, test_channel, epsilon);

        fmt::print("original raw value: {:.6f}, Perturbed raw value: {:.6f}\n",
                   raw[test_idx], perturbed_raw[test_idx]);

        auto perturbed_raw_tensor = mgr.tensorT(perturbed_raw);

        mgr.sequence()->eval<kp::OpSyncDevice>({perturbed_raw_tensor});

        // Run forward pass with perturbed input (this doesn't change original input)
        renderer.render(perturbed_raw_tensor, z_vals_tensor, rays_d_tensor);
        renderer.sync_outputs();

        // Get perturbed RGB
        auto perturbed_rgb = renderer.get_rgb();

        // Get perturbed RGB values
        std::vector<float> perturbed_rgb_values = {
            perturbed_rgb->data()[test_ray * 3 + 0],
            perturbed_rgb->data()[test_ray * 3 + 1],
            perturbed_rgb->data()[test_ray * 3 + 2]};

        fmt::print("Perturbed RGB for Ray {}: [{:.6f}, {:.6f}, {:.6f}]\n",
                   test_ray,
                   perturbed_rgb_values[0],
                   perturbed_rgb_values[1],
                   perturbed_rgb_values[2]);

        // Calculate difference in output
        std::vector<float> output_diff = {
            perturbed_rgb_values[0] - original_rgb_values[0],
            perturbed_rgb_values[1] - original_rgb_values[1],
            perturbed_rgb_values[2] - original_rgb_values[2]};

        fmt::print("Output difference: [{:.6f}, {:.6f}, {:.6f}]\n",
                   output_diff[0], output_diff[1], output_diff[2]);

        // Calculate numerical gradient (change in output / change in input)
        std::vector<float> numerical_gradients = {
            output_diff[0] / epsilon,
            output_diff[1] / epsilon,
            output_diff[2] / epsilon};

        fmt::print("Numerical gradients (dRGB/dInput): [{:.6f}, {:.6f}, {:.6f}]\n",
                   numerical_gradients[0], numerical_gradients[1], numerical_gradients[2]);

        // Get analytical gradient from backward pass
        // Our grad_raw holds dL/dInput, where L is the loss function
        // For our test, L = RGB[test_channel], so dL/dInput = dRGB[test_channel]/dInput
        float analytical_gradient = grad_raw->data()[test_idx];

        fmt::print("Analytical gradient from backward pass: {:.6f}\n", analytical_gradient);

        // Verify that the analytical gradient matches the numerical gradient for the relevant channel
        TEST_ASSERT(float_equal(numerical_gradients[test_channel], analytical_gradient, 1e-3f),
                    "Analytical gradient should match numerical gradient for the relevant channel");

        fmt::print("Volume rendering forward and backward tests passed successfully!\n");
        fmt::print("Note: Backward pass calculates gradients but doesn't update inputs.\n");
        fmt::print("In a training loop, these gradients would be used by the optimizer\n");
        fmt::print("to update network parameters.\n");
        return 0;
    } catch (const std::exception &e) {
        fmt::print(stderr, "Test failed with exception: {}\n", e.what());
        return 1;
    }
}