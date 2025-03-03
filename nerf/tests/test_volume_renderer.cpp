// nerf/tests/test_volume_renderer.cpp
#include <glm/glm.hpp>
#include <fmt/core.h>
#include <kompute/Kompute.hpp>
#include "../volume_renderer.h"

// Simple test assertion macro
#define TEST_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            fmt::print(stderr, "Assertion failed: {} at {}:{}\n", message, __FILE__, __LINE__); \
            return 1; \
        } \
    } while (0)

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
        uint32_t n_samples = 4; // 4 samples per ray
        
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
        
        fmt::print("Volume rendering tests passed successfully!\n");
        return 0;
    } catch (const std::exception& e) {
        fmt::print(stderr, "Test failed with exception: {}\n", e.what());
        return 1;
    }
}