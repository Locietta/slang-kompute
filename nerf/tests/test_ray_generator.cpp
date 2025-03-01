// nerf/tests/test_ray_generator.cpp
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <fmt/core.h>
#include <kompute/Kompute.hpp>
#include "../ray_generator.h"

// Simple test assertion macro
#define TEST_ASSERT(condition, message)                                                         \
    do {                                                                                        \
        if (!(condition)) {                                                                     \
            fmt::print(stderr, "Assertion failed: {} at {}:{}\n", message, __FILE__, __LINE__); \
            return 1;                                                                           \
        }                                                                                       \
    } while (0)

int main() {
    try {
        // Initialize Kompute
        kp::Manager mgr;
        fmt::print("Initialized Kompute manager with device: {}\n", std::string_view(mgr.getDeviceProperties().deviceName));

        // Set up test parameters - use a small size for easier debugging
        uint32_t width = 3;
        uint32_t height = 3;
        float focal = 1000.0f; // Use a large focal length to minimize distortion

        // Create intrinsic matrix
        glm::mat3 intrinsic(
            focal, 0.0f, width / 2.0f,
            0.0f, focal, height / 2.0f,
            0.0f, 0.0f, 1.0f);

        fmt::print("Intrinsic matrix:\n");
        fmt::print("{} {} {}\n", intrinsic[0][0], intrinsic[0][1], intrinsic[0][2]);
        fmt::print("{} {} {}\n", intrinsic[1][0], intrinsic[1][1], intrinsic[1][2]);
        fmt::print("{} {} {}\n", intrinsic[2][0], intrinsic[2][1], intrinsic[2][2]);

        // Create camera-to-world matrix (identity for simplicity)
        glm::mat4 c2w = glm::mat4(1.0f);

        fmt::print("Camera-to-world matrix:\n");
        for (int i = 0; i < 4; i++) {
            fmt::print("{} {} {} {}\n", c2w[0][i], c2w[1][i], c2w[2][i], c2w[3][i]);
        }

        // Create ray generator
        nerf::RayGenerator ray_generator(mgr, width, height, intrinsic);

        // Generate rays
        ray_generator.generate_rays(c2w);

        // Get ray data
        auto ray_origins = ray_generator.get_ray_origins_sync();
        auto ray_directions = ray_generator.get_ray_directions_sync();

        // Print all ray directions for debugging
        fmt::print("Ray directions:\n");
        for (uint32_t y = 0; y < height; y++) {
            for (uint32_t x = 0; x < width; x++) {
                uint32_t idx = y * width + x;
                float dx = ray_directions->data()[idx * 3 + 0];
                float dy = ray_directions->data()[idx * 3 + 1];
                float dz = ray_directions->data()[idx * 3 + 2];

                fmt::print("Pixel ({}, {}): direction ({}, {}, {})\n",
                           x, y, dx, dy, dz);
            }
        }

        // Verify ray origins are all at the camera center (0,0,0)
        for (uint32_t i = 0; i < width * height; i++) {
            float ox = ray_origins->data()[i * 3 + 0];
            float oy = ray_origins->data()[i * 3 + 1];
            float oz = ray_origins->data()[i * 3 + 2];

            TEST_ASSERT(std::abs(ox) < 0.001f,
                        fmt::format("Ray origin x should be 0, got {}", ox));
            TEST_ASSERT(std::abs(oy) < 0.001f,
                        fmt::format("Ray origin y should be 0, got {}", oy));
            TEST_ASSERT(std::abs(oz) < 0.001f,
                        fmt::format("Ray origin z should be 0, got {}", oz));
        }

        // Test ray direction for the center pixel
        uint32_t center_idx = (height / 2) * width + (width / 2);
        float center_dir_x = ray_directions->data()[center_idx * 3 + 0];
        float center_dir_y = ray_directions->data()[center_idx * 3 + 1];
        float center_dir_z = ray_directions->data()[center_idx * 3 + 2];

        fmt::print("Center ray direction: ({}, {}, {})\n",
                   center_dir_x, center_dir_y, center_dir_z);

        // Center ray should point straight ahead (0,0,-1) when normalized
        TEST_ASSERT(std::abs(center_dir_x) < 0.001f,
                    fmt::format("Center ray x should be close to 0, got {}", center_dir_x));
        TEST_ASSERT(std::abs(center_dir_y) < 0.001f,
                    fmt::format("Center ray y should be close to 0, got {}", center_dir_y));
        TEST_ASSERT(center_dir_z < -0.999f,
                    fmt::format("Center ray z should be close to -1, got {}", center_dir_z));

        fmt::print("Ray generator tests passed successfully!\n");
        return 0;
    } catch (const std::exception &e) {
        fmt::print(stderr, "Test failed with exception: {}\n", e.what());
        return 1;
    }
}