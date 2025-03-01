// nerf/tests/test_ray_sampler.cpp
#include <glm/glm.hpp>
#include <fmt/core.h>
#include <kompute/Kompute.hpp>
#include "../ray_generator.h"
#include "../ray_sampler.h"

// Simple test assertion macro
#define TEST_ASSERT(condition, message)                                                         \
    do {                                                                                        \
        if (!(condition)) {                                                                     \
            fmt::print(stderr, "Assertion failed: {} at {}:{}\n", message, __FILE__, __LINE__); \
            return 1;                                                                           \
        }                                                                                       \
    } while (0)

// Template specialization for any glm::vec<N, T> type
template<int N, typename T>
struct fmt::formatter<glm::vec<N, T>> {
    // Parse format specifications
    constexpr auto parse(format_parse_context &ctx) {
        // Just use default formatting options
        return ctx.begin();
    }

    // Format the vector
    template<typename FormatContext>
    auto format(const glm::vec<N, T> &vec, FormatContext &ctx) const {
        fmt::format_to(ctx.out(), "(");

        for (int i = 0; i < N; ++i) {
            fmt::format_to(ctx.out(), "{}", vec[i]);
            if (i < N - 1) {
                fmt::format_to(ctx.out(), ", ");
            }
        }

        return fmt::format_to(ctx.out(), ")");
    }
};

int main() {
    try {
        // Initialize Kompute
        kp::Manager mgr;
        fmt::print("Initialized Kompute manager with device: {}\n", std::string_view(mgr.getDeviceProperties().deviceName));

        // Set up test parameters
        uint32_t width = 4;
        uint32_t height = 4;
        float focal = 50.0f;
        uint32_t num_rays = width * height;
        uint32_t n_samples = 16;
        float near = 2.0f;
        float far = 6.0f;

        // Create intrinsic matrix
        glm::mat3 intrinsic(
            focal, 0.0f, width / 2.0f,
            0.0f, focal, height / 2.0f,
            0.0f, 0.0f, 1.0f);

        // Create camera-to-world matrix (identity for simplicity)
        glm::mat4 c2w = glm::mat4(1.0f);

        // Create ray generator and generate rays
        nerf::RayGenerator ray_generator(mgr, width, height, intrinsic);
        ray_generator.generate_rays(c2w);
        auto ray_origins = ray_generator.get_ray_origins_sync();
        auto ray_directions = ray_generator.get_ray_directions_sync();

        // Create ray sampler
        nerf::RaySampler ray_sampler(mgr, ray_origins, ray_directions, n_samples, false, near, far);

        // Sample points along rays
        ray_sampler.sample_points();

        // Get samples
        auto sample_positions = ray_sampler.get_sample_positions_sync();
        auto sample_directions = ray_sampler.get_sample_directions_sync();
        auto sample_z_vals = ray_sampler.get_sample_z_vals_sync();

        // Test sample counts
        TEST_ASSERT(sample_positions->size() == num_rays * n_samples * 3, "Sample positions size is incorrect");
        TEST_ASSERT(sample_directions->size() == num_rays * n_samples * 3, "Sample directions size is incorrect");
        TEST_ASSERT(sample_z_vals->size() == num_rays * n_samples, "Z values size is incorrect");

        // Test sample z values (should be linearly spaced between near and far)
        for (uint32_t i = 0; i < num_rays; i++) {
            for (uint32_t j = 0; j < n_samples; j++) {
                float z = sample_z_vals->data()[i * n_samples + j];
                float expected_z = near + (far - near) * (float)j / (float)(n_samples - 1);

                // Allow small floating point error
                TEST_ASSERT(std::abs(z - expected_z) < 0.001f,
                            fmt::format("Z value at ray {} sample {} should be {}, got {}",
                                        i, j, expected_z, z));
            }
        }

        // Test sample positions (should be origin + direction * z)
        for (uint32_t i = 0; i < num_rays; i++) {
            glm::vec3 origin = {
                ray_origins->data()[i * 3 + 0],
                ray_origins->data()[i * 3 + 1],
                ray_origins->data()[i * 3 + 2]};

            glm::vec3 direction = {
                ray_directions->data()[i * 3 + 0],
                ray_directions->data()[i * 3 + 1],
                ray_directions->data()[i * 3 + 2]};

            for (uint32_t j = 0; j < n_samples; j++) {
                float z = sample_z_vals->data()[i * n_samples + j];

                glm::vec3 expected_pos = {
                    origin.x + direction.x * z,
                    origin.y + direction.y * z,
                    origin.z + direction.z * z};

                glm::vec3 actual_pos = {
                    sample_positions->data()[(i * n_samples + j) * 3 + 0],
                    sample_positions->data()[(i * n_samples + j) * 3 + 1],
                    sample_positions->data()[(i * n_samples + j) * 3 + 2]};

                fmt::println("Expected position: {}, Actual position: {}", expected_pos, actual_pos);

                // Check position matches expected
                TEST_ASSERT(std::abs(actual_pos.x - expected_pos.x) < 0.001f, "Position x mismatch");
                TEST_ASSERT(std::abs(actual_pos.y - expected_pos.y) < 0.001f, "Position y mismatch");
                TEST_ASSERT(std::abs(actual_pos.z - expected_pos.z) < 0.001f, "Position z mismatch");

                // Check direction was copied correctly
                TEST_ASSERT(std::abs(sample_directions->data()[(i * n_samples + j) * 3 + 0] - direction.x) < 0.001f,
                            "Direction x mismatch");
                TEST_ASSERT(std::abs(sample_directions->data()[(i * n_samples + j) * 3 + 1] - direction.y) < 0.001f,
                            "Direction y mismatch");
                TEST_ASSERT(std::abs(sample_directions->data()[(i * n_samples + j) * 3 + 2] - direction.z) < 0.001f,
                            "Direction z mismatch");
            }
        }

        // Now test with perturbation
        nerf::RaySampler ray_sampler2(mgr, ray_origins, ray_directions, n_samples, true, near, far);
        ray_sampler2.sample_points();
        auto perturbed_z_vals = ray_sampler2.get_sample_z_vals_sync();

        // Test that perturbed z values are different from unperturbed ones
        bool some_values_different = false;
        for (uint32_t i = 0; i < num_rays * n_samples; i++) {
            if (std::abs(perturbed_z_vals->data()[i] - sample_z_vals->data()[i]) > 0.001f) {
                some_values_different = true;
                break;
            }
        }
        TEST_ASSERT(some_values_different, "Perturbed z values should differ from unperturbed ones");

        // Test that perturbed z values are within the expected range
        for (uint32_t i = 0; i < num_rays * n_samples; i++) {
            float z = perturbed_z_vals->data()[i];
            TEST_ASSERT(z >= near && z <= far, "Perturbed z value should be in range [near, far]");
        }

        fmt::print("Ray sampler tests passed successfully!\n");
        return 0;
    } catch (const std::exception &e) {
        fmt::print(stderr, "Test failed with exception: {}\n", e.what());
        return 1;
    }
}