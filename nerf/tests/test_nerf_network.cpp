// nerf/tests/test_nerf_network.cpp
#include <glm/glm.hpp>
#include <fmt/core.h>
#include <kompute/Kompute.hpp>
#include "../nerf_network.h"
#include "../positional_encoder.h"

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
        fmt::print("Initialized Kompute manager with device: {}\n",
                   std::string_view(mgr.getDeviceProperties().deviceName));

        // Create a batch of test positions
        uint32_t batch_size = 10;
        std::vector<float> positions(batch_size * 3);
        for (size_t i = 0; i < positions.size(); i++) {
            positions[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f; // [-1, 1]
        }

        auto positions_tensor = mgr.tensorT(positions);

        // Create positional encoder for the input
        nerf::PositionalEncoder encoder(mgr, 10, true, true);

        // Encode positions
        encoder.encode(positions_tensor);
        auto encoded_positions = encoder.get_encoded();

        // Get the encoded dimension first
        uint32_t encoded_dim = encoder.get_output_dim();

        // Create a small NeRF network for testing
        nerf::MLPParams params;
        params.D = 4;                  // 4 layers
        params.W = 128;                // 128 neurons per layer
        params.skips = 2;              // Skip connection at layer 2
        params.use_viewdirs = false;   // No view dependency for simple test
        params.input_ch = encoded_dim; // 3D position input
        params.input_ch_views = 0;     // No view inputs
        params.output_ch = 4;          // RGBA output

        // Create network
        nerf::NerfNetwork network(mgr, params);

        // Forward pass through network
        network.forward(encoded_positions);

        // Get output
        auto output = network.get_output_sync();

        fmt::println("output size: {} Expected: {}", output->size(), batch_size * params.output_ch);

        // Check output shape
        TEST_ASSERT(output->size() == batch_size * params.output_ch,
                    "Output size does not match expected");

        // Print some outputs to verify they're reasonable
        fmt::print("Network outputs for first 3 samples:\n");
        for (uint32_t i = 0; i < 3; i++) {
            fmt::print("Sample {}: ", i);
            for (uint32_t j = 0; j < params.output_ch; j++) {
                fmt::print("{:.6f} ", output->data()[i * params.output_ch + j]);
            }
            fmt::print("\n");
        }

        // Test saving and loading weights
        std::string weights_file = "test_nerf_weights.bin";
        network.save_weights(weights_file);

        // Create a new network with the same parameters
        nerf::NerfNetwork network2(mgr, params);

        // Load the weights
        network2.load_weights(weights_file);

        // Forward pass through second network
        network2.forward(encoded_positions);
        auto output2 = network2.get_output_sync();

        // Check that outputs match
        for (uint32_t i = 0; i < output->size(); i++) {
            TEST_ASSERT(std::abs(output->data()[i] - output2->data()[i]) < 1e-5,
                        "Outputs don't match after loading weights");
        }

        // Test with viewdirs if supported
        if (params.use_viewdirs) {

            // Create view directions
            std::vector<float> directions(batch_size * 3);
            for (size_t i = 0; i < directions.size(); i++) {
                directions[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f; // [-1, 1]
            }

            auto directions_tensor = mgr.tensorT(directions);

            // Create view encoder
            nerf::PositionalEncoder view_encoder(mgr, nerf::EncoderType::DIRECTION);

            // Encode view directions
            view_encoder.encode(directions_tensor);
            auto encoded_views = view_encoder.get_encoded();

            uint32_t encoded_view_dim = view_encoder.get_output_dim();

            // Create a new network with viewdirs support
            nerf::MLPParams view_params = params;
            view_params.use_viewdirs = true;
            view_params.input_ch_views = encoded_view_dim;

            nerf::NerfNetwork view_network(mgr, view_params);

            // Forward pass with view directions
            view_network.forward(encoded_positions, encoded_views);

            // Get output
            auto view_output = view_network.get_output_sync();

            // Check output shape
            TEST_ASSERT(view_output->size() == batch_size * view_params.output_ch,
                        "View output size does not match expected");

            // Print some outputs to verify they're reasonable
            fmt::print("View network outputs for first 3 samples:\n");
            for (uint32_t i = 0; i < 3; i++) {
                fmt::print("Sample {}: ", i);
                for (uint32_t j = 0; j < view_params.output_ch; j++) {
                    fmt::print("{:.6f} ", view_output->data()[i * view_params.output_ch + j]);
                }
                fmt::print("\n");
            }
        }

        fmt::print("NeRF network tests passed successfully!\n");
        return 0;
    } catch (const std::exception &e) {
        fmt::print(stderr, "Test failed with exception: {}\n", e.what());
        return 1;
    }
}