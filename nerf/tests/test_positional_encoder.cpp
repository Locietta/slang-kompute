// nerf/tests/test_positional_encoder.cpp
#include <glm/glm.hpp>
#include <fmt/core.h>
#include <kompute/Kompute.hpp>
#include "../positional_encoder.h"

// Simple test assertion macro
#define TEST_ASSERT(condition, message)                                                         \
    do {                                                                                        \
        if (!(condition)) {                                                                     \
            fmt::print(stderr, "Assertion failed: {} at {}:{}\n", message, __FILE__, __LINE__); \
            return 1;                                                                           \
        }                                                                                       \
    } while (0)

// Reference CPU implementation of positional encoding
std::vector<float> compute_reference_encoding(const std::vector<float> &inputs,
                                              const std::vector<float> &freqs,
                                              bool include_input) {
    std::vector<float> result;
    size_t num_vectors = inputs.size() / 3;
    size_t output_dim = (include_input ? 3 : 0) + 2 * 3 * freqs.size();
    result.reserve(num_vectors * output_dim);

    for (size_t vec_idx = 0; vec_idx < num_vectors; vec_idx++) {
        // Extract input vector
        float x = inputs[vec_idx * 3 + 0];
        float y = inputs[vec_idx * 3 + 1];
        float z = inputs[vec_idx * 3 + 2];

        // Include original input if specified
        if (include_input) {
            result.push_back(x);
            result.push_back(y);
            result.push_back(z);
        }

        // For each frequency band
        for (size_t freq_idx = 0; freq_idx < freqs.size(); freq_idx++) {
            float freq = freqs[freq_idx];

            // For each dimension, add sin and cos
            result.push_back(std::sin(x * freq));
            result.push_back(std::cos(x * freq));

            result.push_back(std::sin(y * freq));
            result.push_back(std::cos(y * freq));

            result.push_back(std::sin(z * freq));
            result.push_back(std::cos(z * freq));
        }
    }

    return result;
}

int main() {
    try {
        // Initialize Kompute
        kp::Manager mgr;
        fmt::print("Initialized Kompute manager with device: {}\n",
                   std::string_view(mgr.getDeviceProperties().deviceName));

        // Test parameters
        uint32_t num_vectors = 10;

        // Create random input data
        std::vector<float> input_data(num_vectors * 3);
        for (size_t i = 0; i < input_data.size(); i++) {
            input_data[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f; // [-1, 1]
        }

        // Create input tensor
        auto input_tensor = mgr.tensorT(input_data);

        // Test 1: Position encoder with custom parameters
        uint32_t num_freqs = 10;
        bool include_input = true;
        bool log_sampling = true;

        nerf::PositionalEncoder encoder(mgr, num_freqs, include_input, log_sampling);

        // Verify output dimension
        uint32_t expected_output_dim = include_input ? 3 : 0;
        expected_output_dim += 2 * 3 * num_freqs;
        TEST_ASSERT(encoder.get_output_dim() == expected_output_dim,
                    "Output dimension mismatch");

        // Encode input
        encoder.encode(input_tensor);

        // Get encoded output
        auto encoded_tensor = encoder.get_encoded();

        // Verify output size
        TEST_ASSERT(encoded_tensor->size() == num_vectors * expected_output_dim,
                    "Encoded output size mismatch");

        // Compute reference frequency bands
        std::vector<float> freqs(num_freqs);
        for (uint32_t i = 0; i < num_freqs; i++) {
            if (log_sampling) {
                freqs[i] = std::pow(2.0f, static_cast<float>(i));
            } else {
                float max_freq_log2 = static_cast<float>(num_freqs - 1);
                freqs[i] = std::pow(2.0f,
                                    static_cast<float>(i) / static_cast<float>(num_freqs - 1) * max_freq_log2);
            }
        }

        // Compute reference encoding
        std::vector<float> expected_output = compute_reference_encoding(input_data, freqs, include_input);

        // Compare GPU output with CPU reference
        TEST_ASSERT(expected_output.size() == encoded_tensor->size(),
                    "Reference and GPU output sizes don't match");

        for (size_t i = 0; i < expected_output.size(); i++) {
            float gpu_val = encoded_tensor->data()[i];
            float cpu_val = expected_output[i];

            if (std::abs(gpu_val - cpu_val) > 1e-4) {
                fmt::print("Mismatch at index {}: GPU={}, CPU={}, diff={}\n",
                           i, gpu_val, cpu_val, std::abs(gpu_val - cpu_val));
                TEST_ASSERT(false, "Output value mismatch");
            }
        }

        // Test 2: Using predefined encoder type (Position)
        nerf::PositionalEncoder pos_encoder(mgr, nerf::EncoderType::POSITION);

        // Encode with position encoder
        pos_encoder.encode(input_tensor);
        auto pos_encoded = pos_encoder.get_encoded();

        // Verify position encoder output size
        expected_output_dim = 3 + 2 * 3 * 10; // include_input + 2 * dims * num_freqs
        TEST_ASSERT(pos_encoded->size() == num_vectors * expected_output_dim,
                    "Position encoder output size mismatch");

        // Test 3: Using predefined encoder type (Direction)
        nerf::PositionalEncoder dir_encoder(mgr, nerf::EncoderType::DIRECTION);

        // Encode with direction encoder
        dir_encoder.encode(input_tensor);
        auto dir_encoded = dir_encoder.get_encoded();

        // Verify direction encoder output size
        expected_output_dim = 3 + 2 * 3 * 4; // include_input + 2 * dims * num_freqs
        TEST_ASSERT(dir_encoded->size() == num_vectors * expected_output_dim,
                    "Direction encoder output size mismatch");

        // Test 4: Different batch size
        uint32_t num_vectors2 = 5;
        std::vector<float> input_data2(num_vectors2 * 3);
        for (size_t i = 0; i < input_data2.size(); i++) {
            input_data2[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
        }

        auto input_tensor2 = mgr.tensorT(input_data2);

        // Encode with the same position encoder but different batch size
        pos_encoder.encode(input_tensor2);
        auto encoded2 = pos_encoder.get_encoded();

        // Verify new output size
        TEST_ASSERT(encoded2->size() == num_vectors2 * pos_encoder.get_output_dim(),
                    "Encoder output size mismatch with different batch size");

        fmt::print("Positional encoder tests passed successfully!\n");
        return 0;
    } catch (const std::exception &e) {
        fmt::print(stderr, "Test failed with exception: {}\n", e.what());
        return 1;
    }
}