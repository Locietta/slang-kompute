// nerf/positional_encoder.h
#pragma once

#include <kompute/Kompute.hpp>
#include <memory>
#include <vector>
#include <glm/glm.hpp>

namespace nerf {

// Enum for different types of positional encoders
enum class EncoderType {
    POSITION, // For encoding positions (higher frequency)
    DIRECTION // For encoding view directions (lower frequency)
};

class PositionalEncoder {
public:
    // Constructor with individual parameters
    PositionalEncoder(kp::Manager &manager,
                      uint32_t num_freqs = 10,
                      bool include_input = true,
                      bool log_sampling = true);

    // Constructor with predefined encoder type
    PositionalEncoder(kp::Manager &manager, EncoderType type);

    ~PositionalEncoder() = default;

    // Encode input positions or directions
    void encode(std::shared_ptr<kp::TensorT<float>> inputs);

    // Get encoded output
    std::shared_ptr<kp::TensorT<float>> get_encoded();

    // Get output dimension
    uint32_t get_output_dim() const { return output_dim_; }

private:
    kp::Manager &manager_;
    uint32_t num_freqs_;
    bool include_input_;
    bool log_sampling_;
    uint32_t input_dim_;  // Dimension of input (usually 3 for position, direction)
    uint32_t output_dim_; // Dimension of output encoding

    // Frequency bands (store on CPU, upload to GPU as constants)
    std::vector<float> frequency_bands_;

    // Output tensor (will be resized as needed)
    std::shared_ptr<kp::TensorT<float>> encoded_;
    std::shared_ptr<kp::TensorT<float>> freqs_tensor_;

    // Initialize frequency bands
    void init_frequency_bands();

    // Create algorithm for a specific batch size
    std::shared_ptr<kp::Algorithm> create_algorithm(
        std::shared_ptr<kp::TensorT<float>> inputs,
        std::shared_ptr<kp::TensorT<float>> encoded);
};

} // namespace nerf