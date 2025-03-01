// nerf/positional_encoder.cpp
#include "positional_encoder.h"
#include "utils.hpp"

namespace nerf {

PositionalEncoder::PositionalEncoder(kp::Manager &manager,
                                     uint32_t num_freqs,
                                     bool include_input,
                                     bool log_sampling)
    : manager_(manager),
      num_freqs_(num_freqs),
      include_input_(include_input),
      log_sampling_(log_sampling),
      input_dim_(3) // Fixed to 3 for positions/directions in NeRF
{
    // Initialize frequency bands
    init_frequency_bands();

    // Create frequency tensor and sync it to device (this doesn't change)
    freqs_tensor_ = manager_.tensorT(frequency_bands_);
    manager_.sequence()
        ->record<kp::OpSyncDevice>({freqs_tensor_})
        ->eval();

    // Calculate output dimension
    // If include_input, we keep the original input
    // Then we add sine and cosine for each frequency band and each input dimension
    output_dim_ = include_input_ ? input_dim_ : 0;
    output_dim_ += 2 * input_dim_ * num_freqs_;
}

PositionalEncoder::PositionalEncoder(kp::Manager &manager, EncoderType type)
    : PositionalEncoder(manager,
                        (type == EncoderType::POSITION) ? 10 : 4, // num_freqs
                        true,                                     // include_input
                        true)                                     // log_sampling
{
}

void PositionalEncoder::init_frequency_bands() {
    frequency_bands_.resize(num_freqs_);

    for (uint32_t i = 0; i < num_freqs_; i++) {
        if (log_sampling_) {
            // Log sampling: 2^0, 2^1, 2^2, ..., 2^(L-1)
            frequency_bands_[i] = std::pow(2.0f, static_cast<float>(i));
        } else {
            // Linear sampling from 2^0 to 2^(L-1)
            float max_freq_log2 = static_cast<float>(num_freqs_ - 1);
            frequency_bands_[i] = std::pow(2.0f,
                                           static_cast<float>(i) / static_cast<float>(num_freqs_ - 1) * max_freq_log2);
        }
    }
}

std::shared_ptr<kp::Algorithm> PositionalEncoder::create_algorithm(
    std::shared_ptr<kp::TensorT<float>> inputs,
    std::shared_ptr<kp::TensorT<float>> encoded) {

    // Load shader code
    constexpr auto k_cs_code = bytes_to_words(
#include "positional_encoder.spv.h"
    );
    std::vector<uint32_t> spirv(k_cs_code.begin(), k_cs_code.end());

    // Set up algorithm parameters
    std::vector<std::shared_ptr<kp::Memory>> algo_params = {
        freqs_tensor_,
        inputs,
        encoded};

    // Calculate workgroup size
    uint32_t workgroup_size = 64;
    uint32_t num_vectors = inputs->size() / input_dim_;
    uint32_t num_groups = divide_and_round_up(num_vectors, workgroup_size);

    // Create algorithm with specialization constants
    return manager_.algorithm<uint32_t>(
        algo_params, spirv, kp::Workgroup({num_groups, 1, 1}),
        {
            num_freqs_,
            include_input_ ? 1u : 0u, // Convert bool to uint32_t for specialization constant
        },
        {});
}

// Modify positional_encoder.cpp
void PositionalEncoder::encode(std::shared_ptr<kp::TensorT<float>> inputs) {
    // Calculate number of input vectors
    uint32_t num_vectors = inputs->size() / input_dim_;

    // Create or resize output tensor if needed
    if (!encoded_ || encoded_->size() != num_vectors * output_dim_) {
        std::vector<float> zeros(num_vectors * output_dim_, 0.0f);
        encoded_ = manager_.tensorT(zeros);
    }

    // Create algorithm for this specific input/output pair
    auto algorithm = create_algorithm(inputs, encoded_);

    // Run the encoding algorithm - don't sync to CPU
    manager_.sequence()
        ->record<kp::OpSyncDevice>({inputs})
        ->record<kp::OpAlgoDispatch>(algorithm)
        ->eval();
}

std::shared_ptr<kp::TensorT<float>> PositionalEncoder::get_encoded() {
    return encoded_;
}

std::shared_ptr<kp::TensorT<float>> PositionalEncoder::get_encoded_sync() {
    // Sync to CPU before returning for testing purposes
    manager_.sequence()
        ->record<kp::OpSyncLocal>({encoded_})
        ->eval();
    return encoded_;
}

} // namespace nerf