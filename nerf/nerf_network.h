// nerf/nerf_network.h
#pragma once

#include <kompute/Kompute.hpp>
#include <memory>
#include <vector>
#include <glm/glm.hpp>

namespace nerf {

// Structure to hold MLP network parameters
struct MLPParams {
    uint32_t D;             // Number of layers in the network
    uint32_t W;             // Width of each layer (number of neurons)
    uint32_t skips;         // Layer at which to add skip connection
    bool use_viewdirs;      // Whether to use view directions
    uint32_t input_ch;      // Input channels for positions
    uint32_t input_ch_views;// Input channels for view directions
    uint32_t output_ch;     // Output channels (4 for RGBA, 5 for RGBA + uncertainty)
};

class NerfNetwork {
public:
    NerfNetwork(kp::Manager &manager, const MLPParams &params);
    ~NerfNetwork() = default;

    // Forward pass through the network
    void forward(std::shared_ptr<kp::TensorT<float>> positions, 
                 std::shared_ptr<kp::TensorT<float>> directions = nullptr);

    // Get the output of the network
    std::shared_ptr<kp::TensorT<float>> get_output();
    
    // Get the output of the network (synced to CPU)
    std::shared_ptr<kp::TensorT<float>> get_output_sync();

    // Get the network parameters
    MLPParams get_params() const { return params_; }

    // Initialize with random weights
    void initialize_weights();

    // Save weights to file
    void save_weights(const std::string &filename);

    // Load weights from file
    void load_weights(const std::string &filename);

    // Get all network weights as a vector
    std::vector<std::shared_ptr<kp::TensorT<float>>> get_weights();

    // Set weights from a vector
    void set_weights(const std::vector<std::shared_ptr<kp::TensorT<float>>> &weights);

private:
    kp::Manager &manager_;
    MLPParams params_;

    // Network weights
    std::vector<std::shared_ptr<kp::TensorT<float>>> weights_; // Weights for each layer
    std::vector<std::shared_ptr<kp::TensorT<float>>> biases_;  // Biases for each layer

    // For view-dependent model
    std::shared_ptr<kp::TensorT<float>> feature_weights_;
    std::shared_ptr<kp::TensorT<float>> feature_bias_;
    std::shared_ptr<kp::TensorT<float>> alpha_weights_;
    std::shared_ptr<kp::TensorT<float>> alpha_bias_;
    std::shared_ptr<kp::TensorT<float>> rgb_weights_;
    std::shared_ptr<kp::TensorT<float>> rgb_bias_;
    std::vector<std::shared_ptr<kp::TensorT<float>>> view_weights_; // Weights for view branch
    std::vector<std::shared_ptr<kp::TensorT<float>>> view_biases_;  // Biases for view branch

    // Input/output tensors
    std::shared_ptr<kp::TensorT<float>> positions_;
    std::shared_ptr<kp::TensorT<float>> directions_;
    std::shared_ptr<kp::TensorT<float>> output_;
    std::shared_ptr<kp::TensorT<float>> hidden_states_; // For intermediate layer outputs
    std::shared_ptr<kp::TensorT<float>> skip_input_;    // For skip connection input

    // Algorithms for each layer
    std::vector<std::shared_ptr<kp::Algorithm>> layer_algos_;
    std::shared_ptr<kp::Algorithm> view_merge_algo_;  // Merge features with view direction
    std::shared_ptr<kp::Algorithm> view_output_algo_; // Final RGB output from view branch
    std::shared_ptr<kp::Algorithm> density_algo_;     // Density output
    std::shared_ptr<kp::Algorithm> output_algo_;      // Final output assembly

    // Initialize algorithms for each layer
    void initialize_algorithms();

    // Initialize tensor for intermediate results
    void initialize_tensors(uint32_t batch_size);

    // Create a single layer algorithm
    std::shared_ptr<kp::Algorithm> create_layer_algorithm(
        uint32_t input_dim, uint32_t output_dim, 
        std::shared_ptr<kp::TensorT<float>> input_tensor,
        std::shared_ptr<kp::TensorT<float>> output_tensor,
        std::shared_ptr<kp::TensorT<float>> weights,
        std::shared_ptr<kp::TensorT<float>> biases,
        bool skip_connection = false,
        std::shared_ptr<kp::TensorT<float>> skip_input = nullptr);
};

} // namespace nerf