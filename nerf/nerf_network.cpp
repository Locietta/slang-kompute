// nerf/nerf_network.cpp
#include "nerf_network.h"
#include "utils.hpp"
#include <fstream>
#include <random>
#include <stdexcept>

namespace nerf {

const static std::vector<uint32_t> k_spirv_mlp = [] {
    constexpr auto layer_cs_code = bytes_to_words(
#include "nerf_network_layer.spv.h"
    );
    return std::vector<uint32_t>(layer_cs_code.begin(), layer_cs_code.end());
}();

const static std::vector<uint32_t> k_spirv_mlp_backward = [] {
    constexpr auto layer_cs_code = bytes_to_words(
#include "nerf_network_layer_backward.spv.h"
    );
    return std::vector<uint32_t>(layer_cs_code.begin(), layer_cs_code.end());
}();

const static std::vector<uint32_t> k_spirv_feature = [] {
    constexpr auto layer_cs_code = bytes_to_words(
#include "nerf_feature_extraction.spv.h"
    );
    return std::vector<uint32_t>(layer_cs_code.begin(), layer_cs_code.end());
}();

const static std::vector<uint32_t> k_spirv_feature_backward = [] {
    constexpr auto layer_cs_code = bytes_to_words(
#include "nerf_feature_backward.spv.h"
    );
    return std::vector<uint32_t>(layer_cs_code.begin(), layer_cs_code.end());
}();

const static std::vector<uint32_t> k_spirv_view = [] {
    constexpr auto layer_cs_code = bytes_to_words(
#include "nerf_view_dependent.spv.h"
    );
    return std::vector<uint32_t>(layer_cs_code.begin(), layer_cs_code.end());
}();

const static std::vector<uint32_t> k_spirv_view_backward = [] {
    constexpr auto layer_cs_code = bytes_to_words(
#include "nerf_view_backward.spv.h"
    );
    return std::vector<uint32_t>(layer_cs_code.begin(), layer_cs_code.end());
}();

NerfNetwork::NerfNetwork(kp::Manager &manager, const MLPParams &params)
    : manager_(manager), params_(params) {

    // Initialize weights
    initialize_weights();

    // Initialize gradients
    initialize_gradients();

    // Initialize algorithms
    initialize_algorithms();
}

void NerfNetwork::initialize_weights() {
    // Create random number generator
    std::random_device rd;
    std::mt19937 gen(rd());

    const float std_dev = sqrt(1.0f / (float) params_.W);

    std::normal_distribution<float> dist(0.0f, std_dev); // Small standard deviation for initialization

    // Initialize MLP weights
    weights_.clear();
    biases_.clear();

    // First layer: input_ch -> W
    uint32_t prev_dim = params_.input_ch;

    for (uint32_t i = 0; i < params_.D; i++) {
        uint32_t current_dim = params_.W;

        // For layer at 'skips', we add a skip connection
        if (i == params_.skips) {
            prev_dim = params_.W + params_.input_ch;
        }

        // Initialize weights for this layer
        std::vector<float> layer_weights(prev_dim * current_dim);
        for (auto &w : layer_weights) {
            w = dist(gen);
        }
        weights_.push_back(manager_.tensorT(layer_weights));

        // Initialize biases for this layer
        std::vector<float> layer_biases(current_dim, 0.0f);
        biases_.push_back(manager_.tensorT(layer_biases));

        prev_dim = current_dim;
    }

    if (params_.use_viewdirs) {
        // Feature network for view-dependent effects
        std::vector<float> feature_w(params_.W * params_.W);
        std::vector<float> feature_b(params_.W, 0.0f);
        for (auto &w : feature_w) {
            w = dist(gen);
        }
        feature_weights_ = manager_.tensorT(feature_w);
        feature_bias_ = manager_.tensorT(feature_b);

        // Alpha (density) output
        std::vector<float> alpha_w(params_.W);
        std::vector<float> alpha_b(1, 0.0f);
        for (auto &w : alpha_w) {
            w = dist(gen);
        }
        alpha_weights_ = manager_.tensorT(alpha_w);
        alpha_bias_ = manager_.tensorT(alpha_b);

        // View-dependent branch
        view_weights_.clear();
        view_biases_.clear();

        // First view layer: input_ch_views + W -> W/2
        uint32_t view_prev_dim = params_.input_ch_views + params_.W;
        uint32_t view_current_dim = params_.W / 2;

        std::vector<float> view_layer_weights(view_prev_dim * view_current_dim);
        for (auto &w : view_layer_weights) {
            w = dist(gen);
        }
        view_weights_.push_back(manager_.tensorT(view_layer_weights));

        std::vector<float> view_layer_biases(view_current_dim, 0.0f);
        view_biases_.push_back(manager_.tensorT(view_layer_biases));

        // RGB output layer
        std::vector<float> rgb_w(view_current_dim * 3);
        std::vector<float> rgb_b(3, 0.0f);
        for (auto &w : rgb_w) {
            w = dist(gen);
        }
        rgb_weights_ = manager_.tensorT(rgb_w);
        rgb_bias_ = manager_.tensorT(rgb_b);
    } else {
        // Simple output layer
        std::vector<float> output_w(params_.W * params_.output_ch);
        std::vector<float> output_b(params_.output_ch, 0.0f);
        for (auto &w : output_w) {
            w = dist(gen);
        }
        rgb_weights_ = manager_.tensorT(output_w);
        rgb_bias_ = manager_.tensorT(output_b);
    }

    // Sync all weights to the device
    std::vector<std::shared_ptr<kp::Memory>> all_weights;
    all_weights.insert(all_weights.end(), weights_.begin(), weights_.end());
    all_weights.insert(all_weights.end(), biases_.begin(), biases_.end());

    if (params_.use_viewdirs) {
        all_weights.push_back(feature_weights_);
        all_weights.push_back(feature_bias_);
        all_weights.push_back(alpha_weights_);
        all_weights.push_back(alpha_bias_);
        all_weights.insert(all_weights.end(), view_weights_.begin(), view_weights_.end());
        all_weights.insert(all_weights.end(), view_biases_.begin(), view_biases_.end());
        all_weights.push_back(rgb_weights_);
        all_weights.push_back(rgb_bias_);
    } else {
        all_weights.push_back(rgb_weights_);
        all_weights.push_back(rgb_bias_);
    }

    manager_.sequence()
        ->record<kp::OpSyncDevice>(all_weights)
        ->eval();
}

void NerfNetwork::initialize_gradients() {
    // Initialize gradients for MLP weights and biases
    grad_weights_.clear();
    grad_biases_.clear();

    for (size_t i = 0; i < weights_.size(); i++) {
        std::vector<float> grad_w(weights_[i]->size(), 0.0f);
        std::vector<float> grad_b(biases_[i]->size(), 0.0f);
        grad_weights_.push_back(manager_.tensorT(grad_w));
        grad_biases_.push_back(manager_.tensorT(grad_b));
    }

    // Initialize gradients for hidden states - one per layer
    grad_hidden_states_.clear();
    for (uint32_t i = 0; i < params_.D; i++) {
        // We'll resize these properly in backward() when we know the batch size
        std::vector<float> grad_layer(0, 0.0f);
        grad_hidden_states_.push_back(manager_.tensorT(grad_layer));
    }

    if (params_.use_viewdirs) {
        // Initialize gradients for view-dependent components
        std::vector<float> grad_feature_w(params_.W * params_.W, 0.0f);
        std::vector<float> grad_feature_b(params_.W, 0.0f);
        grad_feature_weights_ = manager_.tensorT(grad_feature_w);
        grad_feature_bias_ = manager_.tensorT(grad_feature_b);

        std::vector<float> grad_alpha_w(params_.W, 0.0f);
        std::vector<float> grad_alpha_b(1, 0.0f);
        grad_alpha_weights_ = manager_.tensorT(grad_alpha_w);
        grad_alpha_bias_ = manager_.tensorT(grad_alpha_b);

        // View branch gradients
        grad_view_weights_.clear();
        grad_view_biases_.clear();

        uint32_t view_prev_dim = params_.input_ch_views + params_.W;
        uint32_t view_current_dim = params_.W / 2;

        std::vector<float> grad_view_w(view_prev_dim * view_current_dim, 0.0f);
        std::vector<float> grad_view_b(view_current_dim, 0.0f);
        grad_view_weights_.push_back(manager_.tensorT(grad_view_w));
        grad_view_biases_.push_back(manager_.tensorT(grad_view_b));

        std::vector<float> grad_rgb_w(view_current_dim * 3, 0.0f);
        std::vector<float> grad_rgb_b(3, 0.0f);
        grad_rgb_weights_ = manager_.tensorT(grad_rgb_w);
        grad_rgb_bias_ = manager_.tensorT(grad_rgb_b);

        // Additional storage for feature extraction gradients
        std::vector<float> grad_density_feat(0, 0.0f); // Will resize in backward()
        grad_density_features_ = manager_.tensorT(grad_density_feat);
    } else {
        // Simple output gradients
        std::vector<float> grad_rgb_w(params_.W * params_.output_ch, 0.0f);
        std::vector<float> grad_rgb_b(params_.output_ch, 0.0f);
        grad_rgb_weights_ = manager_.tensorT(grad_rgb_w);
        grad_rgb_bias_ = manager_.tensorT(grad_rgb_b);
    }
}

void NerfNetwork::initialize_algorithms() {
    // We'll create the actual algorithms when we know the batch size in forward()
    layer_algos_.clear();
}

void NerfNetwork::initialize_tensors(uint32_t batch_size) {
    // Initialize separate hidden state tensors for each layer
    hidden_states_.clear();

    // Create a tensor for each layer's output
    for (uint32_t i = 0; i < params_.D; i++) {
        std::vector<float> layer_output(batch_size * params_.W, 0.0f);
        hidden_states_.push_back(manager_.tensorT(layer_output));
    }

    if (params_.use_viewdirs) {
        // For view-dependent model, we need additional tensors
        std::vector<float> output_data(batch_size * params_.output_ch, 0.0f);
        output_ = manager_.tensorT(output_data);
    } else {
        // Simple output is directly from the MLP
        std::vector<float> output_data(batch_size * params_.output_ch, 0.0f);
        output_ = manager_.tensorT(output_data);
    }

    // For skip connections, we need to store the input
    if (params_.skips > 0) {
        std::vector<float> skip_data(batch_size * params_.input_ch, 0.0f);
        skip_input_ = manager_.tensorT(skip_data);
    }
}

std::shared_ptr<kp::Algorithm> NerfNetwork::create_layer_algorithm(
    uint32_t input_dim, uint32_t output_dim,
    std::shared_ptr<kp::TensorT<float>> input_tensor,
    std::shared_ptr<kp::TensorT<float>> output_tensor,
    std::shared_ptr<kp::TensorT<float>> weights,
    std::shared_ptr<kp::TensorT<float>> biases,
    bool skip_connection,
    std::shared_ptr<kp::TensorT<float>> skip_input) {

    // Set up parameters
    std::vector<std::shared_ptr<kp::Memory>> algo_params = {
        input_tensor,
        output_tensor,
        weights,
        biases};

    if (skip_connection) {
        algo_params.push_back(skip_input);
    }

    // Calculate workgroup size
    constexpr uint32_t workgroup_size = 64;
    uint32_t batch_size = input_tensor->size() / input_dim;
    uint32_t num_groups = divide_and_round_up(batch_size, workgroup_size);

    // Layer parameters
    struct LayerParams {
        uint32_t input_dim;
        uint32_t output_dim;
        uint32_t batch_size;
        uint32_t use_skip_connection;
    };

    LayerParams layer_params = {
        input_dim,
        output_dim,
        batch_size,
        skip_connection ? 1u : 0u};

    // Create algorithm
    return manager_.algorithm<uint32_t, LayerParams>(
        algo_params, k_spirv_mlp, kp::Workgroup({num_groups, 1, 1}), {}, {layer_params});
}

void NerfNetwork::forward(std::shared_ptr<kp::TensorT<float>> positions,
                          std::shared_ptr<kp::TensorT<float>> directions) {
    // Store input tensors
    positions_ = positions;
    directions_ = directions;

    // Get batch size from positions tensor
    uint32_t batch_size = positions_->size() / params_.input_ch;

    // Initialize tensors if not already done or if batch size changed
    if (hidden_states_.empty() || (hidden_states_.size() > 0 && hidden_states_[0]->size() != batch_size * params_.W)) {
        initialize_tensors(batch_size);
    }

    // Sync input tensors to device
    manager_.sequence()
        ->record<kp::OpSyncDevice>({positions_})
        ->eval();

    if (directions_ && params_.use_viewdirs) {
        manager_.sequence()
            ->record<kp::OpSyncDevice>({directions_})
            ->eval();
    }

    // Clear existing algorithms - we'll create new ones for this batch size
    layer_algos_.clear();

    // Process MLP layers
    for (uint32_t i = 0; i < params_.D; i++) {
        uint32_t input_dim = (i == 0) ? params_.input_ch : params_.W;
        uint32_t output_dim = params_.W;

        // For layer at 'skips', we need to concatenate the original input
        bool use_skip = (i == params_.skips && i > 0);
        if (use_skip) {
            // Copy positions to skip_input
            manager_.sequence()
                ->record<kp::OpSyncDevice>({positions_,
                                            skip_input_})
                ->eval();

            input_dim = params_.W + params_.input_ch;
        }

        // Get input and output tensors for this layer
        std::shared_ptr<kp::TensorT<float>> input_tensor;
        if (i == 0) {
            input_tensor = positions_;
        } else {
            input_tensor = hidden_states_[i - 1];
        }

        std::shared_ptr<kp::TensorT<float>> output_tensor = hidden_states_[i];

        // Create algorithm for this layer
        auto layer_algo = create_layer_algorithm(
            input_dim, output_dim,
            input_tensor, output_tensor,
            weights_[i], biases_[i],
            use_skip, skip_input_);

        layer_algos_.push_back(layer_algo);

        // Run the layer
        manager_.sequence()
            ->record<kp::OpAlgoDispatch>(layer_algo)
            ->eval();
    }

    // After MLP processing, handle output
    if (params_.use_viewdirs) {
        // 1. Extract features and compute density from the last MLP layer
        std::shared_ptr<kp::TensorT<float>> last_layer_output = hidden_states_[params_.D - 1];

        // Create intermediate tensors for extracted features and density
        std::vector<float> features_data(batch_size * params_.W, 0.0f);
        std::vector<float> density_data(batch_size, 0.0f);
        auto features_tensor = manager_.tensorT(features_data);
        auto density_tensor = manager_.tensorT(density_data);

        // Create feature extraction algorithm
        std::vector<std::shared_ptr<kp::Memory>> feature_algo_params = {
            last_layer_output,
            features_tensor,
            density_tensor,
            feature_weights_,
            feature_bias_,
            alpha_weights_,
            alpha_bias_};

        constexpr uint32_t workgroup_size = 64;
        uint32_t num_groups = divide_and_round_up(batch_size, workgroup_size);

        // Feature extraction parameters
        struct FeatureParams {
            uint32_t batch_size;
            uint32_t feature_dim;
        };

        FeatureParams feature_params = {
            batch_size,
            params_.W};

        // Create and run feature extraction algorithm
        auto feature_algo = manager_.algorithm<uint32_t, FeatureParams>(
            feature_algo_params, k_spirv_feature, kp::Workgroup({num_groups, 1, 1}), {}, {feature_params});

        manager_.sequence()
            ->record<kp::OpAlgoDispatch>(feature_algo)
            ->eval();

        // 2. Process view-dependent branch
        // Make sure directions are valid
        if (!directions_ || directions_->size() != batch_size * params_.input_ch_views) {
            throw std::runtime_error("View directions tensor has incorrect size or is not provided");
        }

        // Create view-dependent algorithm
        std::vector<std::shared_ptr<kp::Memory>> view_algo_params = {
            features_tensor,
            directions_,
            density_tensor,
            output_,
            view_weights_[0], // Assuming one view layer for simplicity
            view_biases_[0],
            rgb_weights_,
            rgb_bias_};

        // View-dependent parameters
        struct ViewParams {
            uint32_t batch_size;
            uint32_t feature_dim;
            uint32_t view_dim;
            uint32_t rgb_dim;
        };

        ViewParams view_params = {
            batch_size,
            params_.W,
            params_.input_ch_views,
            3 // RGB has 3 components
        };

        // Create and run view-dependent algorithm
        auto view_algo = manager_.algorithm<uint32_t, ViewParams>(
            view_algo_params, k_spirv_view, kp::Workgroup({num_groups, 1, 1}), {}, {view_params});

        manager_.sequence()
            ->record<kp::OpAlgoDispatch>(view_algo)
            ->eval();
    } else {
        // For simple model, just apply final linear layer
        std::shared_ptr<kp::TensorT<float>> last_layer_output = hidden_states_[params_.D - 1];

        // Create final output algorithm
        auto output_algo = create_layer_algorithm(
            params_.W, params_.output_ch,
            last_layer_output, output_,
            rgb_weights_, rgb_bias_,
            false, nullptr // No skip connection for final layer
        );

        // Run final layer
        manager_.sequence()
            ->record<kp::OpAlgoDispatch>(output_algo)
            ->eval();
    }
}

void NerfNetwork::backward(std::shared_ptr<kp::TensorT<float>> grad_output) {
    grad_output_ = grad_output;
    // Get batch size from positions tensor
    uint32_t batch_size = positions_->size() / params_.input_ch;

    // Sync gradient input to device
    manager_.sequence()
        ->record<kp::OpSyncDevice>({grad_output_})
        ->eval();

    if (params_.use_viewdirs) {
        // For view-dependent model, we need to backpropagate through:
        // 1. RGB output layer
        // 2. View-dependent branch
        // 3. Feature extraction layer
        // 4. Main MLP

        // 1. Backpropagate through view-dependent branch
        constexpr uint32_t workgroup_size = 64;
        uint32_t num_groups = divide_and_round_up(batch_size, workgroup_size);

        // View-dependent parameters
        struct ViewParams {
            uint32_t batch_size;
            uint32_t feature_dim;
            uint32_t view_dim;
            uint32_t rgb_dim;
        };

        ViewParams view_params = {
            batch_size,
            params_.W,
            params_.input_ch_views,
            3 // RGB has 3 components
        };

        // Create temporary buffer for alpha gradients
        std::vector<float> alpha_grad_data(batch_size, 0.0f);
        auto alpha_grad = manager_.tensorT(alpha_grad_data);

        // Create view backward algorithm
        std::vector<std::shared_ptr<kp::Memory>> view_backward_params = {
            grad_output_,           // Gradient from output
            feature_weights_,       // Extracted features
            directions_,            // View directions
            grad_density_features_, // Gradient for features
            grad_view_weights_[0],  // Gradient for view weights
            grad_view_biases_[0],   // Gradient for view bias
            grad_rgb_weights_,      // Gradient for RGB weights
            grad_rgb_bias_,         // Gradient for RGB bias
            view_weights_[0],       // Original view weights
            rgb_weights_,           // Original RGB weights
            view_biases_[0],        // Original view bias
            rgb_bias_,              // Original RGB bias
            alpha_grad              // Gradient for density (alpha)
        };

        auto view_backward_algo = manager_.algorithm<uint32_t, ViewParams>(
            view_backward_params, k_spirv_view_backward,
            kp::Workgroup({num_groups, 1, 1}), {}, {view_params});

        manager_.sequence()
            ->record<kp::OpAlgoDispatch>(view_backward_algo)
            ->eval();

        // 2. Backpropagate through feature extraction
        struct FeatureParams {
            uint32_t batch_size;
            uint32_t feature_dim;
        };

        FeatureParams feature_params = {
            batch_size,
            params_.W};

        // Get last layer output
        std::shared_ptr<kp::TensorT<float>> last_layer_output = hidden_states_[params_.D - 1];

        // Create feature backward algorithm
        std::vector<std::shared_ptr<kp::Memory>> feature_backward_params = {
            grad_density_features_,             // Gradient from view branch
            alpha_grad,                         // Gradient for density
            last_layer_output,                  // Last layer features
            grad_feature_weights_,              // Gradient for feature weights
            grad_feature_bias_,                 // Gradient for feature bias
            grad_alpha_weights_,                // Gradient for alpha weights
            grad_alpha_bias_,                   // Gradient for alpha bias
            grad_hidden_states_[params_.D - 1], // Gradient for MLP output
            feature_weights_,                   // Original feature weights
            alpha_weights_,                     // Original alpha weights
            feature_bias_                       // Original feature bias
        };

        auto feature_backward_algo = manager_.algorithm<uint32_t, FeatureParams>(
            feature_backward_params, k_spirv_feature_backward,
            kp::Workgroup({num_groups, 1, 1}), {}, {feature_params});

        manager_.sequence()
            ->record<kp::OpAlgoDispatch>(feature_backward_algo)
            ->eval();
    } else {
        // For simple model, backpropagate directly from output to last layer

        // Backpropagate through final output layer
        struct LayerParams {
            uint32_t input_dim;
            uint32_t output_dim;
            uint32_t batch_size;
            uint32_t use_skip_connection;
        };

        LayerParams output_layer_params = {
            params_.W,
            params_.output_ch,
            batch_size,
            0u // No skip connection
        };

        std::shared_ptr<kp::TensorT<float>> last_layer_output = hidden_states_[params_.D - 1];

        // Create backward algorithm for output layer
        std::vector<std::shared_ptr<kp::Memory>> output_backward_params = {
            grad_output_,                       // Gradient from output
            last_layer_output,                  // Layer input
            output_,                            // Layer output
            rgb_weights_,                       // Layer weights
            rgb_bias_,                          // Layer bias
            grad_hidden_states_[params_.D - 1], // Gradient for previous layer
            grad_rgb_weights_,                  // Gradient for weights
            grad_rgb_bias_,                     // Gradient for bias
            nullptr,                            // No skip connection
            nullptr                             // No skip gradient
        };

        constexpr uint32_t workgroup_size = 64;
        uint32_t num_groups = divide_and_round_up(batch_size, workgroup_size);

        auto output_backward_algo = manager_.algorithm<uint32_t, LayerParams>(
            output_backward_params, k_spirv_mlp_backward,
            kp::Workgroup({num_groups, 1, 1}), {}, {output_layer_params});

        manager_.sequence()
            ->record<kp::OpAlgoDispatch>(output_backward_algo)
            ->eval();
    }

    // 3. Backpropagate through main MLP layers
    for (int32_t i = params_.D - 1; i >= 0; i--) {
        uint32_t input_dim = (i == 0) ? params_.input_ch : params_.W;
        uint32_t output_dim = params_.W;

        // For layer at 'skips', we have a skip connection
        bool use_skip = (i == params_.skips && i > 0);
        if (use_skip) {
            input_dim = params_.W + params_.input_ch;
        }

        // Get input and output tensors for this layer
        std::shared_ptr<kp::TensorT<float>> input_tensor;
        if (i == 0) {
            input_tensor = positions_;
        } else {
            input_tensor = hidden_states_[i - 1];
        }

        std::shared_ptr<kp::TensorT<float>> output_tensor = hidden_states_[i];

        // Get gradient tensors
        std::shared_ptr<kp::TensorT<float>> grad_input_tensor;
        if (i == 0) {
            // For first layer, we don't need to backpropagate to input
            std::vector<float> dummy_grad(batch_size * input_dim, 0.0f);
            grad_input_tensor = manager_.tensorT(dummy_grad);
        } else {
            grad_input_tensor = grad_hidden_states_[i - 1];
        }

        std::shared_ptr<kp::TensorT<float>> grad_output_tensor = grad_hidden_states_[i];

        // Create backward algorithm parameters
        struct LayerParams {
            uint32_t input_dim;
            uint32_t output_dim;
            uint32_t batch_size;
            uint32_t use_skip_connection;
        };

        LayerParams layer_params = {
            input_dim,
            output_dim,
            batch_size,
            use_skip ? 1u : 0u};

        // Create skip connection tensors if needed
        std::shared_ptr<kp::TensorT<float>> grad_skip = nullptr;
        if (use_skip) {
            std::vector<float> grad_skip_data(batch_size * params_.input_ch, 0.0f);
            grad_skip = manager_.tensorT(grad_skip_data);
        }

        // Create backward algorithm
        std::vector<std::shared_ptr<kp::Memory>> layer_backward_params = {
            grad_output_tensor,               // Gradient from next layer
            input_tensor,                     // Layer input
            output_tensor,                    // Layer output
            weights_[i],                      // Layer weights
            biases_[i],                       // Layer bias
            grad_input_tensor,                // Gradient for previous layer
            grad_weights_[i],                 // Gradient for weights
            grad_biases_[i],                  // Gradient for bias
            use_skip ? skip_input_ : nullptr, // Skip connection input
            use_skip ? grad_skip : nullptr    // Gradient for skip input
        };

        constexpr uint32_t workgroup_size = 64;
        uint32_t num_groups = divide_and_round_up(batch_size, workgroup_size);

        auto layer_backward_algo = manager_.algorithm<uint32_t, LayerParams>(
            layer_backward_params, k_spirv_mlp_backward,
            kp::Workgroup({num_groups, 1, 1}), {}, {layer_params});

        manager_.sequence()
            ->record<kp::OpAlgoDispatch>(layer_backward_algo)
            ->eval();
    }
}

std::shared_ptr<kp::TensorT<float>> NerfNetwork::get_output() {
    return output_;
}

std::shared_ptr<kp::TensorT<float>> NerfNetwork::get_output_sync() {
    manager_.sequence()
        ->record<kp::OpSyncLocal>({output_})
        ->eval();
    return output_;
}

void NerfNetwork::save_weights(const std::string &filename) {
    // First sync all weights from the device
    std::vector<std::shared_ptr<kp::Memory>> all_weights;
    all_weights.insert(all_weights.end(), weights_.begin(), weights_.end());
    all_weights.insert(all_weights.end(), biases_.begin(), biases_.end());

    if (params_.use_viewdirs) {
        all_weights.push_back(feature_weights_);
        all_weights.push_back(feature_bias_);
        all_weights.push_back(alpha_weights_);
        all_weights.push_back(alpha_bias_);
        all_weights.insert(all_weights.end(), view_weights_.begin(), view_weights_.end());
        all_weights.insert(all_weights.end(), view_biases_.begin(), view_biases_.end());
        all_weights.push_back(rgb_weights_);
        all_weights.push_back(rgb_bias_);
    } else {
        all_weights.push_back(rgb_weights_);
        all_weights.push_back(rgb_bias_);
    }

    manager_.sequence()
        ->record<kp::OpSyncLocal>(all_weights)
        ->eval();

    // Open file for writing
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    // Write header information
    file.write(reinterpret_cast<const char *>(&params_.D), sizeof(params_.D));
    file.write(reinterpret_cast<const char *>(&params_.W), sizeof(params_.W));
    file.write(reinterpret_cast<const char *>(&params_.skips), sizeof(params_.skips));
    file.write(reinterpret_cast<const char *>(&params_.use_viewdirs), sizeof(params_.use_viewdirs));
    file.write(reinterpret_cast<const char *>(&params_.input_ch), sizeof(params_.input_ch));
    file.write(reinterpret_cast<const char *>(&params_.input_ch_views), sizeof(params_.input_ch_views));
    file.write(reinterpret_cast<const char *>(&params_.output_ch), sizeof(params_.output_ch));

    // Write main MLP weights
    for (size_t i = 0; i < weights_.size(); i++) {
        uint32_t size = weights_[i]->size();
        file.write(reinterpret_cast<const char *>(&size), sizeof(size));
        file.write(reinterpret_cast<const char *>(weights_[i]->data()), size * sizeof(float));

        size = biases_[i]->size();
        file.write(reinterpret_cast<const char *>(&size), sizeof(size));
        file.write(reinterpret_cast<const char *>(biases_[i]->data()), size * sizeof(float));
    }

    if (params_.use_viewdirs) {
        // Write view-dependent model weights
        uint32_t size = feature_weights_->size();
        file.write(reinterpret_cast<const char *>(&size), sizeof(size));
        file.write(reinterpret_cast<const char *>(feature_weights_->data()), size * sizeof(float));

        size = feature_bias_->size();
        file.write(reinterpret_cast<const char *>(&size), sizeof(size));
        file.write(reinterpret_cast<const char *>(feature_bias_->data()), size * sizeof(float));

        size = alpha_weights_->size();
        file.write(reinterpret_cast<const char *>(&size), sizeof(size));
        file.write(reinterpret_cast<const char *>(alpha_weights_->data()), size * sizeof(float));

        size = alpha_bias_->size();
        file.write(reinterpret_cast<const char *>(&size), sizeof(size));
        file.write(reinterpret_cast<const char *>(alpha_bias_->data()), size * sizeof(float));

        for (size_t i = 0; i < view_weights_.size(); i++) {
            size = view_weights_[i]->size();
            file.write(reinterpret_cast<const char *>(&size), sizeof(size));
            file.write(reinterpret_cast<const char *>(view_weights_[i]->data()), size * sizeof(float));

            size = view_biases_[i]->size();
            file.write(reinterpret_cast<const char *>(&size), sizeof(size));
            file.write(reinterpret_cast<const char *>(view_biases_[i]->data()), size * sizeof(float));
        }

        size = rgb_weights_->size();
        file.write(reinterpret_cast<const char *>(&size), sizeof(size));
        file.write(reinterpret_cast<const char *>(rgb_weights_->data()), size * sizeof(float));

        size = rgb_bias_->size();
        file.write(reinterpret_cast<const char *>(&size), sizeof(size));
        file.write(reinterpret_cast<const char *>(rgb_bias_->data()), size * sizeof(float));
    } else {
        // Write simple output weights
        uint32_t size = rgb_weights_->size();
        file.write(reinterpret_cast<const char *>(&size), sizeof(size));
        file.write(reinterpret_cast<const char *>(rgb_weights_->data()), size * sizeof(float));

        size = rgb_bias_->size();
        file.write(reinterpret_cast<const char *>(&size), sizeof(size));
        file.write(reinterpret_cast<const char *>(rgb_bias_->data()), size * sizeof(float));
    }
}

void NerfNetwork::load_weights(const std::string &filename) {
    // Open file for reading
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for reading: " + filename);
    }

    // Read header information
    MLPParams loaded_params;
    file.read(reinterpret_cast<char *>(&loaded_params.D), sizeof(loaded_params.D));
    file.read(reinterpret_cast<char *>(&loaded_params.W), sizeof(loaded_params.W));
    file.read(reinterpret_cast<char *>(&loaded_params.skips), sizeof(loaded_params.skips));
    file.read(reinterpret_cast<char *>(&loaded_params.use_viewdirs), sizeof(loaded_params.use_viewdirs));
    file.read(reinterpret_cast<char *>(&loaded_params.input_ch), sizeof(loaded_params.input_ch));
    file.read(reinterpret_cast<char *>(&loaded_params.input_ch_views), sizeof(loaded_params.input_ch_views));
    file.read(reinterpret_cast<char *>(&loaded_params.output_ch), sizeof(loaded_params.output_ch));

    // Verify parameters match
    if (loaded_params.D != params_.D ||
        loaded_params.W != params_.W ||
        loaded_params.skips != params_.skips ||
        loaded_params.use_viewdirs != params_.use_viewdirs ||
        loaded_params.input_ch != params_.input_ch ||
        loaded_params.input_ch_views != params_.input_ch_views ||
        loaded_params.output_ch != params_.output_ch) {
        throw std::runtime_error("Network parameters in file do not match current network");
    }

    // Read main MLP weights
    for (size_t i = 0; i < weights_.size(); i++) {
        uint32_t size;
        file.read(reinterpret_cast<char *>(&size), sizeof(size));
        if (size != weights_[i]->size()) {
            throw std::runtime_error("Weight size mismatch in file");
        }
        file.read(reinterpret_cast<char *>(weights_[i]->data()), size * sizeof(float));

        file.read(reinterpret_cast<char *>(&size), sizeof(size));
        if (size != biases_[i]->size()) {
            throw std::runtime_error("Bias size mismatch in file");
        }
        file.read(reinterpret_cast<char *>(biases_[i]->data()), size * sizeof(float));
    }

    if (params_.use_viewdirs) {
        // Read view-dependent model weights
        uint32_t size;
        file.read(reinterpret_cast<char *>(&size), sizeof(size));
        if (size != feature_weights_->size()) {
            throw std::runtime_error("Feature weight size mismatch in file");
        }
        file.read(reinterpret_cast<char *>(feature_weights_->data()), size * sizeof(float));

        file.read(reinterpret_cast<char *>(&size), sizeof(size));
        if (size != feature_bias_->size()) {
            throw std::runtime_error("Feature bias size mismatch in file");
        }
        file.read(reinterpret_cast<char *>(feature_bias_->data()), size * sizeof(float));

        file.read(reinterpret_cast<char *>(&size), sizeof(size));
        if (size != alpha_weights_->size()) {
            throw std::runtime_error("Alpha weight size mismatch in file");
        }
        file.read(reinterpret_cast<char *>(alpha_weights_->data()), size * sizeof(float));

        file.read(reinterpret_cast<char *>(&size), sizeof(size));
        if (size != alpha_bias_->size()) {
            throw std::runtime_error("Alpha bias size mismatch in file");
        }
        file.read(reinterpret_cast<char *>(alpha_bias_->data()), size * sizeof(float));

        for (size_t i = 0; i < view_weights_.size(); i++) {
            file.read(reinterpret_cast<char *>(&size), sizeof(size));
            if (size != view_weights_[i]->size()) {
                throw std::runtime_error("View weight size mismatch in file");
            }
            file.read(reinterpret_cast<char *>(view_weights_[i]->data()), size * sizeof(float));

            file.read(reinterpret_cast<char *>(&size), sizeof(size));
            if (size != view_biases_[i]->size()) {
                throw std::runtime_error("View bias size mismatch in file");
            }
            file.read(reinterpret_cast<char *>(view_biases_[i]->data()), size * sizeof(float));
        }

        file.read(reinterpret_cast<char *>(&size), sizeof(size));
        if (size != rgb_weights_->size()) {
            throw std::runtime_error("RGB weight size mismatch in file");
        }
        file.read(reinterpret_cast<char *>(rgb_weights_->data()), size * sizeof(float));

        file.read(reinterpret_cast<char *>(&size), sizeof(size));
        if (size != rgb_bias_->size()) {
            throw std::runtime_error("RGB bias size mismatch in file");
        }
        file.read(reinterpret_cast<char *>(rgb_bias_->data()), size * sizeof(float));
    } else {
        // Read simple output weights
        uint32_t size;
        file.read(reinterpret_cast<char *>(&size), sizeof(size));
        if (size != rgb_weights_->size()) {
            throw std::runtime_error("Output weight size mismatch in file");
        }
        file.read(reinterpret_cast<char *>(rgb_weights_->data()), size * sizeof(float));

        file.read(reinterpret_cast<char *>(&size), sizeof(size));
        if (size != rgb_bias_->size()) {
            throw std::runtime_error("Output bias size mismatch in file");
        }
        file.read(reinterpret_cast<char *>(rgb_bias_->data()), size * sizeof(float));
    }

    // Sync all weights to the device
    std::vector<std::shared_ptr<kp::Memory>> all_weights;
    all_weights.insert(all_weights.end(), weights_.begin(), weights_.end());
    all_weights.insert(all_weights.end(), biases_.begin(), biases_.end());

    if (params_.use_viewdirs) {
        all_weights.push_back(feature_weights_);
        all_weights.push_back(feature_bias_);
        all_weights.push_back(alpha_weights_);
        all_weights.push_back(alpha_bias_);
        all_weights.insert(all_weights.end(), view_weights_.begin(), view_weights_.end());
        all_weights.insert(all_weights.end(), view_biases_.begin(), view_biases_.end());
        all_weights.push_back(rgb_weights_);
        all_weights.push_back(rgb_bias_);
    } else {
        all_weights.push_back(rgb_weights_);
        all_weights.push_back(rgb_bias_);
    }

    manager_.sequence()
        ->record<kp::OpSyncDevice>(all_weights)
        ->eval();
}

std::vector<std::shared_ptr<kp::Memory>> NerfNetwork::get_weights() {
    // Return all weights in a vector without syncing to host
    std::vector<std::shared_ptr<kp::Memory>> all_weights;

    // Add all weights to the result vector
    all_weights.insert(all_weights.end(), weights_.begin(), weights_.end());
    all_weights.insert(all_weights.end(), biases_.begin(), biases_.end());

    if (params_.use_viewdirs) {
        all_weights.push_back(feature_weights_);
        all_weights.push_back(feature_bias_);
        all_weights.push_back(alpha_weights_);
        all_weights.push_back(alpha_bias_);
        all_weights.insert(all_weights.end(), view_weights_.begin(), view_weights_.end());
        all_weights.insert(all_weights.end(), view_biases_.begin(), view_biases_.end());
        all_weights.push_back(rgb_weights_);
        all_weights.push_back(rgb_bias_);
    } else {
        all_weights.push_back(rgb_weights_);
        all_weights.push_back(rgb_bias_);
    }

    return all_weights;
}

std::vector<std::shared_ptr<kp::TensorT<float>>> NerfNetwork::get_weights_tensors() {
    // Return all weights in a vector without syncing to host
    std::vector<std::shared_ptr<kp::TensorT<float>>> all_weights;

    // Add all weights to the result vector
    all_weights.insert(all_weights.end(), weights_.begin(), weights_.end());
    all_weights.insert(all_weights.end(), biases_.begin(), biases_.end());

    if (params_.use_viewdirs) {
        all_weights.push_back(feature_weights_);
        all_weights.push_back(feature_bias_);
        all_weights.push_back(alpha_weights_);
        all_weights.push_back(alpha_bias_);
        all_weights.insert(all_weights.end(), view_weights_.begin(), view_weights_.end());
        all_weights.insert(all_weights.end(), view_biases_.begin(), view_biases_.end());
        all_weights.push_back(rgb_weights_);
        all_weights.push_back(rgb_bias_);
    } else {
        all_weights.push_back(rgb_weights_);
        all_weights.push_back(rgb_bias_);
    }

    return all_weights;
}

std::vector<std::shared_ptr<kp::Memory>> NerfNetwork::get_gradients() {
    // Get gradients for MLP weights and biases
    std::vector<std::shared_ptr<kp::Memory>> all_gradients;

    // Add gradients for MLP weights and biases
    all_gradients.insert(all_gradients.end(), grad_weights_.begin(), grad_weights_.end());
    all_gradients.insert(all_gradients.end(), grad_biases_.begin(), grad_biases_.end());

    if (params_.use_viewdirs) {
        all_gradients.push_back(grad_feature_weights_);
        all_gradients.push_back(grad_feature_bias_);
        all_gradients.push_back(grad_alpha_weights_);
        all_gradients.push_back(grad_alpha_bias_);
        all_gradients.insert(all_gradients.end(), grad_view_weights_.begin(), grad_view_weights_.end());
        all_gradients.insert(all_gradients.end(), grad_view_biases_.begin(), grad_view_biases_.end());
        all_gradients.push_back(grad_rgb_weights_);
        all_gradients.push_back(grad_rgb_bias_);
    } else {
        all_gradients.push_back(grad_rgb_weights_);
        all_gradients.push_back(grad_rgb_bias_);
    }

    return all_gradients;
}

std::vector<std::shared_ptr<kp::TensorT<float>>> NerfNetwork::get_gradients_tensors() {
    // Get gradients for MLP weights and biases
    std::vector<std::shared_ptr<kp::TensorT<float>>> all_gradients;

    // Add gradients for MLP weights and biases
    all_gradients.insert(all_gradients.end(), grad_weights_.begin(), grad_weights_.end());
    all_gradients.insert(all_gradients.end(), grad_biases_.begin(), grad_biases_.end());

    if (params_.use_viewdirs) {
        all_gradients.push_back(grad_feature_weights_);
        all_gradients.push_back(grad_feature_bias_);
        all_gradients.push_back(grad_alpha_weights_);
        all_gradients.push_back(grad_alpha_bias_);
        all_gradients.insert(all_gradients.end(), grad_view_weights_.begin(), grad_view_weights_.end());
        all_gradients.insert(all_gradients.end(), grad_view_biases_.begin(), grad_view_biases_.end());
        all_gradients.push_back(grad_rgb_weights_);
        all_gradients.push_back(grad_rgb_bias_);
    } else {
        all_gradients.push_back(grad_rgb_weights_);
        all_gradients.push_back(grad_rgb_bias_);
    }

    return all_gradients;
}
void NerfNetwork::set_weights(const std::vector<std::shared_ptr<kp::TensorT<float>>> &weights) {
    // Check if we have the correct number of weights
    size_t expected_count = weights_.size() + biases_.size();
    if (params_.use_viewdirs) {
        expected_count += 4 + view_weights_.size() + view_biases_.size() + 2;
    } else {
        expected_count += 2;
    }

    if (weights.size() != expected_count) {
        throw std::runtime_error("Incorrect number of weights provided");
    }

    // Copy weights
    size_t idx = 0;

    // Main MLP weights
    for (size_t i = 0; i < weights_.size(); i++) {
        if (weights[idx]->size() != weights_[i]->size()) {
            throw std::runtime_error("Weight size mismatch");
        }
        std::copy(weights[idx]->data(), weights[idx]->data() + weights_[i]->size(), weights_[i]->data());
        idx++;
    }

    // Main MLP biases
    for (size_t i = 0; i < biases_.size(); i++) {
        if (weights[idx]->size() != biases_[i]->size()) {
            throw std::runtime_error("Bias size mismatch");
        }
        std::copy(weights[idx]->data(), weights[idx]->data() + biases_[i]->size(), biases_[i]->data());
        idx++;
    }

    if (params_.use_viewdirs) {
        // View-dependent model weights
        if (weights[idx]->size() != feature_weights_->size()) {
            throw std::runtime_error("Feature weight size mismatch");
        }
        std::copy(weights[idx]->data(), weights[idx]->data() + feature_weights_->size(), feature_weights_->data());
        idx++;

        if (weights[idx]->size() != feature_bias_->size()) {
            throw std::runtime_error("Feature bias size mismatch");
        }
        std::copy(weights[idx]->data(), weights[idx]->data() + feature_bias_->size(), feature_bias_->data());
        idx++;

        if (weights[idx]->size() != alpha_weights_->size()) {
            throw std::runtime_error("Alpha weight size mismatch");
        }
        std::copy(weights[idx]->data(), weights[idx]->data() + alpha_weights_->size(), alpha_weights_->data());
        idx++;

        if (weights[idx]->size() != alpha_bias_->size()) {
            throw std::runtime_error("Alpha bias size mismatch");
        }
        std::copy(weights[idx]->data(), weights[idx]->data() + alpha_bias_->size(), alpha_bias_->data());
        idx++;

        // View weights and biases
        for (size_t i = 0; i < view_weights_.size(); i++) {
            if (weights[idx]->size() != view_weights_[i]->size()) {
                throw std::runtime_error("View weight size mismatch");
            }
            std::copy(weights[idx]->data(), weights[idx]->data() + view_weights_[i]->size(), view_weights_[i]->data());
            idx++;
        }

        for (size_t i = 0; i < view_biases_.size(); i++) {
            if (weights[idx]->size() != view_biases_[i]->size()) {
                throw std::runtime_error("View bias size mismatch");
            }
            std::copy(weights[idx]->data(), weights[idx]->data() + view_biases_[i]->size(), view_biases_[i]->data());
            idx++;
        }

        if (weights[idx]->size() != rgb_weights_->size()) {
            throw std::runtime_error("RGB weight size mismatch");
        }
        std::copy(weights[idx]->data(), weights[idx]->data() + rgb_weights_->size(), rgb_weights_->data());
        idx++;

        if (weights[idx]->size() != rgb_bias_->size()) {
            throw std::runtime_error("RGB bias size mismatch");
        }
        std::copy(weights[idx]->data(), weights[idx]->data() + rgb_bias_->size(), rgb_bias_->data());
        idx++;
    } else {
        // Simple output weights
        if (weights[idx]->size() != rgb_weights_->size()) {
            throw std::runtime_error("Output weight size mismatch");
        }
        std::copy(weights[idx]->data(), weights[idx]->data() + rgb_weights_->size(), rgb_weights_->data());
        idx++;

        if (weights[idx]->size() != rgb_bias_->size()) {
            throw std::runtime_error("Output bias size mismatch");
        }
        std::copy(weights[idx]->data(), weights[idx]->data() + rgb_bias_->size(), rgb_bias_->data());
        idx++;
    }

    // Sync all weights to the device
    std::vector<std::shared_ptr<kp::Memory>> all_weights;
    all_weights.insert(all_weights.end(), weights_.begin(), weights_.end());
    all_weights.insert(all_weights.end(), biases_.begin(), biases_.end());

    if (params_.use_viewdirs) {
        all_weights.push_back(feature_weights_);
        all_weights.push_back(feature_bias_);
        all_weights.push_back(alpha_weights_);
        all_weights.push_back(alpha_bias_);
        all_weights.insert(all_weights.end(), view_weights_.begin(), view_weights_.end());
        all_weights.insert(all_weights.end(), view_biases_.begin(), view_biases_.end());
        all_weights.push_back(rgb_weights_);
        all_weights.push_back(rgb_bias_);
    } else {
        all_weights.push_back(rgb_weights_);
        all_weights.push_back(rgb_bias_);
    }

    manager_.sequence()
        ->record<kp::OpSyncDevice>(all_weights)
        ->eval();
}

} // namespace nerf