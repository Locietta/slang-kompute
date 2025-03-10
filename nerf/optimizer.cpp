// nerf/optimizer.cpp
#include "optimizer.h"
#include <stdexcept>
#include <cmath>

#include "utils.hpp"
#include "OpClear.h"

namespace nerf {

namespace impl {
// Push constants struct for shader
struct PushConstantData {
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    float beta1_t;
    float beta2_t;
};
} // namespace impl

const static std::vector<uint32_t> k_spirv_adam_update = [] {
    constexpr auto adam_update_code = bytes_to_words(
#include "adam_update.spv.h"
    );
    return std::vector<uint32_t>(adam_update_code.begin(), adam_update_code.end());
}();

Adam::Adam(kp::Manager &manager,
           const std::vector<std::shared_ptr<kp::TensorT<float>>> &params,
           const std::vector<std::shared_ptr<kp::TensorT<float>>> &grads,
           const AdamParams &super_params)
    : manager_(manager), super_params_(super_params), parameters_(params), gradients_(grads) {
    // Initialize moment accumulators & algorithms
    for (size_t i = 0; i < parameters_.size(); ++i) {
        initialize_parameter_moment(i);
        algorithms_[i] = create_adam_algorithm(parameters_[i], gradients_[i], exp_avg_[i], exp_avg_sq_[i],
                                               parameters_[i]->size(), 1.0f, 1.0f);
    }
}

std::shared_ptr<kp::Algorithm> Adam::create_adam_algorithm(
    std::shared_ptr<kp::TensorT<float>> param,
    std::shared_ptr<kp::TensorT<float>> grad,
    std::shared_ptr<kp::TensorT<float>> m,
    std::shared_ptr<kp::TensorT<float>> v,
    uint32_t param_size,
    float beta1_t,
    float beta2_t) {

    // Calculate workgroup size
    uint32_t workgroup_size = 64;
    uint32_t num_groups = divide_and_round_up(param_size, workgroup_size);

    impl::PushConstantData push_data = {
        super_params_.learning_rate,
        super_params_.beta1,
        super_params_.beta2,
        super_params_.epsilon,
        beta1_t,
        beta2_t};

    // Create algorithm
    return manager_.algorithm<uint32_t, impl::PushConstantData>(
        {param, grad, m, v},
        k_spirv_adam_update,
        kp::Workgroup({num_groups, 1, 1}),
        {param_size}, // Specialization constant
        {push_data});
}

void Adam::add_parameter_with_gradient(std::shared_ptr<kp::TensorT<float>> param,
                                       std::shared_ptr<kp::TensorT<float>> grad) {
    if (param->size() != grad->size()) {
        throw std::runtime_error("Parameter and gradient must have the same size");
    }

    // Add to parameter and gradient lists
    parameters_.push_back(param);
    gradients_.push_back(grad);

    // Initialize state for this parameter
    initialize_parameter_moment(parameters_.size() - 1);
}

void Adam::add_parameters_with_gradients(const std::vector<std::shared_ptr<kp::TensorT<float>>> &params,
                                         const std::vector<std::shared_ptr<kp::TensorT<float>>> &grads) {
    if (params.size() != grads.size()) {
        throw std::runtime_error("Parameter and gradient vectors must have the same size");
    }

    auto seq = manager_.sequence();

    // Add parameters and gradients to lists & initialize the moment
    for (size_t i = 0; i < params.size(); ++i) {
        if (params[i]->size() != grads[i]->size()) {
            throw std::runtime_error("Parameter and gradient must have the same size");
        }

        parameters_.push_back(params[i]);
        gradients_.push_back(grads[i]);

        const auto param_idx = parameters_.size() - 1;

        // Initialize the moment
        const auto param_size = parameters_[param_idx]->size();

        // Create moment accumulators
        auto m = manager_.tensorT<float>(param_size);
        auto v = manager_.tensorT<float>(param_size);

        // record clear
        seq->record<kp::OpClear>({m, v});

        // Add to state lists
        exp_avg_.push_back(m);
        exp_avg_sq_.push_back(v);
    }

    seq->eval();
}

void Adam::initialize_parameter_moment(size_t param_idx) {
    const auto param_size = parameters_[param_idx]->size();

    // Create moment accumulators
    auto m = manager_.tensorT<float>(param_size);
    auto v = manager_.tensorT<float>(param_size);

    // Initialize with zeros
    manager_.sequence()
        ->record<kp::OpClear>({m, v})
        ->eval();

    // Add to state lists
    exp_avg_.push_back(m);
    exp_avg_sq_.push_back(v);
}

void Adam::set_moments(const std::vector<std::shared_ptr<kp::TensorT<float>>> &first_moments,
                       const std::vector<std::shared_ptr<kp::TensorT<float>>> &second_moments) {
    // Verify that the input vectors match our parameter count
    if (first_moments.size() != parameters_.size() ||
        second_moments.size() != parameters_.size()) {
        throw std::runtime_error("Moment vector count doesn't match parameter count");
    }

    // Copy moment vectors
    for (size_t i = 0; i < parameters_.size(); ++i) {
        // Make sure dimensions match
        if (first_moments[i]->size() != parameters_[i]->size() ||
            second_moments[i]->size() != parameters_[i]->size()) {
            throw std::runtime_error("Moment vector dimension mismatch");
        }

        // Copy data (assuming tensors are already in CPU memory)
        std::memcpy(exp_avg_[i]->data(), first_moments[i]->data(),
                    parameters_[i]->size() * sizeof(float));
        std::memcpy(exp_avg_sq_[i]->data(), second_moments[i]->data(),
                    parameters_[i]->size() * sizeof(float));
    }

    // Sync data to GPU
    auto seq = manager_.sequence();
    for (auto i = 0zu; i < exp_avg_.size(); ++i) {
        seq->record<kp::OpSyncDevice>({exp_avg_[i]});
        seq->record<kp::OpSyncDevice>({exp_avg_sq_[i]});
    }
    seq->eval();
}

void Adam::step() {
    step_count_++;

    // Calculate bias correction terms
    float beta1_t = std::pow(super_params_.beta1, step_count_);
    float beta2_t = std::pow(super_params_.beta2, step_count_);

    // Update each parameter
    for (size_t i = 0; i < parameters_.size(); i++) {
        const auto param_size = parameters_[i]->size();

        // fill super params
        impl::PushConstantData push_data = {
            super_params_.learning_rate,
            super_params_.beta1,
            super_params_.beta2,
            super_params_.epsilon,
            beta1_t, beta2_t};

        // Execute the Adam update
        manager_.sequence()
            // ->record<kp::OpSyncDevice>({parameters_[i], gradients_[i], exp_avg_[i], exp_avg_sq_[i]})
            ->record<kp::OpAlgoDispatch>(algorithms_[i], std::vector<impl::PushConstantData>{push_data})
            ->eval();
    }
}

void Adam::reset() {
    step_count_ = 0;

    // Reset all moment accumulators to zero
    auto seq = manager_.sequence();
    for (size_t i = 0; i < parameters_.size(); i++) {
        seq->record<kp::OpClear>({exp_avg_[i], exp_avg_sq_[i]});
    }
    seq->eval();
}

} // namespace nerf