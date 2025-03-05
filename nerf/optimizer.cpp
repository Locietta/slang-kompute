// nerf/optimizer.cpp
#include "optimizer.h"
#include <stdexcept>
#include <cmath>

#include "utils.hpp"
#include "OpClear.h"

namespace nerf {

Adam::Adam(kp::Manager &manager, const AdamParams &params)
    : manager_(manager), params_(params) {
}

std::shared_ptr<kp::Algorithm> Adam::create_adam_algorithm(
    std::shared_ptr<kp::TensorT<float>> param,
    std::shared_ptr<kp::TensorT<float>> grad,
    std::shared_ptr<kp::TensorT<float>> m,
    std::shared_ptr<kp::TensorT<float>> v,
    uint32_t param_size,
    float beta1_t,
    float beta2_t) {

    // Load shader code
    constexpr auto adam_update_code = bytes_to_words(
#include "adam_update.spv.h"
    );
    std::vector<uint32_t> spirv(adam_update_code.begin(), adam_update_code.end());

    // Calculate workgroup size
    uint32_t workgroup_size = 64;
    uint32_t num_groups = divide_and_round_up(param_size, workgroup_size);

    // Push constants struct for shader
    struct PushConstantData {
        float learning_rate;
        float beta1;
        float beta2;
        float epsilon;
        float beta1_t;
        float beta2_t;
    };

    PushConstantData push_data = {
        params_.learning_rate,
        params_.beta1,
        params_.beta2,
        params_.epsilon,
        beta1_t,
        beta2_t};

    // Create algorithm
    return manager_.algorithm<uint32_t, PushConstantData>(
        {param, grad, m, v},
        spirv,
        kp::Workgroup({num_groups, 1, 1}),
        {param_size}, // Specialization constant
        {push_data});
}

void Adam::add_parameter(std::shared_ptr<kp::TensorT<float>> param) {
    // Create a zero-inited gradient tensor of the same size as the parameter
    auto grad = manager_.tensorT(std::vector<float>(param->size(), 0.0f));

    // Add parameter and gradient
    add_parameter_with_gradient(param, grad);
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
    initialize_parameter_state(parameters_.size() - 1);
}

void Adam::initialize_parameter_state(size_t param_idx) {
    const auto param_size = parameters_[param_idx]->size();

    // Create moment accumulators
    std::vector<float> zeros(param_size, 0.0f);
    auto m = manager_.tensorT<float>(zeros);
    auto v = manager_.tensorT<float>(zeros);

    // Initialize with zeros
    manager_.sequence()
        ->record<kp::OpSyncDevice>({m, v})
        ->eval();

    // Add to state lists
    exp_avg_.push_back(m);
    exp_avg_sq_.push_back(v);
}

void Adam::step() {
    step_count_++;

    // Calculate bias correction terms
    float beta1_t = std::pow(params_.beta1, step_count_);
    float beta2_t = std::pow(params_.beta2, step_count_);

    // Update each parameter
    for (size_t i = 0; i < parameters_.size(); i++) {
        const auto param_size = parameters_[i]->size();

        // Create algorithm for this specific parameter update
        auto algorithm = create_adam_algorithm(
            parameters_[i], gradients_[i], exp_avg_[i], exp_avg_sq_[i],
            param_size, beta1_t, beta2_t);

        // Execute the Adam update
        manager_.sequence()
            ->record<kp::OpSyncDevice>({parameters_[i], gradients_[i], exp_avg_[i], exp_avg_sq_[i]})
            ->record<kp::OpAlgoDispatch>(algorithm)
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