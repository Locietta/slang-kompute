// nerf/loss.cpp
#include "loss.h"
#include "utils.hpp"
#include "OpClear.h"
#include <stdexcept>

namespace nerf {

// Load shader code upon program init
static const std::vector<uint32_t> k_spirv_forward = [] {
    constexpr auto mse_forward_code = bytes_to_words(
#include "mse_forward.spv.h"
    );
    return std::vector<uint32_t>(mse_forward_code.begin(), mse_forward_code.end());
}();

static const std::vector<uint32_t> k_spirv_backward = [] {
    constexpr auto mse_backward_code = bytes_to_words(
#include "mse_backward.spv.h"
    );
    return std::vector<uint32_t>(mse_backward_code.begin(), mse_backward_code.end());
}();

MSELoss::MSELoss(kp::Manager &manager) : manager_(manager) {
    // Create tensor to store loss value
    // Initialize to zero
    loss_value_ = manager_.tensorT(std::vector<float>{0.0f});
}

std::shared_ptr<kp::Algorithm> MSELoss::create_forward_algorithm(
    std::shared_ptr<kp::TensorT<float>> predicted,
    std::shared_ptr<kp::TensorT<float>> target,
    std::shared_ptr<kp::TensorT<float>> loss_output) {

    // Calculate workgroup size
    constexpr uint32_t workgroup_size = 256;
    const uint32_t num_elements = predicted->size();
    const uint32_t num_groups = divide_and_round_up(num_elements, workgroup_size);

    // Create algorithm
    return manager_.algorithm(
        {predicted, target, loss_output},
        k_spirv_forward,
        kp::Workgroup({num_groups, 1, 1}));
}

std::shared_ptr<kp::Algorithm> MSELoss::create_backward_algorithm(
    std::shared_ptr<kp::TensorT<float>> predicted,
    std::shared_ptr<kp::TensorT<float>> target,
    std::shared_ptr<kp::TensorT<float>> grad_output) {
    // Calculate workgroup size
    constexpr uint32_t workgroup_size = 64;
    const uint32_t num_elements = predicted->size();
    const uint32_t num_groups = divide_and_round_up(num_elements, workgroup_size);

    // Create algorithm
    return manager_.algorithm(
        {predicted, target, grad_output},
        k_spirv_backward,
        kp::Workgroup({num_groups, 1, 1}));
}

float MSELoss::compute(std::shared_ptr<kp::TensorT<float>> predicted,
                       std::shared_ptr<kp::TensorT<float>> target) {
    // Check tensor dimensions
    if (predicted->size() != target->size()) {
        throw std::runtime_error("Predicted and target tensors must have the same size");
    }

    // Create algorithm for this specific input/output
    auto algorithm = create_forward_algorithm(predicted, target, loss_value_);

    // Run MSE computation
    manager_.sequence()
        ->record<kp::OpSyncDevice>({predicted, target})
        ->record<kp::OpClear>({loss_value_}, 0.0f)
        ->record<kp::OpAlgoDispatch>(algorithm)
        ->record<kp::OpSyncLocal>({loss_value_})
        ->eval();

    // Return the loss value
    return loss_value_->data()[0];
}

void MSELoss::backward(std::shared_ptr<kp::TensorT<float>> predicted,
                       std::shared_ptr<kp::TensorT<float>> target,
                       std::shared_ptr<kp::TensorT<float>> grad_output) {
    // Check tensor dimensions
    if (predicted->size() != target->size() || predicted->size() != grad_output->size()) {
        throw std::runtime_error("Predicted, target, and gradient tensors must have the same size");
    }

    // Create algorithm for this specific input/output
    auto algorithm = create_backward_algorithm(predicted, target, grad_output);

    // Run gradient computation
    manager_.sequence()
        ->record<kp::OpSyncDevice>({predicted, target, grad_output})
        ->record<kp::OpAlgoDispatch>(algorithm)
        ->eval();
}

} // namespace nerf