// nerf/loss.h
#pragma once

#include <kompute/Kompute.hpp>
#include <memory>

namespace nerf {

class MSELoss {
public:
    MSELoss(kp::Manager &manager);
    ~MSELoss() = default;

    // Compute MSE loss between predicted and target tensors
    // Returns the loss value
    float compute(std::shared_ptr<kp::TensorT<float>> predicted,
                  std::shared_ptr<kp::TensorT<float>> target);

    // Compute MSE gradients for backpropagation
    void backward(std::shared_ptr<kp::TensorT<float>> predicted,
                  std::shared_ptr<kp::TensorT<float>> target,
                  std::shared_ptr<kp::TensorT<float>> grad_output);

private:
    kp::Manager &manager_;
    std::shared_ptr<kp::TensorT<float>> loss_value_;

    // Create algorithm for forward pass
    std::shared_ptr<kp::Algorithm> create_forward_algorithm(
        std::shared_ptr<kp::TensorT<float>> predicted,
        std::shared_ptr<kp::TensorT<float>> target,
        std::shared_ptr<kp::TensorT<float>> loss_output);

    // Create algorithm for backward pass
    std::shared_ptr<kp::Algorithm> create_backward_algorithm(
        std::shared_ptr<kp::TensorT<float>> predicted,
        std::shared_ptr<kp::TensorT<float>> target,
        std::shared_ptr<kp::TensorT<float>> grad_output);
};

} // namespace nerf