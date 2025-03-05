// nerf/optimizer.h
#pragma once

#include <kompute/Kompute.hpp>
#include <memory>
#include <vector>

namespace nerf {

struct AdamParams {
    float learning_rate = 5e-4f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
};

class Adam {
public:
    Adam(kp::Manager &manager, const AdamParams &params = {});
    ~Adam() = default;

    // Add a parameter tensor to be optimized
    void add_parameter(std::shared_ptr<kp::TensorT<float>> param);

    // Add a parameter tensor and its gradient
    void add_parameter_with_gradient(std::shared_ptr<kp::TensorT<float>> param,
                                     std::shared_ptr<kp::TensorT<float>> grad);

    // Update parameters based on gradients
    void step();

    // Reset moment accumulators
    void reset();

    uint32_t get_step_count() const { return step_count_; }
    void set_step_count(uint32_t step_count) { step_count_ = step_count; }

    const std::vector<std::shared_ptr<kp::TensorT<float>>> &get_first_moment_vectors() const {
        return exp_avg_;
    }

    const std::vector<std::shared_ptr<kp::TensorT<float>>> &get_second_moment_vectors() const {
        return exp_avg_sq_;
    }

    void set_moments(const std::vector<std::shared_ptr<kp::TensorT<float>>> &first_moments,
                     const std::vector<std::shared_ptr<kp::TensorT<float>>> &second_moments);

private:
    kp::Manager &manager_;
    AdamParams params_;
    uint32_t step_count_ = 0;

    // Lists of parameters and their corresponding states
    std::vector<std::shared_ptr<kp::TensorT<float>>> parameters_;
    std::vector<std::shared_ptr<kp::TensorT<float>>> gradients_;
    std::vector<std::shared_ptr<kp::TensorT<float>>> exp_avg_;    // First moment (momentum)
    std::vector<std::shared_ptr<kp::TensorT<float>>> exp_avg_sq_; // Second moment (velocity)

    // Create algorithm for Adam update
    std::shared_ptr<kp::Algorithm> create_adam_algorithm(
        std::shared_ptr<kp::TensorT<float>> param,
        std::shared_ptr<kp::TensorT<float>> grad,
        std::shared_ptr<kp::TensorT<float>> m,
        std::shared_ptr<kp::TensorT<float>> v,
        uint32_t param_size,
        float beta1_t,
        float beta2_t);

    // Initialize a parameter's state
    void initialize_parameter_state(size_t param_idx);
};

} // namespace nerf