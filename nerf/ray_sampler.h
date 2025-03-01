// nerf/ray_sampler.h
#pragma once

#include <kompute/Kompute.hpp>
#include <memory>
#include <random>
#include <glm/glm.hpp>

namespace nerf {

// Make sure this matches the shader's struct exactly
struct SamplingParams {
    float near;           // Near plane distance
    float far;            // Far plane distance
    uint32_t num_samples; // Number of samples per ray
};

class RaySampler {
public:
    RaySampler(kp::Manager &manager,
               std::shared_ptr<kp::TensorT<float>> ray_origins,
               std::shared_ptr<kp::TensorT<float>> ray_directions,
               uint32_t n_samples = 16, bool perturb = true, float near = 2.0f, float far = 6.0f);
    ~RaySampler() = default;

    // Sample points along the rays with given parameters
    void sample_points();

    // Get sample positions, directions, and z-values
    std::shared_ptr<kp::TensorT<float>> get_sample_positions();
    std::shared_ptr<kp::TensorT<float>> get_sample_directions();
    std::shared_ptr<kp::TensorT<float>> get_sample_z_vals();

    // Get sample positions, directions, and z-values (with CPU sync - for testing)
    std::shared_ptr<kp::TensorT<float>> get_sample_positions_sync();
    std::shared_ptr<kp::TensorT<float>> get_sample_directions_sync();
    std::shared_ptr<kp::TensorT<float>> get_sample_z_vals_sync();

private:
    kp::Manager &manager_;
    uint32_t num_rays_;

    // Input ray data
    std::shared_ptr<kp::TensorT<float>> ray_origins_;
    std::shared_ptr<kp::TensorT<float>> ray_directions_;

    // Sample data
    std::shared_ptr<kp::TensorT<float>> sample_positions_;
    std::shared_ptr<kp::TensorT<float>> sample_directions_;
    std::shared_ptr<kp::TensorT<float>> sample_z_vals_;

    // Sampling parameters
    SamplingParams sampling_params_;

    // Random generator for perturbation
    std::mt19937 gen_;
    // random number buffer to upload perturbation values
    std::shared_ptr<kp::TensorT<float>> randoms_;

    // Algorithm
    std::shared_ptr<kp::Algorithm> ray_sampler_algo_;

    // Enable perturbation for ray sampling
    bool perturb_ = true;
};

} // namespace nerf