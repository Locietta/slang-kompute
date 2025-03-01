// nerf/ray_sampler.cpp
#include "ray_sampler.h"
#include "utils.hpp"
#include <random>

namespace nerf {

RaySampler::RaySampler(kp::Manager &manager,
                       std::shared_ptr<kp::TensorT<float>> ray_origins,
                       std::shared_ptr<kp::TensorT<float>> ray_directions,
                       uint32_t n_samples, bool perturb, float near, float far)
    : manager_(manager),
      ray_origins_(ray_origins),
      ray_directions_(ray_directions),
      gen_(std::random_device{}()),
      perturb_(perturb) {

    // Calculate number of rays
    num_rays_ = ray_origins_->size() / 3;

    // Create tensors for sample data
    sample_positions_ = manager_.tensor(std::vector<float>(num_rays_ * n_samples * 3, 0.0f));
    sample_directions_ = manager_.tensor(std::vector<float>(num_rays_ * n_samples * 3, 0.0f));
    sample_z_vals_ = manager_.tensor(std::vector<float>(num_rays_ * n_samples, 0.0f));

    // perturbation buffer
    std::vector<float> randoms(num_rays_ * n_samples);
    randoms_ = manager_.tensorT(randoms);

    // Initialize default sampling parameters
    sampling_params_.near = near;
    sampling_params_.far = far;
    sampling_params_.num_samples = n_samples;

    // Send input tensors and random seeds to device (they don't change)
    manager_.sequence()
        ->record<kp::OpSyncDevice>({ray_origins_, ray_directions_, randoms_})
        ->eval();

    // Create algorithm
    std::vector<std::shared_ptr<kp::Memory>> params = {
        ray_origins_, ray_directions_,
        sample_positions_, sample_directions_, sample_z_vals_,
        randoms_};

    // Load shader code
    constexpr auto k_cs_code = bytes_to_words(
#include "ray_sampler.spv.h"
    );
    std::vector<uint32_t> spirv(k_cs_code.begin(), k_cs_code.end());

    // Calculate workgroup size
    uint32_t workgroup_size = 64;
    uint32_t num_groups = divide_and_round_up(num_rays_, workgroup_size);

    // Create algorithm with push constants
    ray_sampler_algo_ = manager_.algorithm<uint32_t, SamplingParams>(
        params, spirv, kp::Workgroup({num_groups, 1, 1}), {uint32_t(perturb_)}, {sampling_params_});
}

void RaySampler::sample_points() {
    // Update sampling parameters
    const auto n_samples = sampling_params_.num_samples;

    if (perturb_) {
        auto dist = std::uniform_real_distribution<float>(1e-4, 1.0);
        for (auto i = 0zu; i < num_rays_; ++i) {
            // TODO: use low-discrepancy generator for more consistent sampling
            auto randoms_on_ray = std::vector<float>(n_samples);
            for (auto j = 0zu; j < n_samples - 2; ++j) {
                randoms_on_ray[j] = dist(gen_);
            }
            randoms_on_ray[n_samples - 2] = 0.0;
            randoms_on_ray[n_samples - 1] = 1.0;
            std::ranges::sort(randoms_on_ray);
            std::ranges::copy(randoms_on_ray, randoms_->data() + i * n_samples);
        }
    }

    // Run algorithm with push constants and get results
    manager_.sequence()
        ->record<kp::OpSyncDevice>({ray_origins_, ray_directions_, randoms_})
        ->record<kp::OpAlgoDispatch>(ray_sampler_algo_, std::vector{sampling_params_})
        ->eval();
}

std::shared_ptr<kp::TensorT<float>> RaySampler::get_sample_positions() {
    return sample_positions_;
}

std::shared_ptr<kp::TensorT<float>> RaySampler::get_sample_directions() {
    return sample_directions_;
}

std::shared_ptr<kp::TensorT<float>> RaySampler::get_sample_z_vals() {
    return sample_z_vals_;
}

std::shared_ptr<kp::TensorT<float>> RaySampler::get_sample_positions_sync() {
    manager_.sequence()
        ->record<kp::OpSyncLocal>({sample_positions_})
        ->eval();
    return sample_positions_;
}

std::shared_ptr<kp::TensorT<float>> RaySampler::get_sample_directions_sync() {
    manager_.sequence()
        ->record<kp::OpSyncLocal>({sample_directions_})
        ->eval();
    return sample_directions_;
}

std::shared_ptr<kp::TensorT<float>> RaySampler::get_sample_z_vals_sync() {
    manager_.sequence()
        ->record<kp::OpSyncLocal>({sample_z_vals_})
        ->eval();
    return sample_z_vals_;
}

} // namespace nerf