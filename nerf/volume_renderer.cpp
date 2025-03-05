// nerf/volume_renderer.cpp
#include "volume_renderer.h"
#include "utils.hpp"
#include <random>
#include <stdexcept>

namespace nerf {

const static std::vector<uint32_t> k_spirv = [] {
    constexpr auto volume_render_code = bytes_to_words(
#include "volume_render.spv.h"
    );

    return std::vector<uint32_t>(volume_render_code.begin(), volume_render_code.end());
}();

const static std::vector<uint32_t> k_backward_spirv = [] {
    constexpr auto volume_render_backward_code = bytes_to_words(
#include "volume_render_backward.spv.h"
    );

    return std::vector<uint32_t>(volume_render_backward_code.begin(), volume_render_backward_code.end());
}();

VolumeRenderer::VolumeRenderer(kp::Manager &manager, const VolumeRenderingParams &params)
    : manager_(manager), params_(params) {
}

void VolumeRenderer::initialize_tensors(uint32_t batch_size, uint32_t n_samples) {
    // Initialize output tensors with proper size
    rgb_map_ = manager_.tensorT<float>(batch_size * 3);
    disp_map_ = manager_.tensorT<float>(batch_size);
    acc_map_ = manager_.tensorT<float>(batch_size);
    weights_ = manager_.tensorT<float>(batch_size * n_samples);
    depth_map_ = manager_.tensorT<float>(batch_size);

    // Initialize noise tensor if needed
    if (params_.raw_noise_std > 0.0f) {
        std::vector<float> noise_data(batch_size * n_samples);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, params_.raw_noise_std);

        for (auto &n : noise_data) {
            n = dist(gen);
        }
        noise_ = manager_.tensorT(noise_data);
    } else {
        // Zero noise
        noise_ = manager_.tensorT<float>(batch_size * n_samples);
    }
}

void VolumeRenderer::initialize_algorithm(uint32_t batch_size, uint32_t n_samples) {
    // Common parameters for shaders
    uint32_t workgroup_size = 64;
    uint32_t ray_groups = divide_and_round_up(batch_size, workgroup_size);

    // Parameters for the unified renderer
    struct RenderParams {
        uint32_t batch_size;
        uint32_t n_samples;
        float noise_std;
        uint32_t white_bkgd;
    };

    RenderParams render_params = {
        batch_size,
        n_samples,
        params_.raw_noise_std,
        params_.white_background ? 1u : 0u};

    // Create the single rendering algorithm
    render_algo_ = manager_.algorithm<uint32_t, RenderParams>(
        {raw_, z_vals_, rays_d_, noise_, weights_, rgb_map_, disp_map_, acc_map_, depth_map_},
        k_spirv,
        kp::Workgroup({ray_groups, 1, 1}),
        {},
        {render_params});
}

/// assume raw, z_vals, rays_d are already on device
void VolumeRenderer::render(std::shared_ptr<kp::TensorT<float>> raw,
                            std::shared_ptr<kp::TensorT<float>> z_vals,
                            std::shared_ptr<kp::TensorT<float>> rays_d) {
    // Store input tensors
    raw_ = raw;
    z_vals_ = z_vals;
    rays_d_ = rays_d;

    // Get dimensions
    uint32_t batch_size = rays_d_->size() / 3;
    uint32_t n_samples = z_vals_->size() / batch_size;

    // Check input dimensions
    if (raw_->size() != batch_size * n_samples * 4) {
        throw std::runtime_error("Raw output has incorrect dimensions");
    }

    // Initialize tensors and algorithm if needed
    if (!rgb_map_ || rgb_map_->size() != batch_size * 3) {
        initialize_tensors(batch_size, n_samples);
    }
    initialize_algorithm(batch_size, n_samples);

    // Execute the unified rendering algorithm
    manager_.sequence()
        ->record<kp::OpAlgoDispatch>(render_algo_)
        ->eval();
}

void VolumeRenderer::backward(std::shared_ptr<kp::TensorT<float>> grad_rgb,
                              std::shared_ptr<kp::TensorT<float>> grad_disp,
                              std::shared_ptr<kp::TensorT<float>> grad_acc) {
    if (!rgb_map_) {
        throw std::runtime_error("Must call render() before backward()");
    }

    uint32_t batch_size = rays_d_->size() / 3;
    uint32_t n_samples = z_vals_->size() / batch_size;

    // Verify input dimensions
    if (grad_rgb->size() != batch_size * 3) {
        throw std::runtime_error("grad_rgb has incorrect dimensions");
    }

    // Initialize gradient tensors if needed
    if (!grad_raw_ || grad_raw_->size() != raw_->size()) {
        grad_raw_ = manager_.tensorT<float>(raw_->size());
    }

    // Handle optional inputs, create zero-filled tensors if not provided
    if (!grad_disp) {
        if (!internal_grad_disp_ || internal_grad_disp_->size() != batch_size) {
            std::vector<float> zeros(batch_size, 0.0f);
            internal_grad_disp_ = manager_.tensorT(zeros);
        }
        grad_disp = internal_grad_disp_;
    } else if (grad_disp->size() != batch_size) {
        throw std::runtime_error("grad_disp has incorrect dimensions");
    }

    if (!grad_acc) {
        if (!internal_grad_acc_ || internal_grad_acc_->size() != batch_size) {
            std::vector<float> zeros(batch_size, 0.0f);
            internal_grad_acc_ = manager_.tensorT(zeros);
        }
        grad_acc = internal_grad_acc_;
    } else if (grad_acc->size() != batch_size) {
        throw std::runtime_error("grad_acc has incorrect dimensions");
    }

    // Common parameters for shaders
    uint32_t workgroup_size = 64;
    uint32_t ray_groups = divide_and_round_up(batch_size, workgroup_size);

    // Parameters for the backward pass
    struct RenderParams {
        uint32_t batch_size;
        uint32_t n_samples;
        float noise_std;
        uint32_t white_bkgd;
    };

    RenderParams render_params = {
        batch_size,
        n_samples,
        params_.raw_noise_std,
        params_.white_background ? 1u : 0u};

    // Create the backward algorithm if needed
    auto backward_algo = manager_.algorithm<uint32_t, RenderParams>(
        {raw_, z_vals_, rays_d_, weights_, grad_rgb, grad_disp, grad_acc, grad_raw_},
        k_backward_spirv,
        kp::Workgroup({ray_groups, 1, 1}),
        {},
        {render_params});

    // Execute the backward algorithm
    manager_.sequence()
        ->record<kp::OpAlgoDispatch>(backward_algo)
        ->eval();
}

std::shared_ptr<kp::TensorT<float>> VolumeRenderer::get_grad_raw() {
    return grad_raw_;
}

std::shared_ptr<kp::TensorT<float>> VolumeRenderer::get_rgb() {
    return rgb_map_;
}

std::shared_ptr<kp::TensorT<float>> VolumeRenderer::get_disparity() {
    return disp_map_;
}

std::shared_ptr<kp::TensorT<float>> VolumeRenderer::get_accumulated() {
    return acc_map_;
}

std::shared_ptr<kp::TensorT<float>> VolumeRenderer::get_weights() {
    return weights_;
}

std::shared_ptr<kp::TensorT<float>> VolumeRenderer::get_depth() {
    return depth_map_;
}

void VolumeRenderer::sync_outputs() {
    manager_.sequence()
        ->record<kp::OpSyncLocal>({rgb_map_, disp_map_, acc_map_, weights_, depth_map_})
        ->eval();
}

} // namespace nerf