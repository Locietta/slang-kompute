// nerf/volume_renderer.cpp (Simplified version)
#include "volume_renderer.h"
#include "utils.hpp"
#include <random>
#include <stdexcept>

namespace nerf {

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
    // Create shader for unified volume rendering
    constexpr auto volume_render_code = bytes_to_words(
#include "volume_render.spv.h"
    );
    std::vector<uint32_t> render_spirv(volume_render_code.begin(), volume_render_code.end());

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
        render_spirv,
        kp::Workgroup({ray_groups, 1, 1}),
        {},
        {render_params});
}

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
        initialize_algorithm(batch_size, n_samples);
    }

    // Ensure inputs are on the device
    manager_.sequence()
        ->record<kp::OpSyncDevice>({raw_, z_vals_, rays_d_, noise_})
        ->eval();

    // Execute the unified rendering algorithm
    manager_.sequence()
        ->record<kp::OpAlgoDispatch>(render_algo_)
        ->eval();
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