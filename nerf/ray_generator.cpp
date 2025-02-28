// nerf/ray_generator.cpp
#include "ray_generator.h"
#include "utils.hpp"
#include <fmt/core.h>

namespace nerf {

RayGenerator::RayGenerator(kp::Manager &manager, uint32_t width, uint32_t height, const glm::mat3 &intrinsic)
    : manager_(manager), width_(width), height_(height) {

    // Create tensors for ray data
    ray_origins_ = manager_.tensor(std::vector<float>(width_ * height_ * 3, 0.0f));
    ray_directions_ = manager_.tensor(std::vector<float>(width_ * height_ * 3, 0.0f));

    // Initialize camera parameters
    camera_params_.c2w = glm::mat4(1.0f); // Identity by default
    camera_params_.intrinsic = glm::mat4(0.0f);

    // Convert mat3 intrinsic to mat4 (padded with zeros)
    camera_params_.intrinsic[0][0] = intrinsic[0][0]; // fx
    camera_params_.intrinsic[0][2] = intrinsic[0][2]; // cx
    camera_params_.intrinsic[1][1] = intrinsic[1][1]; // fy
    camera_params_.intrinsic[1][2] = intrinsic[1][2]; // cy
    camera_params_.intrinsic[2][2] = intrinsic[2][2]; // 1.0
    camera_params_.intrinsic[3][3] = 1.0f;            // Homogeneous coordinate

    camera_params_.resolution = glm::vec2(width_, height_);
    camera_params_.padding = glm::vec2(0.0f);

    // Create parameters vector (only the output tensors)
    std::vector<std::shared_ptr<kp::Memory>> params = {
        ray_origins_, ray_directions_};

    // Load shader code
    constexpr auto k_cs_code = bytes_to_words(
#include "ray_generator.spv.h"
    );
    std::vector<uint32_t> spirv(k_cs_code.begin(), k_cs_code.end());

    // Calculate workgroup size
    uint32_t workgroup_size_x = 8;
    uint32_t workgroup_size_y = 8;
    uint32_t dispatch_x = divide_and_round_up(width_, workgroup_size_x);
    uint32_t dispatch_y = divide_and_round_up(height_, workgroup_size_y);

    // Create algorithm with push constants
    ray_generator_algo_ = manager_.algorithm<float, CameraParams>(
        params, spirv, kp::Workgroup({dispatch_x, dispatch_y, 1}), {}, {camera_params_});
}

void RayGenerator::generate_rays(const glm::mat4 &c2w) {
    // Update camera matrix in push constants
    camera_params_.c2w = c2w;

    // Create a new sequence for ray generation and execute it
    manager_.sequence()
        ->record<kp::OpAlgoDispatch>(ray_generator_algo_, std::vector{camera_params_})
        ->record<kp::OpSyncLocal>({ray_origins_, ray_directions_})
        ->eval();
}

std::shared_ptr<kp::TensorT<float>> RayGenerator::get_ray_origins() {
    return ray_origins_;
}

std::shared_ptr<kp::TensorT<float>> RayGenerator::get_ray_directions() {
    return ray_directions_;
}

} // namespace nerf