// nerf/ray_generator.h
#pragma once

#include <kompute/Kompute.hpp>
#include <glm/glm.hpp>
#include <memory>

namespace nerf {

// Make sure this matches the shader's struct exactly
struct CameraParams {
    glm::mat4 c2w;        // Camera to world transform
    glm::mat4 intrinsic;  // Camera intrinsic matrix (padded to mat4)
    glm::vec2 resolution; // Image resolution (width, height)
    glm::vec2 padding;    // Padding to ensure alignment
};

class RayGenerator {
public:
    RayGenerator(kp::Manager &manager, uint32_t width, uint32_t height, const glm::mat3 &intrinsic);
    ~RayGenerator() = default;

    // Generate rays for a given camera pose
    void generate_rays(const glm::mat4 &c2w);

    // Get ray origins and directions
    std::shared_ptr<kp::TensorT<float>> get_ray_origins();
    std::shared_ptr<kp::TensorT<float>> get_ray_directions();

    // Get ray origins and directions (with sync to CPU - for testing)
    std::shared_ptr<kp::TensorT<float>> get_ray_origins_sync();
    std::shared_ptr<kp::TensorT<float>> get_ray_directions_sync();

private:
    kp::Manager &manager_;
    uint32_t width_;
    uint32_t height_;

    // Ray data
    std::shared_ptr<kp::TensorT<float>> ray_origins_;
    std::shared_ptr<kp::TensorT<float>> ray_directions_;

    // Camera parameters as push constants
    CameraParams camera_params_;

    // Algorithm
    std::shared_ptr<kp::Algorithm> ray_generator_algo_;
};

} // namespace nerf