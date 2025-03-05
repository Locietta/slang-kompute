// nerf/volume_renderer.h (Simplified version)
#pragma once

#include <kompute/Kompute.hpp>
#include <memory>
#include <glm/glm.hpp>

namespace nerf {

struct VolumeRenderingParams {
    float raw_noise_std = 0.0f;    // Standard deviation of noise added to regularize sigma_a output
    bool white_background = false; // If true, use white background instead of transparent
};

class VolumeRenderer {
public:
    VolumeRenderer(kp::Manager &manager, const VolumeRenderingParams &params = {});
    ~VolumeRenderer() = default;

    // Render a batch of rays using raw network outputs
    // raw: Output from the network, shape [batch_size, n_samples, 4] (RGB+density)
    // z_vals: Depth values along each ray, shape [batch_size, n_samples]
    // rays_d: Ray directions, shape [batch_size, 3]
    void render(std::shared_ptr<kp::TensorT<float>> raw,
                std::shared_ptr<kp::TensorT<float>> z_vals,
                std::shared_ptr<kp::TensorT<float>> rays_d);

    void backward(std::shared_ptr<kp::TensorT<float>> grad_rgb,
                  std::shared_ptr<kp::TensorT<float>> grad_disp = nullptr,
                  std::shared_ptr<kp::TensorT<float>> grad_acc = nullptr);

    // Get the rendered RGB values
    std::shared_ptr<kp::TensorT<float>> get_rgb();

    // Get the disparity map (inverse depth)
    std::shared_ptr<kp::TensorT<float>> get_disparity();

    // Get the accumulated opacity (alpha) along each ray
    std::shared_ptr<kp::TensorT<float>> get_accumulated();

    // Get the weights for each sample
    std::shared_ptr<kp::TensorT<float>> get_weights();

    // Get the estimated depth map
    std::shared_ptr<kp::TensorT<float>> get_depth();

    std::shared_ptr<kp::TensorT<float>> get_grad_raw();
    
    // Synchronize all outputs to CPU memory
    void sync_outputs();

private:
    kp::Manager &manager_;
    VolumeRenderingParams params_;

    // Input tensors
    std::shared_ptr<kp::TensorT<float>> raw_;
    std::shared_ptr<kp::TensorT<float>> z_vals_;
    std::shared_ptr<kp::TensorT<float>> rays_d_;

    // Output tensors
    std::shared_ptr<kp::TensorT<float>> rgb_map_;
    std::shared_ptr<kp::TensorT<float>> disp_map_;
    std::shared_ptr<kp::TensorT<float>> acc_map_;
    std::shared_ptr<kp::TensorT<float>> weights_;
    std::shared_ptr<kp::TensorT<float>> depth_map_;

    // Noise tensor if needed
    std::shared_ptr<kp::TensorT<float>> noise_;

    // Single rendering algorithm
    std::shared_ptr<kp::Algorithm> render_algo_;

    // Initialize tensors
    void initialize_tensors(uint32_t batch_size, uint32_t n_samples);

    // Initialize algorithm
    void initialize_algorithm(uint32_t batch_size, uint32_t n_samples);

    std::shared_ptr<kp::TensorT<float>> grad_raw_;
    std::shared_ptr<kp::TensorT<float>> internal_grad_disp_;
    std::shared_ptr<kp::TensorT<float>> internal_grad_acc_;
};

} // namespace nerf