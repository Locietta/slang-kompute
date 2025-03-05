// nerf/trainer.h
#pragma once

#include <kompute/Kompute.hpp>
#include <memory>
#include <string>
#include "nerf_network.h"
#include "optimizer.h"
#include "loss.h"
#include "dataset.h"
#include "ray_generator.h"
#include "ray_sampler.h"
#include "volume_renderer.h"

namespace nerf {

struct TrainingConfig {
    uint32_t batch_size = 1024;
    uint32_t n_samples_per_ray = 64;
    uint32_t n_samples_hierarchical = 64;
    uint32_t num_iterations = 200000;
    uint32_t display_interval = 500;
    uint32_t save_interval = 5000;
    uint32_t validation_interval = 2000;
    uint32_t testset_interval = 5000;

    float near = 2.0f;
    float far = 6.0f;

    bool use_hierarchical_sampling = true;
    bool white_background = true;

    std::string save_dir = "./checkpoints";
    std::string log_dir = "./logs";
};

class Trainer {
public:
    Trainer(kp::Manager &manager,
            Dataset &dataset,
            const TrainingConfig &config = {});
    ~Trainer() = default;

    // Run training loop
    void train();

    // Evaluate on test set
    void evaluate();

    // Save/load checkpoint
    void save_checkpoint(const std::string &path);
    void load_checkpoint(const std::string &path);

private:
    kp::Manager &manager_;
    Dataset &dataset_;
    TrainingConfig config_;

    // Network and optimization components
    std::unique_ptr<NerfNetwork> coarse_network_;
    std::unique_ptr<NerfNetwork> fine_network_; // Optional for hierarchical sampling
    std::unique_ptr<Adam> optimizer_;
    std::unique_ptr<MSELoss> loss_fn_;

    // Rendering components
    std::unique_ptr<RayGenerator> ray_generator_;
    std::unique_ptr<RaySampler> coarse_sampler_;
    std::unique_ptr<RaySampler> fine_sampler_; // For hierarchical sampling
    std::unique_ptr<VolumeRenderer> volume_renderer_;

    // Training state
    uint32_t current_iteration_ = 0;
    float best_validation_loss_ = std::numeric_limits<float>::max();

    // Buffers for batch rendering
    std::shared_ptr<kp::TensorT<float>> ray_batch_origins_;
    std::shared_ptr<kp::TensorT<float>> ray_batch_directions_;
    std::shared_ptr<kp::TensorT<float>> target_pixels_;

    // Helper methods
    void initialize_components();
    void prepare_ray_batch(uint32_t image_idx);
    float train_one_iteration();
    float validate();
    void render_test_view(uint32_t view_idx);
    void log_training_stats(float loss);
    void save_image(const std::string &filename,
                    const std::vector<float> &image_data,
                    int width,
                    int height);
};

} // namespace nerf