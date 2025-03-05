// nerf/trainer.cpp (Revised with your suggestions)
#include "trainer.h"
#include "OpClear.h"
#include <random>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <fmt/core.h>

#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
// Include the actual STB image write implementation
#include <stb/stb_image_write.h>
#endif

namespace nerf {

Trainer::Trainer(kp::Manager &manager, Dataset &dataset, const TrainingConfig &config)
    : manager_(manager), dataset_(dataset), config_(config) {
    // Create output directories if they don't exist
    std::filesystem::create_directories(config_.save_dir);
    std::filesystem::create_directories(config_.log_dir);

    // Initialize all components
    initialize_components();
}

void Trainer::initialize_components() {
    // Create coarse network with default NeRF parameters
    MLPParams mlp_params;
    mlp_params.D = 8;               // 8-layer MLP
    mlp_params.W = 256;             // 256 hidden units
    mlp_params.skips = 4;           // Skip connection at 4th layer
    mlp_params.use_viewdirs = true; // Use view directions
    mlp_params.input_ch = 63;       // Pos encoding: 3 + 3*2*10 = 63
    mlp_params.input_ch_views = 27; // Dir encoding: 3 + 3*2*4 = 27
    mlp_params.output_ch = 4;       // RGB + sigma

    coarse_network_ = std::make_unique<NerfNetwork>(manager_, mlp_params);
    coarse_network_->initialize_weights();

    // Create fine network if using hierarchical sampling
    if (config_.use_hierarchical_sampling) {
        fine_network_ = std::make_unique<NerfNetwork>(manager_, mlp_params);
        fine_network_->initialize_weights();
    }

    // Create optimizer
    AdamParams adam_params;
    optimizer_ = std::make_unique<Adam>(manager_, adam_params);

    // Add network parameters to optimizer
    for (auto &param : coarse_network_->get_weights()) {
        optimizer_->add_parameter(std::reinterpret_pointer_cast<kp::TensorT<float>>(param));
    }

    if (config_.use_hierarchical_sampling && fine_network_) {
        for (auto &param : fine_network_->get_weights()) {
            optimizer_->add_parameter(std::reinterpret_pointer_cast<kp::TensorT<float>>(param));
        }
    }

    // Create loss function
    loss_fn_ = std::make_unique<MSELoss>(manager_);

    // Create ray generator with camera intrinsics
    glm::mat3 intrinsic(1.0f);
    intrinsic[0][0] = dataset_.focal;
    intrinsic[1][1] = dataset_.focal;
    intrinsic[0][2] = dataset_.width / 2.0f;
    intrinsic[1][2] = dataset_.height / 2.0f;
    ray_generator_ = std::make_unique<RayGenerator>(manager_, dataset_.width, dataset_.height, intrinsic);

    // Create volume renderer
    VolumeRenderingParams vol_params;
    vol_params.white_background = config_.white_background;
    volume_renderer_ = std::make_unique<VolumeRenderer>(manager_, vol_params);

    // Create batch buffers
    ray_batch_origins_ = manager_.tensorT<float>(config_.batch_size * 3);
    ray_batch_directions_ = manager_.tensorT<float>(config_.batch_size * 3);
    target_pixels_ = manager_.tensorT<float>(config_.batch_size * 3); // RGB values

    // Create samplers (will be initialized in train_one_iteration)
    coarse_sampler_ = nullptr;
    fine_sampler_ = nullptr;
}

void Trainer::prepare_ray_batch(uint32_t image_idx) {
    // Generate rays for the selected camera
    ray_generator_->generate_rays(dataset_.poses[image_idx]);
    auto ray_origins = ray_generator_->get_ray_origins();
    auto ray_directions = ray_generator_->get_ray_directions();

    // Sync to CPU for sampling specific rays
    manager_.sequence()
        ->record<kp::OpSyncLocal>({ray_origins, ray_directions})
        ->eval();

    // Select random pixels/rays
    std::vector<uint32_t> indices;
    indices.reserve(dataset_.width * dataset_.height);
    for (uint32_t i = 0; i < dataset_.width * dataset_.height; i++) {
        indices.push_back(i);
    }

    // Shuffle indices
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    // Take the first batch_size indices
    const auto batch_size = std::min(config_.batch_size, static_cast<uint32_t>(indices.size()));

    // Copy the selected rays to the batch buffers
    std::vector<float> batch_origins(batch_size * 3);
    std::vector<float> batch_directions(batch_size * 3);
    std::vector<float> batch_targets(batch_size * 3);

    for (uint32_t i = 0; i < batch_size; i++) {
        const auto pixel_idx = indices[i];
        const auto row = pixel_idx / dataset_.width;
        const auto col = pixel_idx % dataset_.width;

        // Copy ray origin and direction
        for (uint32_t c = 0; c < 3; c++) {
            batch_origins[i * 3 + c] = ray_origins->data()[(row * dataset_.width + col) * 3 + c];
            batch_directions[i * 3 + c] = ray_directions->data()[(row * dataset_.width + col) * 3 + c];
        }

        // Copy target pixel color (RGB)
        const auto img_idx = image_idx;
        for (uint32_t c = 0; c < 3; c++) {
            batch_targets[i * 3 + c] = dataset_.images[img_idx][(row * dataset_.width + col) * 4 + c];
        }
    }

    // Create tensors directly from vectors (avoiding sync pattern)
    ray_batch_origins_ = manager_.tensorT(batch_origins);
    ray_batch_directions_ = manager_.tensorT(batch_directions);
    target_pixels_ = manager_.tensorT(batch_targets);
}

float Trainer::train_one_iteration() {
    // Select a random training image
    std::random_device rd;
    std::mt19937 g(rd());
    std::uniform_int_distribution<uint32_t> dist(0, dataset_.train_indices.size() - 1);
    const auto train_idx = dataset_.train_indices[dist(g)];

    // Prepare ray batch
    prepare_ray_batch(train_idx);

    // Create or recreate samplers
    if (!coarse_sampler_) {
        coarse_sampler_ = std::make_unique<RaySampler>(
            manager_, ray_batch_origins_, ray_batch_directions_,
            config_.n_samples_per_ray, true, config_.near, config_.far);
    }

    // Coarse sampling
    coarse_sampler_->sample_points();
    auto coarse_positions = coarse_sampler_->get_sample_positions();
    auto coarse_directions = coarse_sampler_->get_sample_directions();
    auto coarse_z_vals = coarse_sampler_->get_sample_z_vals();

    // Forward pass through coarse network
    coarse_network_->forward(coarse_positions, coarse_directions);
    auto coarse_output = coarse_network_->get_output();

    // Volume rendering
    volume_renderer_->render(coarse_output, coarse_z_vals, ray_batch_directions_);
    auto coarse_rgb = volume_renderer_->get_rgb();

    // Compute loss
    float loss = loss_fn_->compute(coarse_rgb, target_pixels_);

    // Backward pass - create gradient tensors
    auto coarse_rgb_grad = manager_.tensorT<float>(coarse_rgb->size());
    auto coarse_output_grad = manager_.tensorT<float>(coarse_output->size());

    // Initialize gradient tensors to zero using OpClear
    manager_.sequence()
        ->record<kp::OpClear>({coarse_rgb_grad, coarse_output_grad})
        ->eval();

    // Compute gradients for coarse rgb
    loss_fn_->backward(coarse_rgb, target_pixels_, coarse_rgb_grad);

    // TODO: Implement backward pass for volume renderer to populate coarse_output_grad

    // TODO: Implement backward pass for network

    // Update parameters
    optimizer_->step();

    return loss;
}

void Trainer::train() {
    fmt::print("Starting NeRF training for {} iterations...\n", config_.num_iterations);

    // Initialize log file
    std::ofstream log_file(config_.log_dir + "/training_log.csv");
    log_file << "iteration,loss,time_ms\n";

    // Training loop
    for (; current_iteration_ < config_.num_iterations; current_iteration_++) {
        auto start_time = std::chrono::high_resolution_clock::now();

        // Train one iteration
        float loss = train_one_iteration();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

        // Log training stats
        if (current_iteration_ % config_.display_interval == 0) {
            fmt::print("Iteration {}/{}: Loss = {:.6f}, Time = {} ms\n",
                       current_iteration_, config_.num_iterations, loss, duration);
            log_file << fmt::format("{},{:.6f},{}\n", current_iteration_, loss, duration);
            log_file.flush();
        }

        // Save checkpoint
        if (current_iteration_ % config_.save_interval == 0) {
            save_checkpoint(config_.save_dir + "/checkpoint_" + std::to_string(current_iteration_) + ".bin");
        }

        // Run validation
        if (current_iteration_ % config_.validation_interval == 0) {
            float val_loss = validate();
            fmt::print("Validation Loss = {:.6f}\n", val_loss);

            // Save best model
            if (val_loss < best_validation_loss_) {
                best_validation_loss_ = val_loss;
                save_checkpoint(config_.save_dir + "/best_model.bin");
            }
        }

        // Render test view
        if (current_iteration_ % config_.testset_interval == 0) {
            // Render a random test view
            std::random_device rd;
            std::mt19937 g(rd());
            std::uniform_int_distribution<uint32_t> dist(0, dataset_.test_indices.size() - 1);
            const auto test_idx = dataset_.test_indices[dist(g)];
            render_test_view(test_idx);
        }
    }

    log_file.close();
    fmt::print("Training completed!\n");
}

float Trainer::validate() {
    float total_loss = 0.0f;

    // Evaluate on validation set
    for (auto val_idx : dataset_.val_indices) {
        // Prepare ray batch for validation image
        prepare_ray_batch(val_idx);

        // Sample points
        coarse_sampler_->sample_points();
        auto val_positions = coarse_sampler_->get_sample_positions();
        auto val_directions = coarse_sampler_->get_sample_directions();
        auto val_z_vals = coarse_sampler_->get_sample_z_vals();

        // Forward pass
        coarse_network_->forward(val_positions, val_directions);
        auto val_output = coarse_network_->get_output();

        // Volume rendering
        volume_renderer_->render(val_output, val_z_vals, ray_batch_directions_);
        auto val_rgb = volume_renderer_->get_rgb();

        // Compute loss (only for this batch)
        float loss = loss_fn_->compute(val_rgb, target_pixels_);
        total_loss += loss;
    }

    // Return average loss
    return total_loss / dataset_.val_indices.size();
}

void Trainer::render_test_view(uint32_t view_idx) {
    // Generate rays for the entire test view
    ray_generator_->generate_rays(dataset_.poses[view_idx]);
    auto ray_origins = ray_generator_->get_ray_origins();
    auto ray_directions = ray_generator_->get_ray_directions();

    // Create output image buffer
    std::vector<float> image_data(dataset_.width * dataset_.height * 3, 0.0f);

    // Process in batches to handle memory constraints
    uint32_t batch_size = config_.batch_size;
    uint32_t total_rays = dataset_.width * dataset_.height;
    uint32_t num_batches = (total_rays + batch_size - 1) / batch_size;

    for (uint32_t batch = 0; batch < num_batches; batch++) {
        uint32_t start_idx = batch * batch_size;
        uint32_t end_idx = std::min((batch + 1) * batch_size, total_rays);
        uint32_t current_batch_size = end_idx - start_idx;

        // Extract batch of rays
        std::vector<float> batch_origins(current_batch_size * 3);
        std::vector<float> batch_directions(current_batch_size * 3);

        // Sync ray data to CPU
        manager_.sequence()
            ->record<kp::OpSyncLocal>({ray_origins, ray_directions})
            ->eval();

        for (uint32_t i = 0; i < current_batch_size; i++) {
            uint32_t ray_idx = start_idx + i;
            for (uint32_t c = 0; c < 3; c++) {
                batch_origins[i * 3 + c] = ray_origins->data()[ray_idx * 3 + c];
                batch_directions[i * 3 + c] = ray_directions->data()[ray_idx * 3 + c];
            }
        }

        // Create tensors for this batch
        auto batch_ray_origins = manager_.tensorT(batch_origins);
        auto batch_ray_directions = manager_.tensorT(batch_directions);

        // Create sampler for this batch
        RaySampler batch_sampler(
            manager_, batch_ray_origins, batch_ray_directions,
            config_.n_samples_per_ray, false, config_.near, config_.far);

        // Sample points
        batch_sampler.sample_points();
        auto positions = batch_sampler.get_sample_positions();
        auto directions = batch_sampler.get_sample_directions();
        auto z_vals = batch_sampler.get_sample_z_vals();

        // Forward pass
        coarse_network_->forward(positions, directions);
        auto output = coarse_network_->get_output();

        // Volume rendering
        volume_renderer_->render(output, z_vals, batch_ray_directions);
        auto rgb = volume_renderer_->get_rgb();

        // Sync RGB results to CPU
        manager_.sequence()
            ->record<kp::OpSyncLocal>({rgb})
            ->eval();

        // Copy results to image buffer
        for (uint32_t i = 0; i < current_batch_size; i++) {
            uint32_t ray_idx = start_idx + i;
            for (uint32_t c = 0; c < 3; c++) {
                image_data[ray_idx * 3 + c] = rgb->data()[i * 3 + c];
            }
        }

        fmt::print("Rendered batch {}/{} for test view {}\n", batch + 1, num_batches, view_idx);
    }

    // Save image
    std::string output_dir = config_.log_dir + "/test_views";
    std::filesystem::create_directories(output_dir);
    std::string filename = std::format("{}/view_{}_iter_{}.png", output_dir, view_idx, current_iteration_);

    save_image(filename, image_data, dataset_.width, dataset_.height);
    fmt::print("Saved rendered test view to {}\n", filename);
}

void Trainer::save_checkpoint(const std::string &path) {
    fmt::print("Saving checkpoint to {}...\n", path);

    // Create binary file for writing
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        fmt::print(stderr, "Error: Failed to open file {} for writing\n", path);
        return;
    }

    // Write magic header and version
    constexpr uint32_t magic = 0x4652454E; // "NERF" in ASCII
    constexpr uint32_t version = 1;
    file.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
    file.write(reinterpret_cast<const char *>(&version), sizeof(version));

    // Write training metadata
    file.write(reinterpret_cast<const char *>(&current_iteration_), sizeof(current_iteration_));
    file.write(reinterpret_cast<const char *>(&best_validation_loss_), sizeof(best_validation_loss_));

    // Get network parameters and sync them to CPU
    auto coarse_weights = coarse_network_->get_weights();
    manager_.sequence()->record<kp::OpSyncLocal>(coarse_weights)->eval();

    // Write number of parameters
    uint32_t num_params = coarse_weights.size();
    file.write(reinterpret_cast<const char *>(&num_params), sizeof(num_params));

    // Write each parameter tensor
    for (uint32_t i = 0; i < num_params; i++) {
        auto tensor = std::dynamic_pointer_cast<kp::TensorT<float>>(coarse_weights[i]);

        // Write tensor dimensions
        uint32_t tensor_size = tensor->size();
        file.write(reinterpret_cast<const char *>(&tensor_size), sizeof(tensor_size));

        // Write tensor data
        auto tensor_data = tensor->vector();
        file.write(reinterpret_cast<const char *>(tensor_data.data()),
                   tensor_size * sizeof(float));
    }

    // Save fine network if it exists
    bool has_fine_network = (fine_network_ != nullptr);
    file.write(reinterpret_cast<const char *>(&has_fine_network), sizeof(has_fine_network));

    if (has_fine_network) {
        auto fine_weights = fine_network_->get_weights();
        manager_.sequence()->record<kp::OpSyncLocal>(fine_weights)->eval();

        // Write number of fine network parameters
        uint32_t num_fine_params = fine_weights.size();
        file.write(reinterpret_cast<const char *>(&num_fine_params), sizeof(num_fine_params));

        // Write each fine network parameter tensor
        for (uint32_t i = 0; i < num_fine_params; i++) {
            auto tensor = std::dynamic_pointer_cast<kp::TensorT<float>>(fine_weights[i]);

            // Write tensor dimensions
            uint32_t tensor_size = tensor->size();
            file.write(reinterpret_cast<const char *>(&tensor_size), sizeof(tensor_size));

            // Write tensor data
            auto tensor_data = tensor->vector();
            file.write(reinterpret_cast<const char *>(tensor_data.data()),
                       tensor_size * sizeof(float));
        }
    }

    // Save optimizer state (step count and moment vectors)
    uint32_t optimizer_step_count = optimizer_->get_step_count();
    file.write(reinterpret_cast<const char *>(&optimizer_step_count), sizeof(optimizer_step_count));

    // Save optimizer moment vectors
    auto first_moments = optimizer_->get_first_moment_vectors();
    auto second_moments = optimizer_->get_second_moment_vectors();

    // Sync moment vectors to CPU
    std::vector<std::shared_ptr<kp::Memory>> all_moments;
    all_moments.insert(all_moments.end(), first_moments.begin(), first_moments.end());
    all_moments.insert(all_moments.end(), second_moments.begin(), second_moments.end());
    manager_.sequence()->record<kp::OpSyncLocal>(all_moments)->eval();

    // Write first moments
    uint32_t num_moment_tensors = first_moments.size();
    file.write(reinterpret_cast<const char *>(&num_moment_tensors), sizeof(num_moment_tensors));

    for (uint32_t i = 0; i < num_moment_tensors; i++) {
        auto tensor = std::dynamic_pointer_cast<kp::TensorT<float>>(first_moments[i]);

        // Write tensor dimensions
        uint32_t tensor_size = tensor->size();
        file.write(reinterpret_cast<const char *>(&tensor_size), sizeof(tensor_size));

        // Write tensor data
        auto tensor_data = tensor->vector();
        file.write(reinterpret_cast<const char *>(tensor_data.data()),
                   tensor_size * sizeof(float));
    }

    // Write second moments (should be same count as first moments)
    for (uint32_t i = 0; i < num_moment_tensors; i++) {
        auto tensor = std::dynamic_pointer_cast<kp::TensorT<float>>(second_moments[i]);

        // Write tensor dimensions
        uint32_t tensor_size = tensor->size();
        file.write(reinterpret_cast<const char *>(&tensor_size), sizeof(tensor_size));

        // Write tensor data
        auto tensor_data = tensor->vector();
        file.write(reinterpret_cast<const char *>(tensor_data.data()),
                   tensor_size * sizeof(float));
    }

    file.close();
    fmt::print("Checkpoint saved successfully!\n");
}

void Trainer::load_checkpoint(const std::string &path) {
    fmt::print("Loading checkpoint from {}...\n", path);

    // Open binary file for reading
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        fmt::print(stderr, "Error: Failed to open file {} for reading\n", path);
        return;
    }

    // Read and verify magic header and version
    uint32_t magic, version;
    file.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char *>(&version), sizeof(version));

    const uint32_t expected_magic = 0x4652454E; // "NERF" in ASCII
    if (magic != expected_magic) {
        fmt::print(stderr, "Error: Invalid checkpoint file format\n");
        return;
    }

    if (version != 1) {
        fmt::print(stderr, "Error: Unsupported checkpoint version {}\n", version);
        return;
    }

    // Read training metadata
    file.read(reinterpret_cast<char *>(&current_iteration_), sizeof(current_iteration_));
    file.read(reinterpret_cast<char *>(&best_validation_loss_), sizeof(best_validation_loss_));

    // Read number of parameters
    uint32_t num_params;
    file.read(reinterpret_cast<char *>(&num_params), sizeof(num_params));

    // Get coarse network weights
    auto coarse_weights = coarse_network_->get_weights();

    // Verify parameter count
    if (num_params != coarse_weights.size()) {
        fmt::print(stderr, "Error: Checkpoint has {} parameters, but network expects {}\n",
                   num_params, coarse_weights.size());
        return;
    }

    // Read each parameter tensor
    for (uint32_t i = 0; i < num_params; i++) {
        auto tensor = std::dynamic_pointer_cast<kp::TensorT<float>>(coarse_weights[i]);

        // Read tensor dimensions
        uint32_t tensor_size;
        file.read(reinterpret_cast<char *>(&tensor_size), sizeof(tensor_size));

        // Verify tensor size
        if (tensor_size != tensor->size()) {
            fmt::print(stderr, "Error: Parameter {} has size {}, but network expects {}\n",
                       i, tensor_size, tensor->size());
            return;
        }

        // Read tensor data
        std::vector<float> tensor_data(tensor_size);
        file.read(reinterpret_cast<char *>(tensor_data.data()), tensor_size * sizeof(float));

        // Copy data to tensor and sync to device
        std::memcpy(tensor->data(), tensor_data.data(), tensor_size * sizeof(float));
    }

    // Sync coarse network weights to device
    manager_.sequence()->record<kp::OpSyncDevice>(coarse_weights)->eval();

    // Check for fine network
    bool has_fine_network;
    file.read(reinterpret_cast<char *>(&has_fine_network), sizeof(has_fine_network));

    if (has_fine_network) {
        // Ensure fine network exists
        if (!fine_network_) {
            fmt::print(stderr, "Error: Checkpoint contains fine network, but none was initialized\n");
            return;
        }

        // Read number of fine network parameters
        uint32_t num_fine_params;
        file.read(reinterpret_cast<char *>(&num_fine_params), sizeof(num_fine_params));

        // Get fine network weights
        auto fine_weights = fine_network_->get_weights();

        // Verify parameter count
        if (num_fine_params != fine_weights.size()) {
            fmt::print(stderr, "Error: Checkpoint has {} fine parameters, but network expects {}\n",
                       num_fine_params, fine_weights.size());
            return;
        }

        // Read each fine network parameter tensor
        for (uint32_t i = 0; i < num_fine_params; i++) {
            auto tensor = std::dynamic_pointer_cast<kp::TensorT<float>>(fine_weights[i]);

            // Read tensor dimensions
            uint32_t tensor_size;
            file.read(reinterpret_cast<char *>(&tensor_size), sizeof(tensor_size));

            // Verify tensor size
            if (tensor_size != tensor->size()) {
                fmt::print(stderr, "Error: Fine parameter {} has size {}, but network expects {}\n",
                           i, tensor_size, tensor->size());
                return;
            }

            // Read tensor data
            std::vector<float> tensor_data(tensor_size);
            file.read(reinterpret_cast<char *>(tensor_data.data()), tensor_size * sizeof(float));

            // Copy data to tensor
            std::memcpy(tensor->data(), tensor_data.data(), tensor_size * sizeof(float));
        }

        // Sync fine network weights to device
        manager_.sequence()->record<kp::OpSyncDevice>(fine_weights)->eval();
    }

    // Read optimizer state
    uint32_t optimizer_step_count;
    file.read(reinterpret_cast<char *>(&optimizer_step_count), sizeof(optimizer_step_count));
    optimizer_->set_step_count(optimizer_step_count);

    // Read optimizer moment vectors
    uint32_t num_moment_tensors;
    file.read(reinterpret_cast<char *>(&num_moment_tensors), sizeof(num_moment_tensors));

    // Verify moment tensor count matches parameter count
    if (num_moment_tensors != coarse_weights.size() + (has_fine_network ? fine_network_->get_weights().size() : 0)) {
        fmt::print(stderr, "Error: Moment tensor count mismatch\n");
        return;
    }

    // Create vectors to hold the read moment data
    std::vector<std::shared_ptr<kp::TensorT<float>>> first_moments;
    std::vector<std::shared_ptr<kp::TensorT<float>>> second_moments;

    // Read first moments
    for (uint32_t i = 0; i < num_moment_tensors; i++) {
        // Read tensor dimensions
        uint32_t tensor_size;
        file.read(reinterpret_cast<char *>(&tensor_size), sizeof(tensor_size));

        // Create tensor to hold the data
        auto tensor = manager_.tensorT<float>(tensor_size);

        // Read tensor data
        std::vector<float> tensor_data(tensor_size);
        file.read(reinterpret_cast<char *>(tensor_data.data()), tensor_size * sizeof(float));

        // Copy data to tensor
        std::memcpy(tensor->data(), tensor_data.data(), tensor_size * sizeof(float));

        first_moments.push_back(tensor);
    }

    // Read second moments
    for (uint32_t i = 0; i < num_moment_tensors; i++) {
        // Read tensor dimensions
        uint32_t tensor_size;
        file.read(reinterpret_cast<char *>(&tensor_size), sizeof(tensor_size));

        // Create tensor to hold the data
        auto tensor = manager_.tensorT<float>(tensor_size);

        // Read tensor data
        std::vector<float> tensor_data(tensor_size);
        file.read(reinterpret_cast<char *>(tensor_data.data()), tensor_size * sizeof(float));

        // Copy data to tensor
        std::memcpy(tensor->data(), tensor_data.data(), tensor_size * sizeof(float));

        second_moments.push_back(tensor);
    }

    // Set the optimizer's moment vectors
    try {
        optimizer_->set_moments(first_moments, second_moments);
    } catch (const std::exception &e) {
        fmt::print(stderr, "Error setting optimizer moments: {}\n", e.what());
        return;
    }

    file.close();
    fmt::print("Checkpoint loaded successfully! Resuming from iteration {}\n", current_iteration_);
}

void Trainer::save_image(const std::string &filename,
                         const std::vector<float> &image_data,
                         int width,
                         int height) {
    std::vector<unsigned char> image_data_8bit(width * height * 3);

    for (int i = 0; i < width * height; i++) {
        for (int c = 0; c < 3; c++) {
            // Clamp to [0,1] then scale to [0,255]
            float val = std::max(0.0f, std::min(1.0f, image_data[i * 3 + c]));
            image_data_8bit[i * 3 + c] = static_cast<unsigned char>(val * 255.0f);
        }
    }

    // Save as PNG using stb_image_write
    int result = stbi_write_png(
        filename.c_str(),
        width,
        height,
        3, // RGB components
        image_data_8bit.data(),
        width * 3 // stride in bytes
    );

    if (!result) {
        fmt::print("Error: Failed to save image {}\n", filename);
    } else {
        fmt::print("Saved image: {}\n", filename);
    }
}

void Trainer::evaluate() {
    fmt::print("Evaluating on test set...\n");

    float total_psnr = 0.0f;
    float total_ssim = 0.0f;
    float total_lpips = 0.0f;

    for (auto test_idx : dataset_.test_indices) {
        // Render full test view
        render_test_view(test_idx);

        // TODO: Calculate metrics (PSNR, SSIM, LPIPS) against ground truth
        // This would require implementing image comparison functions

        fmt::print("Evaluated test view {}\n", test_idx);
    }

    // Print average metrics
    fmt::print("Test Results:\n");
    fmt::print("  Average PSNR: {:.4f}\n", total_psnr / dataset_.test_indices.size());
    fmt::print("  Average SSIM: {:.4f}\n", total_ssim / dataset_.test_indices.size());
    fmt::print("  Average LPIPS: {:.4f}\n", total_lpips / dataset_.test_indices.size());
}

} // namespace nerf