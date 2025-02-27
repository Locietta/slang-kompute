#include "nerf.h"

#include <cmath>
#include <algorithm>
#include <random>
#include <fmt/core.h>
#include <fmt/ranges.h>

#include "utils.hpp"
#include "kompute/Kompute.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

KomputeNeRF::KomputeNeRF(int pos_enc_dim, int hidden_dim)
    : pos_enc_dim_(pos_enc_dim), hidden_dim_(hidden_dim) {
    initialize_kompute();
    create_network();
}

KomputeNeRF::~KomputeNeRF() {
    // Kompute resources are cleared automatically
}

std::vector<uint8_t> KomputeNeRF::render_image(const std::vector<float> &camera_pose, int height, int width, float focal) {
    // Generate camera rays
    std::vector<float> origins;
    std::vector<float> directions;

    // Camera parameters
    float aspect_ratio = static_cast<float>(width) / height;

    // Convert flattened matrix back to 4x4
    std::array<std::array<float, 4>, 4> camera_matrix;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            camera_matrix[i][j] = camera_pose[i * 4 + j];
        }
    }

    // Extract camera position (translation part of the inverse transform)
    float cam_x = camera_matrix[0][3];
    float cam_y = camera_matrix[1][3];
    float cam_z = camera_matrix[2][3];

    // Generate rays for each pixel
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            // Calculate normalized device coordinates
            float ndc_x = (j + 0.5f) / width * 2.0f - 1.0f;
            float ndc_y = 1.0f - (i + 0.5f) / height * 2.0f;// Flip Y for image coordinates

            // Convert to camera space rays
            float dir_x = ndc_x * aspect_ratio / focal;
            float dir_y = ndc_y / focal;
            float dir_z = -1.0f;// Forward direction

            // Transform to world space
            float world_dir_x =
                camera_matrix[0][0] * dir_x +
                camera_matrix[0][1] * dir_y +
                camera_matrix[0][2] * dir_z;
            float world_dir_y =
                camera_matrix[1][0] * dir_x +
                camera_matrix[1][1] * dir_y +
                camera_matrix[1][2] * dir_z;
            float world_dir_z =
                camera_matrix[2][0] * dir_x +
                camera_matrix[2][1] * dir_y +
                camera_matrix[2][2] * dir_z;

            // Normalize direction
            float norm = std::sqrt(world_dir_x * world_dir_x +
                                   world_dir_y * world_dir_y +
                                   world_dir_z * world_dir_z);
            world_dir_x /= norm;
            world_dir_y /= norm;
            world_dir_z /= norm;

            // Add ray origin (camera position)
            origins.push_back(cam_x);
            origins.push_back(cam_y);
            origins.push_back(cam_z);

            // Add ray direction
            directions.push_back(world_dir_x);
            directions.push_back(world_dir_y);
            directions.push_back(world_dir_z);
        }
    }

    int num_rays = origins.size() / 3;

    // Sample points along rays
    std::vector<float> t_vals;
    float near_bound = 2.0f;
    float far_bound = 6.0f;
    int num_samples = 64;

    for (int i = 0; i < num_samples; i++) {
        float t = near_bound + (far_bound - near_bound) * i / (num_samples - 1);
        t_vals.push_back(t);
    }

    // Process rays in batches
    const int batch_size = 1024;
    std::vector<float> image_rgb(width * height * 3, 0.0f);

    for (int batch_start = 0; batch_start < num_rays; batch_start += batch_size) {
        int current_batch_size = std::min(batch_size, num_rays - batch_start);

        // Prepare batch data
        std::vector<float> batch_origins(origins.begin() + batch_start * 3,
                                         origins.begin() + (batch_start + current_batch_size) * 3);
        std::vector<float> batch_directions(directions.begin() + batch_start * 3,
                                            directions.begin() + (batch_start + current_batch_size) * 3);

        // Sample points along rays for this batch
        std::vector<float> batch_points;
        std::vector<float> batch_dirs;

        for (int i = 0; i < current_batch_size; i++) {
            float ox = batch_origins[i * 3];
            float oy = batch_origins[i * 3 + 1];
            float oz = batch_origins[i * 3 + 2];

            float dx = batch_directions[i * 3];
            float dy = batch_directions[i * 3 + 1];
            float dz = batch_directions[i * 3 + 2];

            for (float t : t_vals) {
                // Point = origin + t * direction
                float px = ox + t * dx;
                float py = oy + t * dy;
                float pz = oz + t * dz;

                batch_points.push_back(px);
                batch_points.push_back(py);
                batch_points.push_back(pz);

                // Direction (constant for each sample along the ray)
                batch_dirs.push_back(dx);
                batch_dirs.push_back(dy);
                batch_dirs.push_back(dz);
            }
        }

        // Render batch of points
        auto [rgb, sigma] = render_points(batch_points, batch_dirs);

        // Process results and accumulate for volume rendering
        for (int i = 0; i < current_batch_size; i++) {
            std::vector<float> ray_rgb(3, 0.0f);
            float transmittance = 1.0f;

            for (int j = 0; j < num_samples; j++) {
                int sample_idx = i * num_samples + j;
                float density = sigma[sample_idx];

                // Delta between samples (use constant for last sample)
                float delta = (j < num_samples - 1) ?
                                  (t_vals[j + 1] - t_vals[j]) :
                                  1e-3f;

                // Compute alpha (absorption probability)
                float alpha = 1.0f - std::exp(-density * delta);

                // Weight for this sample
                float weight = alpha * transmittance;

                // Accumulate color
                ray_rgb[0] += weight * rgb[sample_idx * 3];
                ray_rgb[1] += weight * rgb[sample_idx * 3 + 1];
                ray_rgb[2] += weight * rgb[sample_idx * 3 + 2];

                // Update transmittance
                transmittance *= (1.0f - alpha);

                // Early termination
                if (transmittance < 1e-4f) break;
            }

            // Store pixel color
            int pixel_idx = batch_start + i;
            image_rgb[pixel_idx * 3] = ray_rgb[0];
            image_rgb[pixel_idx * 3 + 1] = ray_rgb[1];
            image_rgb[pixel_idx * 3 + 2] = ray_rgb[2];
        }
    }

    // Convert to 8-bit RGB
    std::vector<uint8_t> image(width * height * 3);
    for (int i = 0; i < width * height * 3; i++) {
        image[i] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, image_rgb[i] * 255.0f)));
    }

    return image;
}

bool KomputeNeRF::save_image(const std::vector<uint8_t> &image_data, int width, int height, const std::string &filename) {
    return stbi_write_png(filename.c_str(), width, height, 3, image_data.data(), width * 3) != 0;
}

void KomputeNeRF::initialize_kompute() {
    // Initialize Kompute manager
    manager_ = std::make_shared<kp::Manager>();

    // Select GPU device
    auto physical_devices = manager_->listDevices();

    if (physical_devices.size() > 0) {
        fmt::println("Available devices:");
        for (size_t i = 0; i < physical_devices.size(); i++) {
            auto device_info = physical_devices[i].getProperties();
            fmt::println("Device {}: {}", i, std::string_view(device_info.deviceName));
        }
        physical_device_index_ = 0;// Use the first device by default
        fmt::println("Using device: {}", std::string_view(physical_devices[physical_device_index_].getProperties().deviceName));
    } else {
        throw std::runtime_error("No Vulkan compatible devices found");
    }

    // Create manager with selected device
    manager_ = std::make_shared<kp::Manager>(physical_device_index_);
}

void KomputeNeRF::create_network() {
    // Shader for NeRF neural network using Kompute

    // Compile shader
    constexpr auto nerf_spv = bytes_to_words(
#include "main.spv.h"
    );

    shader_spirv_ = std::vector(nerf_spv.begin(), nerf_spv.end());

    // Initialize random weights
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> normal_dist(0.0f, 1.0f);

    // Network dimensions
    int pos_enc_size = 3 * 2 * pos_enc_dim_;// Position encoding
    int dir_enc_size = 3 * 2 * pos_enc_dim_;// Direction encoding

    // Layer 1: Position encoding to hidden
    weights_l1_.resize(hidden_dim_ * pos_enc_size);
    bias_l1_.resize(hidden_dim_, 0.0f);

    for (auto &w : weights_l1_) {
        w = normal_dist(gen);
    }

    // Layer 2: Hidden to hidden
    weights_l2_.resize(hidden_dim_ * hidden_dim_);
    bias_l2_.resize(hidden_dim_, 0.0f);

    for (auto &w : weights_l2_) {
        w = normal_dist(gen);
    }

    // RGB output layer: Hidden + Direction to RGB
    weights_rgb_.resize(3 * (hidden_dim_ + dir_enc_size));
    bias_rgb_.resize(3, 0.0f);

    for (auto &w : weights_rgb_) {
        w = normal_dist(gen);
    }

    // Sigma output layer: Hidden to density
    weights_sigma_.resize(hidden_dim_);
    bias_sigma_.resize(1, 0.0f);

    for (auto &w : weights_sigma_) {
        w = normal_dist(gen);
    }
}

std::pair<std::vector<float>, std::vector<float>> KomputeNeRF::render_points(
    const std::vector<float> &points, const std::vector<float> &directions) {

    // Create Kompute tensors
    uint32_t num_points = points.size() / 3;

    auto pos_tensor = manager_->tensor(points);
    auto dir_tensor = manager_->tensor(directions);

    // Network weights tensors
    auto weights_l1_tensor = manager_->tensor(weights_l1_);
    auto bias_l1_tensor = manager_->tensor(bias_l1_);
    auto weights_l2_tensor = manager_->tensor(weights_l2_);
    auto bias_l2_tensor = manager_->tensor(bias_l2_);
    auto weights_rgb_tensor = manager_->tensor(weights_rgb_);
    auto bias_rgb_tensor = manager_->tensor(bias_rgb_);
    auto weights_sigma_tensor = manager_->tensor(weights_sigma_);
    auto bias_sigma_tensor = manager_->tensor(bias_sigma_);

    // Output tensors
    std::vector<float> output_rgb(num_points * 3, 0.f);
    std::vector<float> output_sigma(num_points, 0.f);

    auto rgb_tensor = manager_->tensor(output_rgb);
    auto sigma_tensor = manager_->tensor(output_sigma);

    // Create algorithm with shader
    std::vector<std::shared_ptr<kp::Memory>> tensors = {
        pos_tensor, dir_tensor,
        weights_l1_tensor, bias_l1_tensor,
        weights_l2_tensor, bias_l2_tensor,
        weights_rgb_tensor, bias_rgb_tensor,
        weights_sigma_tensor, bias_sigma_tensor,
        rgb_tensor, sigma_tensor};

    manager_->sequence()->eval<kp::OpSyncDevice>(tensors);

    // Push constants
    struct PushConstants {
        uint32_t num_points;
        uint32_t pos_enc_dim;
        uint32_t hidden_dim;
    };

    constexpr uint32_t workgroup_size = 64;// Same as [numthreads(64, 1, 1)] in your shader
    uint32_t num_workgroups = divide_and_round_up(num_points, workgroup_size);

    auto algorithm = manager_->algorithm<float, PushConstants>(
        tensors, shader_spirv_, kp::Workgroup{num_workgroups, 1, 1},
        {}, {{num_points, pos_enc_dim_, hidden_dim_}});

    // Record commands to a sequence
    auto seq = manager_->sequence()->record<kp::OpSyncDevice>({pos_tensor, dir_tensor,
                                                               weights_l1_tensor, bias_l1_tensor,
                                                               weights_l2_tensor, bias_l2_tensor,
                                                               weights_rgb_tensor, bias_rgb_tensor,
                                                               weights_sigma_tensor, bias_sigma_tensor})
                   ->record<kp::OpAlgoDispatch>(algorithm)
                   ->record<kp::OpSyncLocal>({rgb_tensor, sigma_tensor});

    // Submit and wait
    seq->eval();

    std::memcpy(output_rgb.data(), rgb_tensor->data(), output_rgb.size() * sizeof(float));
    std::memcpy(output_sigma.data(), sigma_tensor->data(), output_sigma.size() * sizeof(float));

    return {output_rgb, output_sigma};
}