#pragma once

#include <vector>
#include <memory>
#include <utility>

#include "kompute/Manager.hpp"

class KomputeNeRF {
public:
    KomputeNeRF(int pos_enc_dim = 10, int hidden_dim = 256);
    ~KomputeNeRF();

    // Render a full image from a specified camera pose
    std::vector<uint8_t> render_image(const std::vector<float> &camera_pose, int height, int width, float focal);

    // Save rendered image to a file
    bool save_image(const std::vector<uint8_t> &image_data, int width, int height, const std::string &filename);

private:
    // Kompute manager and resources
    std::shared_ptr<kp::Manager> manager_;
    uint32_t physical_device_index_ = 0;
    std::vector<uint32_t> shader_spirv_;

    // NeRF network parameters
    uint32_t pos_enc_dim_;
    uint32_t hidden_dim_;
    std::vector<float> weights_l1_;
    std::vector<float> bias_l1_;
    std::vector<float> weights_l2_;
    std::vector<float> bias_l2_;
    std::vector<float> weights_rgb_;
    std::vector<float> bias_rgb_;
    std::vector<float> weights_sigma_;
    std::vector<float> bias_sigma_;

    void initialize_kompute();
    void create_network();
    std::pair<std::vector<float>, std::vector<float>> render_points(
        const std::vector<float> &points, const std::vector<float> &directions);
};