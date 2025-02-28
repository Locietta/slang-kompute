// nerf/dataset.cpp
#include "dataset.h"

#include <cmath>
#include <fstream>
#include <filesystem>
#include <numeric>
#include <numbers>
#include <stdexcept>
#include <glm/glm.hpp>

#include <nlohmann/json.hpp>
#include <fmt/core.h>
#include <fmt/std.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace nerf {

glm::mat4 Dataset::trans_t(float t) {
    return glm::mat4(
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, t,
        0.0f, 0.0f, 0.0f, 1.0f);
}

glm::mat4 Dataset::rot_phi(float phi) {
    float c = std::cos(phi);
    float s = std::sin(phi);
    return glm::mat4(
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, c, -s, 0.0f,
        0.0f, s, c, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f);
}

glm::mat4 Dataset::rot_theta(float theta) {
    float c = std::cos(theta);
    float s = std::sin(theta);
    return glm::mat4(
        c, 0.0f, -s, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        s, 0.0f, c, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f);
}

glm::mat4 Dataset::pose_spherical(float theta, float phi, float radius) {
    glm::mat4 c2w = trans_t(radius);
    c2w = rot_phi(phi / 180.0f * std::numbers::pi) * c2w;
    c2w = rot_theta(theta / 180.0f * std::numbers::pi) * c2w;

    // Switch axes
    glm::mat4 axes_switch(
        -1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f);
    c2w = axes_switch * c2w;

    return c2w;
}

Dataset Dataset::load_blender_data(const std::string &basedir, bool half_res, int testskip) {
    Dataset dataset;
    std::vector<std::string> splits = {"train", "val", "test"};
    std::map<std::string, json> metas;

    // Load JSON metadata
    for (const auto &split : splits) {
        const auto json_path = fs::path(basedir) / fmt::format("transforms_{}.json", split);
        std::ifstream file(json_path);
        if (!file.is_open()) {
            throw std::runtime_error(fmt::format("Failed to open file: {}", json_path));
        }
        metas[split] = json::parse(file);
    }

    std::vector<std::vector<float>> all_imgs;
    std::vector<glm::mat4> all_poses;
    std::vector<int> counts = {0};

    for (const auto &split : splits) {
        const auto &meta = metas[split];
        std::vector<std::vector<float>> imgs;
        std::vector<glm::mat4> poses;

        int skip = (split == "train" || testskip == 0) ? 1 : testskip;

        for (size_t i = 0; i < meta["frames"].size(); i += skip) {
            const auto &frame = meta["frames"][i];
            const auto file_path = fs::path(basedir) / (std::string(frame["file_path"]) + ".png");

            // Load image
            int w, h, channels;
            unsigned char *img = stbi_load(file_path.string().c_str(), &w, &h, &channels, 4); // Force RGBA
            if (!img) {
                throw std::runtime_error(fmt::format("Failed to load image: {}", file_path));
            }

            // Convert to float and normalize to [0, 1]
            std::vector<float> img_float(w * h * 4);
            for (int j = 0; j < w * h * 4; j++) {
                img_float[j] = static_cast<float>(img[j]) / 255.0f;
            }
            stbi_image_free(img);
            imgs.push_back(img_float);

            // Load transformation matrix
            glm::mat4 pose;
            const auto &transform = frame["transform_matrix"];
            for (int row = 0; row < 4; row++) {
                for (int col = 0; col < 4; col++) {
                    pose[col][row] = transform[row][col];
                }
            }
            poses.push_back(pose);
        }

        counts.push_back(counts.back() + static_cast<int>(imgs.size()));
        all_imgs.insert(all_imgs.end(), imgs.begin(), imgs.end());
        all_poses.insert(all_poses.end(), poses.begin(), poses.end());
    }

    // Set split indices
    dataset.train_indices.resize(counts[1] - counts[0]);
    dataset.val_indices.resize(counts[2] - counts[1]);
    dataset.test_indices.resize(counts[3] - counts[2]);

    std::iota(dataset.train_indices.begin(), dataset.train_indices.end(), counts[0]);
    std::iota(dataset.val_indices.begin(), dataset.val_indices.end(), counts[1]);
    std::iota(dataset.test_indices.begin(), dataset.test_indices.end(), counts[2]);

    // Concatenate images and poses
    dataset.images = all_imgs;
    dataset.poses = all_poses;

    // Calculate camera parameters
    if (!all_imgs.empty() && !all_imgs[0].empty()) {
        // Assuming square images and 4 channels (RGBA)
        int img_size = static_cast<int>(std::sqrt(all_imgs[0].size() / 4));
        dataset.height = img_size;
        dataset.width = img_size;
    } else {
        throw std::runtime_error("No images found in dataset");
    }

    float camera_angle_x = metas["train"]["camera_angle_x"];
    dataset.focal = 0.5f * dataset.width / std::tan(0.5f * camera_angle_x);

    // Generate render poses
    for (float angle = -180.0f; angle < 180.0f; angle += 360.0f / 40.0f) {
        dataset.render_poses.push_back(pose_spherical(angle, -30.0f, 4.0f));
    }

    // Apply half resolution if requested
    if (half_res) {
        dataset.height /= 2;
        dataset.width /= 2;
        dataset.focal /= 2.0f;

        // Resize images (use a simple box filter for downsampling)
        for (auto &img : dataset.images) {
            std::vector<float> img_half_res(dataset.height * dataset.width * 4);
            // Simple 2x2 box filter
            for (int y = 0; y < dataset.height; y++) {
                for (int x = 0; x < dataset.width; x++) {
                    for (int c = 0; c < 4; c++) {
                        float sum = 0.0f;
                        for (int dy = 0; dy < 2; dy++) {
                            for (int dx = 0; dx < 2; dx++) {
                                int orig_idx = ((2 * y + dy) * (dataset.width * 2) + (2 * x + dx)) * 4 + c;
                                sum += img[orig_idx];
                            }
                        }
                        img_half_res[(y * dataset.width + x) * 4 + c] = sum / 4.0f;
                    }
                }
            }
            img = img_half_res;
        }
    }

    return dataset;
}

} // namespace nerf