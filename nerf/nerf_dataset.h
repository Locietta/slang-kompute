#pragma once

#include <string>
#include <vector>
#include <array>
#include <filesystem>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

struct CameraParams {
    float angle_x;
    int width;
    int height;
    float focal;
};

struct ImageTransform {
    std::string file_path;
    float rotation;
    std::array<std::array<float, 4>, 4> transform_matrix;
    std::vector<uint8_t> image_data;// Optional: loaded image data
};

class NeRFDataset {
public:
    explicit NeRFDataset(const std::string &base_path, const std::string &scene_name);

    void load_transforms();
    std::vector<float> flatten_transform_matrix(const ImageTransform &transform);
    bool load_image(ImageTransform &transform);

    const CameraParams &get_camera_params() const;
    const std::vector<ImageTransform> &get_train_transforms() const;
    const std::vector<ImageTransform> &get_test_transforms() const;

private:
    std::string base_path_;
    std::string scene_name_;
    fs::path scene_path_;
    fs::path transforms_train_path_;
    fs::path transforms_test_path_;
    fs::path transforms_val_path_;

    CameraParams camera_params_;
    std::vector<ImageTransform> train_transforms_;
    std::vector<ImageTransform> test_transforms_;
};