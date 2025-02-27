#include "nerf_dataset.h"
#include <fstream>
#include <fmt/core.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

NeRFDataset::NeRFDataset(const std::string &base_path, const std::string &scene_name)
    : base_path_(base_path), scene_name_(scene_name) {

    // Construct paths
    scene_path_ = fs::path(base_path_) / scene_name_;
    transforms_train_path_ = scene_path_ / "transforms_train.json";
    transforms_test_path_ = scene_path_ / "transforms_test.json";
    transforms_val_path_ = scene_path_ / "transforms_val.json";

    // Verify paths exist
    if (!fs::exists(scene_path_)) {
        throw std::runtime_error(fmt::format("Scene path does not exist: {}", scene_path_.string()));
    }

    if (!fs::exists(transforms_train_path_)) {
        throw std::runtime_error(fmt::format("Training transforms file not found: {}", transforms_train_path_.string()));
    }

    // Load camera parameters and transforms
    load_transforms();
}

void NeRFDataset::load_transforms() {
    // Load train transforms
    std::ifstream train_file(transforms_train_path_);
    json train_json;
    train_file >> train_json;

    // Get camera parameters
    camera_params_.angle_x = train_json["camera_angle_x"];

    // Calculate approximate focal length based on angle
    if (train_json.contains("w") && train_json.contains("h")) {
        camera_params_.width = train_json["w"];
        camera_params_.height = train_json["h"];
    } else {
        // Default to standard size if not specified
        camera_params_.width = 800;
        camera_params_.height = 800;
    }

    // Calculate focal length using angle_x and width
    camera_params_.focal = 0.5 * camera_params_.width / tan(0.5 * camera_params_.angle_x);

    // Load transforms
    for (const auto &frame : train_json["frames"]) {
        ImageTransform transform;
        transform.file_path = frame["file_path"];

        // Convert file path if needed (remove "./" prefix)
        if (transform.file_path.starts_with("./")) {
            transform.file_path = transform.file_path.substr(2);
        }

        // Add scene path
        transform.file_path = (scene_path_ / transform.file_path).string() + ".png";

        if (frame.contains("rotation")) {
            transform.rotation = frame["rotation"];
        } else {
            transform.rotation = 0.0f;
        }

        // Load transform matrix
        auto matrix = frame["transform_matrix"];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                transform.transform_matrix[i][j] = matrix[i][j];
            }
        }

        train_transforms_.push_back(transform);
    }

    // Load test transforms similarly
    if (fs::exists(transforms_test_path_)) {
        std::ifstream test_file(transforms_test_path_);
        json test_json;
        test_file >> test_json;

        for (const auto &frame : test_json["frames"]) {
            ImageTransform transform;
            transform.file_path = frame["file_path"];

            // Convert file path if needed
            if (transform.file_path.starts_with("./")) {
                transform.file_path = transform.file_path.substr(2);
            }

            // Add scene path
            transform.file_path = (scene_path_ / transform.file_path).string() + ".png";

            if (frame.contains("rotation")) {
                transform.rotation = frame["rotation"];
            } else {
                transform.rotation = 0.0f;
            }

            // Load transform matrix
            auto matrix = frame["transform_matrix"];
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    transform.transform_matrix[i][j] = matrix[i][j];
                }
            }

            test_transforms_.push_back(transform);
        }
    }

    fmt::println("Loaded NeRF dataset: {}", scene_name_);
    fmt::println("  Camera angle_x: {}", camera_params_.angle_x);
    fmt::println("  Image dimensions: {}x{}", camera_params_.width, camera_params_.height);
    fmt::println("  Focal length: {}", camera_params_.focal);
    fmt::println("  Training transforms: {}", train_transforms_.size());
    fmt::println("  Test transforms: {}", test_transforms_.size());
}

std::vector<float> NeRFDataset::flatten_transform_matrix(const ImageTransform &transform) {
    std::vector<float> flattened;
    flattened.reserve(16);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            flattened.push_back(transform.transform_matrix[i][j]);
        }
    }

    return flattened;
}

bool NeRFDataset::load_image(ImageTransform &transform) {
    if (!fs::exists(transform.file_path)) {
        fmt::println("Warning: Image file not found: {}", transform.file_path);
        return false;
    }

    int width, height, channels;
    unsigned char *data = stbi_load(transform.file_path.c_str(), &width, &height, &channels, 3);

    if (!data) {
        fmt::println("Error: Failed to load image: {}", transform.file_path);
        return false;
    }

    // Store the image data
    transform.image_data.resize(width * height * 3);
    std::memcpy(transform.image_data.data(), data, width * height * 3);

    // Free the loaded data
    stbi_image_free(data);
    return true;
}

const CameraParams &NeRFDataset::get_camera_params() const {
    return camera_params_;
}

const std::vector<ImageTransform> &NeRFDataset::get_train_transforms() const {
    return train_transforms_;
}

const std::vector<ImageTransform> &NeRFDataset::get_test_transforms() const {
    return test_transforms_;
}