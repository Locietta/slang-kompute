#include <vector>
#include <string>
#include <fmt/core.h>

// Kompute headers
#include "kompute/Kompute.hpp"
#include "utils.hpp"
#include "nerf_dataset.h"
#include "nerf.h"

int main(int argc, char **argv) {
    try {
        // Parse command line arguments
        std::string dataset_path = "nerf/nerf_synthetic";// Default path
        std::string scene_name = "lego";                 // Default scene

        if (argc > 1) {
            dataset_path = argv[1];
        }

        if (argc > 2) {
            scene_name = argv[2];
        }

        fmt::println("Loading NeRF dataset from: {}", dataset_path);
        fmt::println("Scene: {}", scene_name);

        // Load dataset
        NeRFDataset dataset(dataset_path, scene_name);
        const auto &camera_params = dataset.get_camera_params();
        const auto &test_transforms = dataset.get_test_transforms();

        if (test_transforms.empty()) {
            throw std::runtime_error("No test transforms found in dataset");
        }

        // Initialize NeRF
        KomputeNeRF nerf;

        // Set up camera parameters from dataset
        int width = camera_params.width;
        int height = camera_params.height;
        float focal = camera_params.focal;

        // Render a test view
        const auto &test_view = test_transforms[0];// Use first test view
        std::vector<float> camera_pose = dataset.flatten_transform_matrix(test_view);

        fmt::println("Rendering test view with dimensions: {}x{}", width, height);
        fmt::println("Focal length: {}", focal);

        auto image = nerf.render_image(camera_pose, height, width, focal);

        // Save image
        std::string output_file = fmt::format("{}_nerf_render.png", scene_name);
        fmt::println("Saving rendered image to: {}", output_file);

        if (nerf.save_image(image, width, height, output_file)) {
            fmt::println("Image saved successfully");
        } else {
            fmt::println("Failed to save image");
        }

    } catch (const std::exception &e) {
        fmt::print(stderr, "Error: {}\n", e.what());
        return 1;
    }

    return 0;
}