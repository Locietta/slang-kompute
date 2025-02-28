// nerf/main.cpp
#include <iostream>
#include <kompute/Kompute.hpp>
#include <fmt/core.h>
#include "dataset.h"

int main(int argc, char *argv[]) {
    try {
        // Create Kompute Manager
        kp::Manager mgr;
        auto devices = mgr.listDevices();
        fmt::print("Device list:\n");
        for (size_t i = 0; i < devices.size(); ++i) {
            fmt::print("{}: {}\n", i, std::string_view(devices[i].getProperties().deviceName));
        }

        // Parse command line arguments
        std::string dataset_path = "data/nerf_synthetic/lego";
        bool half_res = true;

        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--dataset" && i + 1 < argc) {
                dataset_path = argv[++i];
            } else if (arg == "--half_res") {
                half_res = true;
            } else if (arg == "--full_res") {
                half_res = false;
            }
        }

        // Load dataset
        fmt::print("Loading NeRF dataset from: {}\n", dataset_path);
        auto dataset = nerf::Dataset::load_blender_data(dataset_path, half_res);

        // Print dataset info
        fmt::print("Dataset loaded successfully:\n");
        fmt::print("  - Images: {}\n", dataset.images.size());
        fmt::print("  - Poses: {}\n", dataset.poses.size());
        fmt::print("  - Render poses: {}\n", dataset.render_poses.size());
        fmt::print("  - Image dimensions: {} x {}\n", dataset.width, dataset.height);
        fmt::print("  - Focal length: {}\n", dataset.focal);
        fmt::print("  - Train/Val/Test split: {}/{}/{}\n",
                   dataset.train_indices.size(),
                   dataset.val_indices.size(),
                   dataset.test_indices.size());

        fmt::print("NeRF dataset loaded successfully!\n");

        // TODO: Implement NeRF model and training loop

        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}