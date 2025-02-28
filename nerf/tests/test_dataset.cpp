// nerf/tests/test_dataset.cpp
#include <string>
#include <filesystem>
#include <fmt/core.h>
#include "../dataset.h"

namespace fs = std::filesystem;

// Simple test assertion macro
#define TEST_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            fmt::print(stderr, "Assertion failed: {} at {}:{}\n", message, __FILE__, __LINE__); \
            return 1; \
        } \
    } while (0)

int main(int argc, char* argv[]) {
    try {
        // Get test data path (default or from command line)
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
        
        fmt::print("Testing dataset loader with path: {}\n", dataset_path);
        
        // Test dataset loading
        auto dataset = nerf::Dataset::load_blender_data(dataset_path, half_res);
        
        // Basic validation tests
        TEST_ASSERT(!dataset.images.empty(), "Dataset should contain images");
        TEST_ASSERT(!dataset.poses.empty(), "Dataset should contain poses");
        TEST_ASSERT(!dataset.render_poses.empty(), "Dataset should contain render poses");
        TEST_ASSERT(dataset.images.size() == dataset.poses.size(), 
                   "Number of images should match number of poses");
        
        // Test image dimensions
        TEST_ASSERT(dataset.width > 0, "Image width should be positive");
        TEST_ASSERT(dataset.height > 0, "Image height should be positive");
        
        // Test camera parameters
        TEST_ASSERT(dataset.focal > 0, "Focal length should be positive");
        
        // Test split indices
        TEST_ASSERT(!dataset.train_indices.empty(), "Train indices should not be empty");
        TEST_ASSERT(!dataset.val_indices.empty(), "Validation indices should not be empty");
        TEST_ASSERT(!dataset.test_indices.empty(), "Test indices should not be empty");
        
        // Test image data
        for (const auto& img : dataset.images) {
            TEST_ASSERT(img.size() == dataset.width * dataset.height * 4, 
                       "Image data size should match WxHx4");
            
            // Check if image data is in [0, 1] range
            for (float pixel : img) {
                TEST_ASSERT(pixel >= 0.0f && pixel <= 1.0f, 
                           "Image pixel values should be in range [0, 1]");
            }
        }
        
        // Test pose matrices
        for (const auto& pose : dataset.poses) {
            // Check if last row is [0,0,0,1] (homogeneous coordinates)
            TEST_ASSERT(pose[0][3] == 0.0f && pose[1][3] == 0.0f && 
                       pose[2][3] == 0.0f && pose[3][3] == 1.0f,
                       "Pose matrix should have proper homogeneous form");
        }
        
        fmt::print("All dataset tests passed successfully!\n");
        return 0;
    } catch (const std::exception& e) {
        fmt::print(stderr, "Test failed with exception: {}\n", e.what());
        return 1;
    }
}