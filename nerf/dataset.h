// nerf/dataset.h
#pragma once

#include <string>
#include <vector>
#include <glm/glm.hpp>

namespace nerf {

class Dataset {
public:
    // Image data and poses
    std::vector<std::vector<float>> images;  // RGBA images flattened [N, H*W*4]
    std::vector<glm::mat4> poses;            // Camera to world transforms
    std::vector<glm::mat4> render_poses;     // Poses for rendering
    
    // Camera parameters
    int height;
    int width;
    float focal;
    
    // Dataset split indices
    std::vector<int> train_indices;
    std::vector<int> val_indices;
    std::vector<int> test_indices;

    // Constructor
    Dataset() = default;

    // Load Blender dataset
    static Dataset load_blender_data(const std::string& basedir, bool half_res = false, int testskip = 1);

private:
    // Helper functions for matrices
    static glm::mat4 trans_t(float t);
    static glm::mat4 rot_phi(float phi);
    static glm::mat4 rot_theta(float theta);
    static glm::mat4 pose_spherical(float theta, float phi, float radius);
};

} // namespace nerf