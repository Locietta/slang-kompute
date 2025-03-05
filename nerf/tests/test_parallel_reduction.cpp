#include <kompute/Kompute.hpp>
#include <memory>
#include <random>
#include <chrono>
#include <fmt/chrono.h>
#include "../OpClear.h"
#include "../utils.hpp"

const static std::vector<uint32_t> k_spirv = [] {
    constexpr auto reduce_bytes = bytes_to_words(
#include "parallel_reduction.spv.h"
    );
    return std::vector<uint32_t>(reduce_bytes.begin(), reduce_bytes.end());
}();

class Reduce {
public:
    Reduce(kp::Manager &manager) : manager_(manager), result_(manager_.tensorT(std::vector<float>{0.0f})) {}
    ~Reduce() = default;

    // Compute MSE loss between predicted and target tensors
    // Returns the loss value
    float reduce(std::shared_ptr<kp::TensorT<float>> array) {
        auto algorithm = create_algorithm(array, result_);
        manager_.sequence()
            ->record<kp::OpSyncDevice>({array})
            ->record<kp::OpClear>({result_}, 0.0f)
            ->record<kp::OpAlgoDispatch>(algorithm)
            ->record<kp::OpSyncLocal>({result_})
            ->eval();

        return result_->data()[0];
    }

private:
    kp::Manager &manager_;
    std::shared_ptr<kp::TensorT<float>> result_;

    // Create algorithm for forward pass
    std::shared_ptr<kp::Algorithm> create_algorithm(
        std::shared_ptr<kp::TensorT<float>> array,
        std::shared_ptr<kp::TensorT<float>> result) {
        constexpr uint32_t workgroup_size = 256;
        const uint32_t num_elements = array->size();
        const uint32_t num_groups = divide_and_round_up(num_elements, workgroup_size);
        return manager_.algorithm(
            {array, result},
            k_spirv,
            kp::Workgroup({num_groups, 1, 1}));
    }
};

#define TEST_ASSERT(condition, message)                                                         \
    do {                                                                                        \
        if (!(condition)) {                                                                     \
            fmt::print(stderr, "Assertion failed: {} at {}:{}\n", message, __FILE__, __LINE__); \
            return 1;                                                                           \
        }                                                                                       \
    } while (0)

bool float_equals(float a, float b, float epsilon = 1e-6f) {
    return std::abs(a - b) / a < epsilon;
}

int main() {
    try {
        kp::Manager mgr(0, {}, {"VK_EXT_shader_atomic_float"});
        fmt::print("Initialized Kompute manager with device: {}\n",
                   std::string_view(mgr.getDeviceProperties().deviceName));

        Reduce op(mgr);

        std::vector<float> data{
            1.0f,
            2.0f,
            3.0f,
            4.0f,
            5.0f,
            6.0f,
            7.0f,
            8.0f,
            9.0f,
            10.0f,
            11.0f,
            12.0f,
            13.0f,
            14.0f,
            15.0f,
            16.0f,
            17.0f,
            18.0f,
            19.0f,
            20.0f,
            21.0f,
            22.0f,
            23.0f,
            24.0f,
            25.0f,
            26.0f,
            27.0f,
            28.0f,
            29.0f,
            30.0f,
            31.0f,
            32.0f,
            33.0f,
            34.0f,
            35.0f,
            36.0f,
            37.0f,
            38.0f,
            39.0f,
            40.0f,
            41.0f,
            42.0f,
            43.0f,
            44.0f,
            45.0f,
            46.0f,
            47.0f,
            48.0f,
            49.0f,
            50.0f,
            51.0f,
            52.0f,
            53.0f,
            54.0f,
            55.0f,
        };

        auto tensor = mgr.tensorT(data);
        float sum = op.reduce(tensor);
        fmt::print("Sum: {}\n", sum);

        TEST_ASSERT(float_equals(sum, 1540.0f), "Sum should be 55.0");

        /// random large array
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(1.0, 100.0);

        constexpr size_t large_size = 1024;
        std::vector<float> large_data(large_size);
        float expected_sum = 0.0f;
        for (size_t i = 0; i < large_size; ++i) {
            large_data[i] = dis(gen);
            expected_sum += large_data[i];
        }

        // timeit
        auto start = std::chrono::high_resolution_clock::now();

        auto large_tensor = mgr.tensorT(large_data);
        float large_sum = op.reduce(large_tensor);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        fmt::print("Time taken by function: {:%Q %q}\n", duration);

        fmt::print("Large sum: {}\n", large_sum);
        fmt::print("Expected large sum: {}\n", expected_sum);

        TEST_ASSERT(float_equals(large_sum, expected_sum), "Large sum should be close to expected sum");

    } catch (std::exception const &e) {
        fmt::print(stderr, "Exception caught: {}\n", e.what());
        return 1;
    }
}