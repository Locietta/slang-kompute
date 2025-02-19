
#include <vector>
#include <bit>
#include <span>
#include <fmt/core.h>

#include <kompute/Kompute.hpp>

template<size_t N>
consteval auto bytes_to_words(std::span<const unsigned char, N> bytes) {
    static_assert(N % 4 == 0, "Byte array size must be a multiple of 4");

    std::array<uint32_t, N / 4> words;
    for (size_t i = 0; i < N; i += 4) {
        // Extract 4-byte chunk as a temporary array
        const auto chunk = std::array{bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]};
        words[i / 4] = std::bit_cast<uint32_t>(chunk);
    }
    return words;
}

int main() {
    constexpr uint32_t iterations = 100;
    float learning_rate = 0.1;

    kp::Manager mgr;

    std::shared_ptr<kp::TensorT<float>> x_i = mgr.tensor({0, 1, 1, 1, 1});
    std::shared_ptr<kp::TensorT<float>> x_j = mgr.tensor({0, 0, 0, 1, 1});

    std::shared_ptr<kp::TensorT<float>> y = mgr.tensor({0, 0, 0, 1, 1});

    std::shared_ptr<kp::TensorT<float>> w_in = mgr.tensor({0.001, 0.001});
    std::shared_ptr<kp::TensorT<float>> w_out_i = mgr.tensor({0, 0, 0, 0, 0});
    std::shared_ptr<kp::TensorT<float>> w_out_j = mgr.tensor({0, 0, 0, 0, 0});

    std::shared_ptr<kp::TensorT<float>> b_in = mgr.tensor({0});
    std::shared_ptr<kp::TensorT<float>> b_out = mgr.tensor({0, 0, 0, 0, 0});

    std::shared_ptr<kp::TensorT<float>> l_out = mgr.tensor({0, 0, 0, 0, 0});

    std::vector<std::shared_ptr<kp::Memory>> params = {x_i, x_j, y,
                                                       w_in, w_out_i, w_out_j,
                                                       b_in, b_out, l_out};

    mgr.sequence()->eval<kp::OpSyncDevice>(params);

    alignas(4) constexpr uint8_t k_cs_code[] = {
#include "compute.spv.h"
    };
    constexpr auto k_cs_code_dword = bytes_to_words(std::span(k_cs_code));
    std::vector<uint32_t> spirv(k_cs_code_dword.begin(), k_cs_code_dword.end());

    std::shared_ptr<kp::Algorithm> algorithm = mgr.algorithm(
        params, spirv, kp::Workgroup({5}), std::vector<float>({5.0}));

    std::shared_ptr<kp::Sequence> sq =
        mgr.sequence()
            ->record<kp::OpSyncDevice>({w_in, b_in})
            ->record<kp::OpAlgoDispatch>(algorithm)
            ->record<kp::OpSyncLocal>({w_out_i, w_out_j, b_out, l_out});

    // Iterate across all expected iterations
    for (size_t i = 0; i < iterations; i++) {
        sq->eval();

        for (size_t j = 0; j < b_out->size(); j++) {
            w_in->data()[0] -= learning_rate * w_out_i->data()[j];
            w_in->data()[1] -= learning_rate * w_out_j->data()[j];
            b_in->data()[0] -= learning_rate * b_out->data()[j];
        }
    }

    KP_LOG_WARN("Result wIn i: {}, wIn j: {}, bIn: {}",
                w_in->data()[0],
                w_in->data()[1],
                b_in->data()[0]);

    if (w_in->data()[0] > 0.01 || w_in->data()[1] < 1.0 || b_in->data()[0] > 0.0) {
        throw std::runtime_error("Result does not match");
    }
}
