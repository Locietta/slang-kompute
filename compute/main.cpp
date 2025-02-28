
#include <vector>
#include <fmt/core.h>
#include <kompute/Kompute.hpp>

#include "utils.hpp"

int main() {
    constexpr uint32_t iterations = 100;
    float learning_rate = 0.5;

    kp::Manager mgr;

    auto devices = mgr.listDevices();
    fmt::print("Device list:\n");
    for (size_t i = 0; i < devices.size(); ++i) {
        fmt::print("{}: {}\n", i, std::string_view(devices[i].getProperties().deviceName));
    }

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

    constexpr auto k_cs_code = bytes_to_words(
#include "compute.spv.h"
    );
    std::vector<uint32_t> spirv(k_cs_code.begin(), k_cs_code.end());

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

        for (auto j = 0zu; j < b_out->size(); ++j) {
            w_in->data()[0] -= learning_rate * w_out_i->data()[j];
            w_in->data()[1] -= learning_rate * w_out_j->data()[j];
            b_in->data()[0] -= learning_rate * b_out->data()[j];
        }

        for (auto j = 0zu; j < l_out->size(); ++j) {
            fmt::println("Loss: {}", l_out->data()[j]);
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
