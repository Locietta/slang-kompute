// nerf/OpClear.h
#pragma once

#include <kompute/Kompute.hpp>
#include <vector>

namespace kp {

struct OpClear : OpBase {
    OpClear(std::vector<std::shared_ptr<kp::Memory>> tensors, kp::Manager &manager, float clear_value = 0.0f);
    OpClear(const OpClear &) = delete;
    ~OpClear() override = default;

    void record(const vk::CommandBuffer &command_buffer) override;
    void preEval(const vk::CommandBuffer &) override {}
    void postEval(const vk::CommandBuffer &) override {}

private:
    kp::Manager &manager_;
    std::vector<std::shared_ptr<kp::Memory>> tensors_; 
    float clear_value_ = 0.0f;
};

} // namespace kp