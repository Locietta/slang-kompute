// nerf/OpClear.cpp
#include "OpClear.h"
#include "utils.hpp"

namespace kp {

const static std::vector<uint32_t> k_spirv = [] {
    constexpr auto cs_code = bytes_to_words(
#include "buffer_clear.spv.h"
    );
    return std::vector(cs_code.begin(), cs_code.end());
}();

OpClear::OpClear(std::vector<std::shared_ptr<kp::Memory>> tensors, kp::Manager &manager, float clear_value)
    : manager_(manager), tensors_(tensors), clear_value_(clear_value) {}

void OpClear::record(const vk::CommandBuffer &command_buffer) {

    for (auto tensor : tensors_) {
        const auto workgroup_size = 64u;
        const auto num_workgroups = divide_and_round_up(tensor->size(), workgroup_size);

        auto algo = manager_.algorithm({tensor}, k_spirv, {num_workgroups, 1, 1}, {}, {clear_value_});
        // TODO: handle Image
        tensor->recordPrimaryMemoryBarrier(command_buffer, vk::AccessFlagBits::eTransferWrite,
                                           vk::AccessFlagBits::eShaderRead,
                                           vk::PipelineStageFlagBits::eTransfer,
                                           vk::PipelineStageFlagBits::eComputeShader);
        algo->setPushConstants(&clear_value_, 1, sizeof(float));
        algo->recordBindCore(command_buffer);
        algo->recordBindPush(command_buffer);
        algo->recordDispatch(command_buffer);
    }
}

} // namespace kp