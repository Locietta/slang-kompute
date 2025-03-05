// nerf/OpClear.cpp
#include "OpClear.h"

#include <bit>

namespace kp {

OpClear::OpClear(std::vector<std::shared_ptr<kp::Memory>> mems, float clear_value)
    : mems_(mems), clear_value_(clear_value) {}

void OpClear::record(const vk::CommandBuffer &command_buffer) {

    for (auto mem : mems_) {
        if (mem->type() == Memory::Type::eTensor) {
            const auto tensor = std::static_pointer_cast<Tensor>(mem);
            const auto buffer = tensor->getPrimaryBuffer();
            if (!buffer) {
                throw std::runtime_error("Tensor primary buffer not allocated");
            }
            command_buffer.fillBuffer(*buffer, 0, tensor->size() * sizeof(float), std::bit_cast<uint32_t>(clear_value_));

        } else if (mem->type() == Memory::Type::eImage) {
            // TODO: Image clear
            const auto image = std::static_pointer_cast<Image>(mem);
            const auto vk_image = image->getPrimaryImage();
            if (!vk_image) {
                throw std::runtime_error("Image primary image not allocated");
            }

            const auto vk_image_layout = image->getPrimaryImageLayout();
            const auto vk_clear_value = vk::ClearColorValue(std::array{
                clear_value_,
                clear_value_,
                clear_value_,
                1.0f,
            });

            const auto range = vk::ImageSubresourceRange{
                vk::ImageAspectFlagBits::eColor,
                0,
                1,
                0,
                1,
            };
            image->recordPrimaryImageBarrier(command_buffer,
                                             vk::AccessFlagBits::eMemoryRead,
                                             vk::AccessFlagBits::eMemoryWrite,
                                             vk::PipelineStageFlagBits::eTransfer,
                                             vk::PipelineStageFlagBits::eTransfer,
                                             vk::ImageLayout::eTransferDstOptimal);
            command_buffer.clearColorImage(*vk_image, vk_image_layout, vk_clear_value, range);
        } else {
            throw std::runtime_error("Kompute Memory unsupported memory type");
        }
    }
}

} // namespace kp