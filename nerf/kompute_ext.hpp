// nerf/kompute_ext.hpp
#pragma once

#include <span>
#include <kompute/Manager.hpp>
#include <access_private.hpp>

ACCESS_PRIVATE_FIELD(kp::Manager, std::shared_ptr<vk::PhysicalDevice>, mPhysicalDevice);       // NOLINT
ACCESS_PRIVATE_FIELD(kp::Manager, std::shared_ptr<vk::Device>, mDevice);                       // NOLINT
ACCESS_PRIVATE_FIELD(kp::Manager, std::vector<std::weak_ptr<kp::Memory>>, mManagedMemObjects); // NOLINT
ACCESS_PRIVATE_FIELD(kp::Manager, bool, mManageResources);                                     // NOLINT

namespace kp {

template <typename T>
std::shared_ptr<kp::TensorT<T>> tensorT( // NOLINT
    kp::Manager &manager,
    const std::span<T> &data,
    Memory::MemoryTypes tensor_type = Memory::MemoryTypes::eDevice) {

    KP_LOG_DEBUG("Kompute Manager tensor creation triggered");
    KP_LOG_DEBUG("Kompute TensorT filling constructor with data size {}",
                 data.size());

    auto physical_device = access_private::mPhysicalDevice(manager);
    auto device = access_private::mDevice(manager);
    auto manage_resources = access_private::mManageResources(manager);
    auto &managed_mem_objects = access_private::mManagedMemObjects(manager);

    std::shared_ptr<TensorT<T>> tensor{
        reinterpret_cast<TensorT<T> *>(new Tensor{
            physical_device,
            device,
            data.data(),
            static_cast<uint32_t>(data.size()),
            sizeof(T),
            Memory::dataType<T>(),
            tensor_type,
        }),
    };

    if (manage_resources) {
        managed_mem_objects.push_back(tensor);
    }

    return tensor;
}

} // namespace kp