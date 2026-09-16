// SPDX-License-Identifier: BSL-1.0
#pragma once

#include "os/os_threading.h"
#include "illixr/vk/display_provider.hpp"

class monado_vulkan_display_provider : public ILLIXR::vulkan::display_provider
{
#if defined(__linux__) && !defined(__ANDROID__)
public:
    // Monado owns this mutex for the lifetime of its Vulkan device. Both the
    // renderer and FFmpeg/server submissions must use it for the shared queue.
    struct os_mutex *shared_queue_mutex = nullptr;
    void lock_queue(ILLIXR::vulkan::queue::queue_type type) override {
        if (type == ILLIXR::vulkan::queue::GRAPHICS && shared_queue_mutex) {
            os_mutex_lock(shared_queue_mutex);
        } else {
            ILLIXR::vulkan::display_provider::lock_queue(type);
        }
    }
    void unlock_queue(ILLIXR::vulkan::queue::queue_type type) override {
        if (type == ILLIXR::vulkan::queue::GRAPHICS && shared_queue_mutex) {
            os_mutex_unlock(shared_queue_mutex);
        } else {
            ILLIXR::vulkan::display_provider::unlock_queue(type);
        }
    }
#endif
};
