// SPDX-License-Identifier: BSL-1.0
#pragma once

// Linux-only conversion resources. Kept separate from the Windows RG/motion
// vector path: FFmpeg expects grayscale RGBA, not raw D16 or packed RG depth.
#include "util/comp_swapchain.h"
#include "util/comp_render_helpers.h"
#include "../drivers/illixr/illixr_framebuffer.h"
#include "shaders/illixr_depth.comp.h"

struct illixr_linux_depth
{
    VkPipeline pipeline;
    VkPipelineLayout layout;
    VkDescriptorSetLayout descriptor_layout;
    VkDescriptorPool pool;
    VkSampler sampler;
    VkDescriptorSet sets[2 * OFFLOAD_BUFFER_POOL_SIZE];
    struct {
        VkImage image;
        VkImageView view;
        VkDeviceMemory memory;
    } images[2 * OFFLOAD_BUFFER_POOL_SIZE];
};

static void
illixr_linux_depth_fini(struct illixr_linux_depth *d, struct vk_bundle *vk)
{
    // Also handles partially completed initialization.
    if (d->pool) vk->vkDestroyDescriptorPool(vk->device, d->pool, NULL);
    if (d->pipeline) vk->vkDestroyPipeline(vk->device, d->pipeline, NULL);
    if (d->layout) vk->vkDestroyPipelineLayout(vk->device, d->layout, NULL);
    if (d->descriptor_layout) vk->vkDestroyDescriptorSetLayout(vk->device, d->descriptor_layout, NULL);
    if (d->sampler) vk->vkDestroySampler(vk->device, d->sampler, NULL);
    for (uint32_t i = 0; i < 2 * OFFLOAD_BUFFER_POOL_SIZE; i++) {
        if (d->images[i].view) vk->vkDestroyImageView(vk->device, d->images[i].view, NULL);
        if (d->images[i].image) vk->vkDestroyImage(vk->device, d->images[i].image, NULL);
        if (d->images[i].memory) vk->vkFreeMemory(vk->device, d->images[i].memory, NULL);
    }
    memset(d, 0, sizeof(*d));
}

static bool
illixr_linux_depth_init(struct illixr_linux_depth *d, struct vk_bundle *vk,
                       struct illixr_framebuffer *fb, uint32_t width, uint32_t height)
{
    VkShaderModule module = VK_NULL_HANDLE;
    VkResult ret = VK_SUCCESS;
#define DEPTH_CHECK(call) do { ret = (call); if (ret != VK_SUCCESS) goto fail; } while (0)
    VkDescriptorSetLayoutBinding bindings[] = {
        {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
         .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
    };
    VkDescriptorSetLayoutCreateInfo dl = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 2, .pBindings = bindings,
    };
    DEPTH_CHECK(vk->vkCreateDescriptorSetLayout(vk->device, &dl, NULL, &d->descriptor_layout));
    VkPushConstantRange push = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .size = 4 * sizeof(float)};
    VkPipelineLayoutCreateInfo pl = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &d->descriptor_layout,
        .pushConstantRangeCount = 1, .pPushConstantRanges = &push,
    };
    DEPTH_CHECK(vk->vkCreatePipelineLayout(vk->device, &pl, NULL, &d->layout));
    VkShaderModuleCreateInfo sm = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = sizeof(shaders_illixr_depth_comp), .pCode = shaders_illixr_depth_comp,
    };
    DEPTH_CHECK(vk->vkCreateShaderModule(vk->device, &sm, NULL, &module));
    VkComputePipelineCreateInfo cp = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = module, .pName = "main"},
        .layout = d->layout,
    };
    DEPTH_CHECK(vk->vkCreateComputePipelines(vk->device, VK_NULL_HANDLE, 1, &cp, NULL, &d->pipeline));
    vk->vkDestroyShaderModule(vk->device, module, NULL);
    module = VK_NULL_HANDLE;
    VkSamplerCreateInfo si = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_NEAREST, .minFilter = VK_FILTER_NEAREST,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
    };
    DEPTH_CHECK(vk->vkCreateSampler(vk->device, &si, NULL, &d->sampler));
    VkDescriptorPoolSize sizes[] = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 * OFFLOAD_BUFFER_POOL_SIZE},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 * OFFLOAD_BUFFER_POOL_SIZE},
    };
    VkDescriptorPoolCreateInfo pi = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 2 * OFFLOAD_BUFFER_POOL_SIZE, .poolSizeCount = 2, .pPoolSizes = sizes,
    };
    DEPTH_CHECK(vk->vkCreateDescriptorPool(vk->device, &pi, NULL, &d->pool));
    VkDescriptorSetLayout layouts[2 * OFFLOAD_BUFFER_POOL_SIZE];
    for (uint32_t i = 0; i < 2 * OFFLOAD_BUFFER_POOL_SIZE; i++) layouts[i] = d->descriptor_layout;
    VkDescriptorSetAllocateInfo da = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = d->pool,
        .descriptorSetCount = 2 * OFFLOAD_BUFFER_POOL_SIZE, .pSetLayouts = layouts,
    };
    DEPTH_CHECK(vk->vkAllocateDescriptorSets(vk->device, &da, d->sets));

    for (uint32_t i = 0; i < 2 * OFFLOAD_BUFFER_POOL_SIZE; i++) {
        // CUDA imports this same allocation. External image creation AND
        // exportable allocation are required; a plain device allocation fails.
        VkExternalMemoryImageCreateInfo ext = {
            .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
            .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
        };
        VkImageCreateInfo image = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .pNext = &ext,
            .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM,
            .extent = {width, height, 1}, .mipLevels = 1, .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };
        DEPTH_CHECK(vk->vkCreateImage(vk->device, &image, NULL, &d->images[i].image));
        VkMemoryRequirements requirements;
        vk->vkGetImageMemoryRequirements(vk->device, d->images[i].image, &requirements);
        uint32_t type;
        if (!vk_get_memory_type(vk, requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &type)) {
            ret = VK_ERROR_FEATURE_NOT_PRESENT;
            goto fail;
        }
        // Match the existing color allocation: this FFmpeg CUDA importer
        // imports opaque-FD memory without CUDA's dedicated-allocation flag.
        VkExportMemoryAllocateInfo export = {
            .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
            .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
        };
        VkMemoryAllocateInfo allocation = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &export,
            .allocationSize = requirements.size, .memoryTypeIndex = type,
        };
        DEPTH_CHECK(vk->vkAllocateMemory(vk->device, &allocation, NULL, &d->images[i].memory));
        DEPTH_CHECK(vk->vkBindImageMemory(vk->device, d->images[i].image, d->images[i].memory, 0));
        VkImageViewCreateInfo view = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = d->images[i].image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM,
            .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1},
        };
        DEPTH_CHECK(vk->vkCreateImageView(vk->device, &view, NULL, &d->images[i].view));
        fb[i].depth_image = d->images[i].image;
        fb[i].depth_view = d->images[i].view;
        fb[i].depth_memory = d->images[i].memory;
        fb[i].depth_size = requirements.size;
        fb[i].depth_offset = 0;
        fb[i].depth_extent = (VkExtent2D){width, height};
        fb[i].depth_valid = 0;
    }
#undef DEPTH_CHECK
    return true;
fail:
    U_LOG_E("Linux depth initialization failed: %s", vk_result_string(ret));
    if (module) vk->vkDestroyShaderModule(vk->device, module, NULL);
    illixr_linux_depth_fini(d, vk);
    return false;
}

static bool
illixr_linux_depth_record(struct illixr_linux_depth *d, struct vk_bundle *vk, VkCommandBuffer cmd,
                         const struct comp_layer *layer, uint32_t eye, uint32_t slot,
                         struct illixr_framebuffer *fb, bool fast_path)
{
    // The color slow path composites arbitrary layers into scratch. Its pixels
    // cannot be paired with one projection layer's depth: skip instead of
    // sending a misleading color/depth pair to positional reprojection.
    if (!fast_path || !layer || layer->data.type != XRT_LAYER_PROJECTION_DEPTH) return false;
    const struct xrt_sub_image *sub = &layer->data.depth.d[eye].sub;
    const struct comp_swapchain *sc = (const struct comp_swapchain *)comp_layer_get_depth_swapchain(layer, eye);
    if (!sc || sub->image_index >= sc->base.base.image_count ||
        sub->array_index >= sc->images[sub->image_index].array_size ||
        sub->rect.extent.w <= 0 || sub->rect.extent.h <= 0) return false;

    // image_index selects a swapchain image; array_index selects a view of
    // one slice in that image. Using array_index for both reads stale frames.
    VkImageView source = get_image_view(&sc->images[sub->image_index], layer->data.flags, sub->array_index);
    VkDescriptorImageInfo infos[] = {
        {.sampler = d->sampler, .imageView = source, .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        {.imageView = d->images[slot].view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL},
    };
    VkWriteDescriptorSet writes[] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = d->sets[slot], .dstBinding = 0,
         .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &infos[0]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = d->sets[slot], .dstBinding = 1,
         .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &infos[1]},
    };
    // Slot ownership and the renderer's queue-idle-before-release guarantee
    // that this descriptor set is not in use by an earlier submission.
    vk->vkUpdateDescriptorSets(vk->device, 2, writes, 0, NULL);
    VkMemoryBarrier source_ready = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
    };
    VkImageMemoryBarrier output = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = d->images[slot].image,
        .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1},
    };
    // Monado's swapchain handoff already makes the source shader-readable.
    // Sample it in place: no transfer layout or source layout restoration is
    // needed. UNDEFINED on our output deliberately discards the previous frame.
    vk->vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &source_ready, 0, NULL, 1, &output);
    float rect[] = {(float)sub->rect.offset.w / sc->vkic.info.width,
                    (float)sub->rect.offset.h / sc->vkic.info.height,
                    (float)sub->rect.extent.w / sc->vkic.info.width,
                    (float)sub->rect.extent.h / sc->vkic.info.height};
    // Match the offload color fast path's raw Vulkan row order. It also copies
    // sub.rect without applying layer.flip_y; flipping only depth misaligns it.
    vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->pipeline);
    vk->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->layout, 0, 1, &d->sets[slot], 0, NULL);
    vk->vkCmdPushConstants(cmd, d->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(rect), rect);
    vk->vkCmdDispatch(cmd, (fb->depth_extent.width + 15) / 16, (fb->depth_extent.height + 15) / 16, 1);
    output.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    output.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    output.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    output.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vk->vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             0, 0, NULL, 0, NULL, 1, &output);
    fb->near_z = layer->data.depth.d[eye].near_z;
    fb->far_z = layer->data.depth.d[eye].far_z;
    return true;
}
