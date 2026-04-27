// Copyright 2019-2024, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  Compositor rendering code.
 * @author Lubosz Sarnecki <lubosz.sarnecki@collabora.com>
 * @author Jakob Bornecrantz <jakob@collabora.com>
 * @author Rylie Pavlik <rylie.pavlik@collabora.com>
 * @author Moshi Turner <moshiturner@protonmail.com>
 * @ingroup comp_main
 */
#include "render/render_interface.h"
#include "xrt/xrt_defines.h"
#include "xrt/xrt_frame.h"
#include "xrt/xrt_compositor.h"
#include "xrt/xrt_results.h"

#include "os/os_time.h"

#include "math/m_api.h"
#include "math/m_matrix_2x2.h"
#include "math/m_space.h"

#include "util/u_misc.h"
#include "util/u_trace_marker.h"
#include "util/u_distortion_mesh.h"
#include "util/u_sink.h"
#include "util/u_var.h"
#include "util/u_frame_times_widget.h"

#include "util/comp_render.h"

#include "main/comp_frame.h"
#include "main/comp_mirror_to_debug_gui.h"

#ifdef XRT_FEATURE_WINDOW_PEEK
#include "main/comp_window_peek.h"
#endif

#include "vk/vk_helpers.h"
#include "vk/vk_cmd.h"
#include "vk/vk_image_readback_to_xf_pool.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#ifdef USE_MONADO_ILLIXR_DRIVER
#include "../drivers/illixr/illixr_component.h"
#include "shaders/depth16_to_rg_spirv.h"
#include "../drivers/illixr/illixr_mv_shmem.h"

#define MOTION_VECTOR_WIDTH 432
#define MOTION_VECTOR_HEIGHT 432

#define ILLIXR_MV_SENTINEL_SIZE 9999.0f
// Globals defined in comp_compositor.c; extern'd here so dispatch_graphics
// can read the shared-memory ring index written by the hook DLL.
extern struct comp_swapchain *g_illixr_mv_sc;
extern IllixrMvShmem *g_illixr_shmem;
extern uint32_t g_illixr_last_seq;
#endif

/*
 *
 * Small internal helpers.
 *
 */

#define CHAIN(STRUCT, NEXT)                                                                                            \
	do {                                                                                                           \
		(STRUCT).pNext = NEXT;                                                                                 \
		NEXT = (VkBaseInStructure *)&(STRUCT);                                                                 \
	} while (false)


/*
 *
 * Private struct(s).
 *
 */

/*!
 * What is the source of the FoV values used for the final image that the
 * compositor produces and is sent to the hardware (or software).
 */
enum comp_target_fov_source
{
	/*!
	 * The FoV values used for the final target is taken from the
	 * distortion information on the @ref xrt_hmd_parts struct.
	 */
	COMP_TARGET_FOV_SOURCE_DISTORTION,

	/*!
	 * The FoV values used for the final target is taken from the
	 * those returned from the device's get_views.
	 */
	COMP_TARGET_FOV_SOURCE_DEVICE_VIEWS,
};

/*!
 * Holds associated vulkan objects and state to render with a distortion.
 *
 * @ingroup comp_main
 */
struct comp_renderer
{
	//! @name Durable members
	//! @brief These don't require the images to be created and don't depend on it.
	//! @{

	//! The compositor we were created by
	struct comp_compositor *c;
	struct comp_settings *settings;

	struct comp_mirror_to_debug_gui mirror_to_debug_gui;

	//! Render pass for graphics pipeline rendering to the scratch buffer.
	struct render_gfx_render_pass scratch_render_pass;

	struct
	{
		struct
		{
			//! Targets for rendering to the scratch buffer.
			struct render_gfx_target_resources targets[COMP_SCRATCH_NUM_IMAGES];
		} views[XRT_MAX_VIEWS];
	} scratch;

	//! @}

	//! @name Image-dependent members
	//! @{

	//! Index of the current buffer/image
	int32_t acquired_buffer;

	//! Which buffer was last submitted and has a fence pending.
	int32_t fenced_buffer;

	/*!
	 * The render pass used to render to the target, it depends on the
	 * target's format so will be recreated each time the target changes.
	 */
	struct render_gfx_render_pass target_render_pass;

	/*!
	 * Array of "rendering" target resources equal in size to the number of
	 * comp_target images. Each target resources holds all of the resources
	 * needed to render to that target and its views.
	 */
	struct render_gfx_target_resources *rtr_array;

	/*!
	 * Array of fences equal in size to the number of comp_target images.
	 */
	VkFence *fences;

	/*!
	 * The number of renderings/fences we've created: set from comp_target when we use that data.
	 */
	uint32_t buffer_count;

#ifdef USE_MONADO_ILLIXR_DRIVER
	struct illixr_framebuffer illixr_framebuffers[2 * OFFLOAD_BUFFER_POOL_SIZE];

	VkPipeline depth_to_rg_pipeline;
	VkPipelineLayout depth_to_rg_layout;
	VkDescriptorSetLayout depth_to_rg_desc_layout;
	VkDescriptorPool depth_to_rg_desc_pool;
	VkDescriptorSet depth_to_rg_desc_sets[2 * OFFLOAD_BUFFER_POOL_SIZE];
	VkSampler depth_sampler;

	// Color downsampled images for encoding (6 total: 3 buffers × 2 eyes)
	struct {
		VkImage image;
		VkDeviceMemory memory;
		VkImageView view;
		VkDeviceSize memory_size;
		VkDeviceSize memory_offset;
		uint32_t width;
		uint32_t height;
	} illixr_color_downsampled[2 * OFFLOAD_BUFFER_POOL_SIZE];

	// Depth downsampled images from Unity (6 total: 3 buffers × 2 eyes)
	struct {
		VkImage image;
		VkDeviceMemory memory;
		VkImageView view;
		VkDeviceSize memory_size;
		VkDeviceSize memory_offset;
		uint32_t width;
		uint32_t height;
	} illixr_depth_downsampled[2 * OFFLOAD_BUFFER_POOL_SIZE];

	// Motion vector images (RGBA16F) copied from Unity's sentinel quad layer
	struct
	{
		VkImage image;
		VkDeviceMemory memory;
		VkImageView view;
		VkDeviceSize memory_size;
		uint32_t width;
		uint32_t height;
	} illixr_motion_vectors[2 * OFFLOAD_BUFFER_POOL_SIZE];

	// RG-encoded depth images for encoder (6 total: 3 buffers × 2 eyes)
	struct {
		VkImage image;
		VkDeviceMemory memory;
		VkImageView view;
		VkDeviceSize memory_size;
		uint32_t width;
		uint32_t height;
	} illixr_depth_rg[2 * OFFLOAD_BUFFER_POOL_SIZE];

	bool illixr_downsampled_created;
#endif
	//! @}
};

struct comp_scratch_view_state
{
	uint32_t index;

	bool used;
};

/// Holds an array of @ref comp_scratch_view_state to match the number of views
struct comp_render_scratch_state
{
	struct comp_scratch_view_state views[2];
};


/*
 *
 * Scratch helpers.
 *
 */

/// Zeroes the object pointed to by @p crss then populates it with the image indices.
static void
scratch_get_init(struct comp_render_scratch_state *crss, struct comp_renderer *r, uint32_t view_count)
{
	struct comp_compositor *c = r->c;
	U_ZERO(crss);

	for (uint32_t i = 0; i < view_count; i++) {
		comp_scratch_single_images_get(&c->scratch.views[i], &crss->views[i].index);
	}
}

/// Calls done or discard on each view in @p crss, depending on whether "used" is set.
static void
scratch_get_fini(struct comp_render_scratch_state *crss, struct comp_renderer *r, uint32_t view_count)
{
	struct comp_compositor *c = r->c;

	for (uint32_t i = 0; i < view_count; i++) {
		if (crss->views[i].used) {
			comp_scratch_single_images_done(&c->scratch.views[i]);
		} else {
			comp_scratch_single_images_discard(&c->scratch.views[i]);
		}
	}
}

/*
 *
 * Functions.
 *
 */

static void
renderer_wait_queue_idle(struct comp_renderer *r)
{
	COMP_TRACE_MARKER();
	struct vk_bundle *vk = &r->c->base.vk;

	os_mutex_lock(&vk->queue_mutex);
	vk->vkQueueWaitIdle(vk->queue);
	os_mutex_unlock(&vk->queue_mutex);
}

static void
calc_viewport_data(struct comp_renderer *r,
                   struct render_viewport_data out_viewport_data[XRT_MAX_VIEWS],
                   size_t view_count)
{
	struct comp_compositor *c = r->c;

	bool pre_rotate = false;
	if (r->c->target->surface_transform & VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR ||
	    r->c->target->surface_transform & VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR) {
		COMP_SPEW(c, "Swapping width and height, since we are pre rotating");
		pre_rotate = true;
	}

	int w_i32 = pre_rotate ? r->c->xdev->hmd->screens[0].h_pixels : r->c->xdev->hmd->screens[0].w_pixels;
	int h_i32 = pre_rotate ? r->c->xdev->hmd->screens[0].w_pixels : r->c->xdev->hmd->screens[0].h_pixels;

	float scale_x = (float)r->c->target->width / (float)w_i32;
	float scale_y = (float)r->c->target->height / (float)h_i32;

	for (uint32_t i = 0; i < view_count; ++i) {
		struct xrt_view *v = &r->c->xdev->hmd->views[i];
		if (pre_rotate) {
			out_viewport_data[i] = (struct render_viewport_data){
			    .x = (uint32_t)(v->viewport.y_pixels * scale_x),
			    .y = (uint32_t)(v->viewport.x_pixels * scale_y),
			    .w = (uint32_t)(v->viewport.h_pixels * scale_x),
			    .h = (uint32_t)(v->viewport.w_pixels * scale_y),
			};
		} else {
			out_viewport_data[i] = (struct render_viewport_data){
			    .x = (uint32_t)(v->viewport.x_pixels * scale_x),
			    .y = (uint32_t)(v->viewport.y_pixels * scale_y),
			    .w = (uint32_t)(v->viewport.w_pixels * scale_x),
			    .h = (uint32_t)(v->viewport.h_pixels * scale_y),
			};
		}
#ifdef USE_MONADO_ILLIXR_DRIVER
		/*
		 * ILLIXR: Double viewport width
		 * In v21, this was done inline:
		 *   l_viewport_data.w *= 2;
		 *   r_viewport_data.w *= 2;
		 */
		//out_viewport_data[i].w *= 2;
#endif
	}
}

static void
calc_vertex_rot_data(struct comp_renderer *r, struct xrt_matrix_2x2 out_vertex_rots[XRT_MAX_VIEWS], size_t view_count)
{
	bool pre_rotate = false;
	if (r->c->target->surface_transform & VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR ||
	    r->c->target->surface_transform & VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR) {
		COMP_SPEW(r->c, "Swapping width and height, since we are pre rotating");
		pre_rotate = true;
	}

	const struct xrt_matrix_2x2 rotation_90_cw = {{
	    .vecs =
	        {
	            {0, 1},
	            {-1, 0},
	        },
	}};

	for (uint32_t i = 0; i < view_count; i++) {
		// Get the view.
		struct xrt_view *v = &r->c->xdev->hmd->views[i];

		// Copy data.
		struct xrt_matrix_2x2 rot = v->rot;

		// Should we rotate.
		if (pre_rotate) {
			m_mat2x2_multiply(&rot, &rotation_90_cw, &rot);
		}

		out_vertex_rots[i] = rot;
	}
}

static void
calc_pose_data(struct comp_renderer *r,
               enum comp_target_fov_source fov_source,
               struct xrt_fov out_fovs[XRT_MAX_VIEWS],
               struct xrt_pose out_world[XRT_MAX_VIEWS],
               struct xrt_pose out_eye[XRT_MAX_VIEWS],
               uint32_t view_count)
{
	COMP_TRACE_MARKER();

	struct xrt_vec3 default_eye_relation = {
	    0.063000f, /*! @todo get actual ipd_meters */
	    0.0f,
	    0.0f,
	};

	struct xrt_space_relation head_relation = XRT_SPACE_RELATION_ZERO;
	struct xrt_fov xdev_fovs[XRT_MAX_VIEWS] = XRT_STRUCT_INIT;
	struct xrt_pose xdev_poses[XRT_MAX_VIEWS] = XRT_STRUCT_INIT;

	xrt_device_get_view_poses(                           //
	    r->c->xdev,                                      //
	    &default_eye_relation,                           //
	    r->c->frame.rendering.predicted_display_time_ns, // at_timestamp_ns
	    view_count,                                      //
	    &head_relation,                                  // out_head_relation
	    xdev_fovs,                                       // out_fovs
	    xdev_poses);                                     // out_poses

	struct xrt_fov dist_fov[XRT_MAX_VIEWS] = XRT_STRUCT_INIT;
	for (uint32_t i = 0; i < view_count; i++) {
		dist_fov[i] = r->c->xdev->hmd->distortion.fov[i];
	}

	bool use_xdev = false; // Probably what we want.

	switch (fov_source) {
	case COMP_TARGET_FOV_SOURCE_DISTORTION: use_xdev = false; break;
	case COMP_TARGET_FOV_SOURCE_DEVICE_VIEWS: use_xdev = true; break;
	}

	for (uint32_t i = 0; i < view_count; i++) {
		const struct xrt_fov fov = use_xdev ? xdev_fovs[i] : dist_fov[i];
		const struct xrt_pose eye_pose = xdev_poses[i];

		struct xrt_space_relation result = {0};
		struct xrt_relation_chain xrc = {0};
		m_relation_chain_push_pose_if_not_identity(&xrc, &eye_pose);
		m_relation_chain_push_relation(&xrc, &head_relation);
		m_relation_chain_resolve(&xrc, &result);

		// Results to callers.
		out_fovs[i] = fov;
		out_world[i] = result.pose;
		out_eye[i] = eye_pose;

		// For remote rendering targets.
		r->c->base.frame_params.fovs[i] = fov;
		r->c->base.frame_params.poses[i] = result.pose;
	}
}

//! @pre comp_target_has_images(r->c->target)
static void
renderer_build_rendering_target_resources(struct comp_renderer *r,
                                          struct render_gfx_target_resources *rtr,
                                          uint32_t index)
{
	COMP_TRACE_MARKER();

	struct comp_compositor *c = r->c;

	VkImageView image_view = r->c->target->images[index].view;
	VkExtent2D extent = {r->c->target->width, r->c->target->height};

	render_gfx_target_resources_init( //
	    rtr,                          //
	    &c->nr,                       //
	    &r->target_render_pass,       //
	    image_view,                   //
	    extent);                      //
}

/*!
 * @pre comp_target_has_images(r->c->target)
 * Update r->buffer_count before calling.
 */
static void
renderer_create_renderings_and_fences(struct comp_renderer *r)
{
	assert(r->fences == NULL);
	if (r->buffer_count == 0) {
		COMP_ERROR(r->c, "Requested 0 command buffers.");
		return;
	}

	COMP_DEBUG(r->c, "Allocating %d Command Buffers.", r->buffer_count);

	struct vk_bundle *vk = &r->c->base.vk;

	bool use_compute = r->settings->use_compute;
	if (!use_compute) {
		r->rtr_array = U_TYPED_ARRAY_CALLOC(struct render_gfx_target_resources, r->buffer_count);

		render_gfx_render_pass_init(          //
		    &r->target_render_pass,           // rgrp
		    &r->c->nr,                        // struct render_resources
		    r->c->target->format,             //
		    VK_ATTACHMENT_LOAD_OP_CLEAR,      // load_op
		    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR); // final_layout

		for (uint32_t i = 0; i < r->buffer_count; ++i) {
			renderer_build_rendering_target_resources(r, &r->rtr_array[i], i);
		}
	}

	r->fences = U_TYPED_ARRAY_CALLOC(VkFence, r->buffer_count);

	for (uint32_t i = 0; i < r->buffer_count; i++) {
		VkFenceCreateInfo fence_info = {
		    .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
		    .flags = VK_FENCE_CREATE_SIGNALED_BIT,
		};

		VkResult ret = vk->vkCreateFence( //
		    vk->device,                   //
		    &fence_info,                  //
		    NULL,                         //
		    &r->fences[i]);               //
		if (ret != VK_SUCCESS) {
			COMP_ERROR(r->c, "vkCreateFence: %s", vk_result_string(ret));
		}

		char buf[] = "Comp Renderer X_XXXX_XXXX";
		snprintf(buf, ARRAY_SIZE(buf), "Comp Renderer %u", i);
		VK_NAME_FENCE(vk, r->fences[i], buf);
	}
}

static void
renderer_close_renderings_and_fences(struct comp_renderer *r)
{
	struct vk_bundle *vk = &r->c->base.vk;
	// Renderings
	if (r->buffer_count > 0 && r->rtr_array != NULL) {
		for (uint32_t i = 0; i < r->buffer_count; i++) {
			render_gfx_target_resources_fini(&r->rtr_array[i]);
		}

		// Close the render pass used for rendering to the target.
		render_gfx_render_pass_fini(&r->target_render_pass);

		free(r->rtr_array);
		r->rtr_array = NULL;
	}

	// Fences
	if (r->buffer_count > 0 && r->fences != NULL) {
		for (uint32_t i = 0; i < r->buffer_count; i++) {
			vk->vkDestroyFence(vk->device, r->fences[i], NULL);
			r->fences[i] = VK_NULL_HANDLE;
		}
		free(r->fences);
		r->fences = NULL;
	}

	r->buffer_count = 0;
	r->acquired_buffer = -1;
	r->fenced_buffer = -1;
}

/*!
 * @brief Ensure that target images and renderings are created, if possible.
 *
 * @param r Self pointer
 * @param force_recreate If true, will tear down and re-create images and renderings, e.g. for a resize
 *
 * @returns true if images and renderings are ready and created.
 *
 * @private @memberof comp_renderer
 * @ingroup comp_main
 */
static bool
renderer_ensure_images_and_renderings(struct comp_renderer *r, bool force_recreate)
{
	struct comp_compositor *c = r->c;
	struct comp_target *target = c->target;

	if (!comp_target_check_ready(target)) {
		// Not ready, so can't render anything.
		return false;
	}

	// We will create images if we don't have any images or if we were told to recreate them.
	bool create = force_recreate || !comp_target_has_images(target) || (r->buffer_count == 0);
	if (!create) {
		return true;
	}

	COMP_DEBUG(c, "Creating images and renderings (force_recreate: %s).", force_recreate ? "true" : "false");

	/*
	 * This makes sure that any pending command buffer has completed
	 * and all resources referred by it can now be manipulated. This
	 * make sure that validation doesn't complain. This is done
	 * during resize so isn't time critical.
	 */
	renderer_wait_queue_idle(r);

	// Make we sure we destroy all dependent things before creating new images.
	renderer_close_renderings_and_fences(r);

	VkImageUsageFlags image_usage = 0;
	if (r->settings->use_compute) {
		image_usage |= VK_IMAGE_USAGE_STORAGE_BIT;
	} else {
		image_usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
	}

	if (c->peek) {
		image_usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
	}

	struct comp_target_create_images_info info = {
	    .extent =
	        {
	            .width = r->c->settings.preferred.width,
	            .height = r->c->settings.preferred.height,
	        },
	    .image_usage = image_usage,
	    .color_space = r->settings->color_space,
	    .present_mode = r->settings->present_mode,
	};

	static_assert(ARRAY_SIZE(info.formats) == ARRAY_SIZE(r->c->settings.formats), "Miss-match format array sizes");
	for (uint32_t i = 0; i < r->c->settings.format_count; i++) {
		info.formats[info.format_count++] = r->c->settings.formats[i];
	}

	comp_target_create_images(r->c->target, &info);

	bool pre_rotate = false;
	if (r->c->target->surface_transform & VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR ||
	    r->c->target->surface_transform & VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR) {
		pre_rotate = true;
	}

	// @todo: is it safe to fail here?
	if (!render_distortion_images_ensure(&r->c->nr, &r->c->base.vk, r->c->xdev, pre_rotate))
		return false;

	r->buffer_count = r->c->target->image_count;

	renderer_create_renderings_and_fences(r);

	assert(r->buffer_count != 0);

#ifdef USE_MONADO_ILLIXR_DRIVER
	// ILLIXR: Initialize timewarp with custom framebuffers
	if (strcmp(r->c->xdev->str, "ILLIXR") == 0) {
		VkExtent2D extent = {
		    .width = r->c->xdev->hmd->screens[0].w_pixels,
		    .height = r->c->xdev->hmd->screens[0].h_pixels,
		};

		// Initialize ILLIXR timewarp without framebuffer data
		// Framebuffers will be accessed later via illixr_get_framebuffer_info()
		illixr_initialize_timewarp(r->target_render_pass.render_pass,
		                           0, // subpass
		                           extent,
		                           NULL,                    // images - not needed at init
		                           NULL,                    // image_views - not needed at init
		                           NULL,                    // device_memory - not needed at init
		                           NULL,                    // sizes - not needed at init
		                           NULL,                    // offsets - not needed at init
		                           OFFLOAD_BUFFER_POOL_SIZE,// num_buffers_per_eye
		                           r->illixr_framebuffers
		);
		COMP_INFO(r->c, "ILLIXR timewarp initialized (extent=%ux%u, buffers=%u)", extent.width, extent.height,
		          OFFLOAD_BUFFER_POOL_SIZE);
		COMP_INFO(r->c, "Framebuffers will be accessed on-demand via illixr_get_framebuffer_info()");
	}

#endif
	return true;
}

#ifdef USE_MONADO_ILLIXR_DRIVER

static VkShaderModule create_embedded_shader_module(struct vk_bundle* vk) {
	VkShaderModuleCreateInfo create_info = {
	    .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
	    .codeSize = depth16_to_rg_spirv_len,
	    .pCode = (const uint32_t*)depth16_to_rg_spirv,
	};

	VkShaderModule shader_module;
	VkResult ret = vk->vkCreateShaderModule(vk->device, &create_info,
	                                        NULL, &shader_module);
	if (ret != VK_SUCCESS) {
		U_LOG_E("Failed to create shader module: %d", ret);
		return VK_NULL_HANDLE;
	}

	return shader_module;
}

static void create_depth_to_rg_pipeline(struct comp_renderer* r) {
	VkShaderModule shader_module = create_embedded_shader_module(&r->c->base.vk);
	struct vk_bundle* vk = &r->c->base.vk;

	// Descriptor set layout
	VkDescriptorSetLayoutBinding bindings[2] = {
	    {
	        .binding = 0,
	        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
	        .descriptorCount = 1,
	        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
	    },
	    {
	        .binding = 1,
	        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
	        .descriptorCount = 1,
	        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
	    },
	};

	VkDescriptorSetLayoutCreateInfo desc_layout_info = {
	    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
	    .bindingCount = 2,
	    .pBindings = bindings,
	};

	VkResult ret = vk->vkCreateDescriptorSetLayout(vk->device, &desc_layout_info,
	                                               NULL, &r->depth_to_rg_desc_layout);
	if (ret != VK_SUCCESS) {
		COMP_ERROR(r->c, "Failed to create depth-to-RG descriptor layout: %d", ret);
		return;
	}

	// Pipeline layout
	VkPipelineLayoutCreateInfo pipeline_layout_info = {
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
	    .setLayoutCount = 1,
	    .pSetLayouts = &r->depth_to_rg_desc_layout,
	};

	ret = vk->vkCreatePipelineLayout(vk->device, &pipeline_layout_info,
	                                 NULL, &r->depth_to_rg_layout);
	if (ret != VK_SUCCESS) {
		COMP_ERROR(r->c, "Failed to create depth-to-RG pipeline layout: %d", ret);
		return;
	}

	// Compute pipeline
	VkComputePipelineCreateInfo pipeline_info = {
	    .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
	    .stage = {
	        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
	        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
	        .module = shader_module,
	        .pName = "main",
	    },
	    .layout = r->depth_to_rg_layout,
	};

	ret = vk->vkCreateComputePipelines(vk->device, VK_NULL_HANDLE, 1,
	                                   &pipeline_info, NULL, &r->depth_to_rg_pipeline);
	if (ret != VK_SUCCESS) {
		COMP_ERROR(r->c, "Failed to create depth-to-RG pipeline: %d", ret);
		return;
	}

	vk->vkDestroyShaderModule(vk->device, shader_module, NULL);

	// Create sampler for depth input
	VkSamplerCreateInfo sampler_info = {
	    .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
	    .magFilter = VK_FILTER_NEAREST, // Nearest for depth (no interpolation)
	    .minFilter = VK_FILTER_NEAREST,
	    .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
	    .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
	    .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
	    .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
	    .mipLodBias = 0.0f,
	    .anisotropyEnable = VK_FALSE,
	    .maxAnisotropy = 1.0f,
	    .compareEnable = VK_FALSE,
	    .compareOp = VK_COMPARE_OP_ALWAYS,
	    .minLod = 0.0f,
	    .maxLod = 0.0f,
	    .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
	    .unnormalizedCoordinates = VK_FALSE,
	};

	ret = vk->vkCreateSampler(vk->device, &sampler_info, NULL, &r->depth_sampler);
	if (ret != VK_SUCCESS) {
		COMP_ERROR(r->c, "Failed to create depth sampler: %d", ret);
		return;
	}
	COMP_INFO(r->c, "Created depth-to-RG conversion pipeline");
}

static void create_depth_to_rg_descriptors(struct comp_renderer* r) {
	struct vk_bundle* vk = &r->c->base.vk;

	// Create descriptor pool
	VkDescriptorPoolSize pool_sizes[] = {
	    {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 * OFFLOAD_BUFFER_POOL_SIZE},
	    {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 * OFFLOAD_BUFFER_POOL_SIZE},
	};

	VkDescriptorPoolCreateInfo pool_info = {
	    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
	    .maxSets = 2 * OFFLOAD_BUFFER_POOL_SIZE,
	    .poolSizeCount = 2,
	    .pPoolSizes = pool_sizes,
	};

	VkResult ret = vk->vkCreateDescriptorPool(vk->device, &pool_info, NULL,
	                                          &r->depth_to_rg_desc_pool);
	if (ret != VK_SUCCESS) {
		COMP_ERROR(r->c, "Failed to create depth-to-RG descriptor pool: %d", ret);
		return;
	}

	// Allocate descriptor sets
	VkDescriptorSetLayout layouts[2 * OFFLOAD_BUFFER_POOL_SIZE];
	for (int i = 0; i < 2 * OFFLOAD_BUFFER_POOL_SIZE; i++) {
		layouts[i] = r->depth_to_rg_desc_layout;
	}

	VkDescriptorSetAllocateInfo alloc_info = {
	    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
	    .descriptorPool = r->depth_to_rg_desc_pool,
	    .descriptorSetCount = 2 * OFFLOAD_BUFFER_POOL_SIZE,
	    .pSetLayouts = layouts,
	};

	ret = vk->vkAllocateDescriptorSets(vk->device, &alloc_info,
	                                   r->depth_to_rg_desc_sets);
	if (ret != VK_SUCCESS) {
		COMP_ERROR(r->c, "Failed to allocate depth-to-RG descriptor sets: %d", ret);
		return;
	}

	// Update descriptor sets
	for (uint32_t i = 0; i < 2 * OFFLOAD_BUFFER_POOL_SIZE; i++) {
		VkDescriptorImageInfo depth_input_info = {
		    .sampler = r->depth_sampler,                      // Use immutable sampler or create one
		    .imageView = r->illixr_depth_downsampled[i].view,  // Depth input
		    .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		};

		VkDescriptorImageInfo rg_output_info = {
		    .imageView = r->illixr_depth_rg[i].view,  // RG output
		    .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
		};

		VkWriteDescriptorSet writes[2] = {
		    {
		        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		        .dstSet = r->depth_to_rg_desc_sets[i],
		        .dstBinding = 0,
		        .dstArrayElement = 0,
		        .descriptorCount = 1,
		        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		        .pImageInfo = &depth_input_info,
		    },
		    {
		        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		        .dstSet = r->depth_to_rg_desc_sets[i],
		        .dstBinding = 1,
		        .dstArrayElement = 0,
		        .descriptorCount = 1,
		        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		        .pImageInfo = &rg_output_info,
		    },
		};

		if (r->illixr_depth_downsampled[i].view == VK_NULL_HANDLE ||
		    r->illixr_depth_rg[i].view == VK_NULL_HANDLE) {
			continue; // Skip invalid views
		}
		vk->vkUpdateDescriptorSets(vk->device, 2, writes, 0, NULL);
	}
}

#endif
//! Create renderer and initialize non-image-dependent members
static void
renderer_init(struct comp_renderer *r, struct comp_compositor *c, VkExtent2D scratch_extent)
{
	COMP_TRACE_MARKER();
	bool bret;

	r->c = c;
	r->settings = &c->settings;

	r->acquired_buffer = -1;
	r->fenced_buffer = -1;
	r->rtr_array = NULL;
	r->illixr_downsampled_created = false;

	// Shared render pass between all scratch images.
	render_gfx_render_pass_init(                   //
	    &r->scratch_render_pass,                   // rgrp
	    &r->c->nr,                                 // struct render_resources
	    VK_FORMAT_R8G8B8A8_SRGB,                   // format
	    VK_ATTACHMENT_LOAD_OP_CLEAR,               // load_op
	    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL); // final_layout

	for (uint32_t i = 0; i < c->nr.view_count; i++) {
		bret = comp_scratch_single_images_ensure(&r->c->scratch.views[i], &r->c->base.vk, scratch_extent);
		if (!bret) {
			COMP_ERROR(c, "comp_scratch_single_images_ensure: false");
			assert(false && "Whelp, can't return an error. But should never really fail.");
		}

		for (uint32_t k = 0; k < COMP_SCRATCH_NUM_IMAGES; k++) {
			struct render_scratch_color_image *rsci = &c->scratch.views[i].images[k];

			render_gfx_target_resources_init(    //
			    &r->scratch.views[i].targets[k], //
			    &r->c->nr,                       //
			    &r->scratch_render_pass,         //
			    rsci->srgb_view,                 //
			    scratch_extent);                 //
		}
	}

	// Try to early-allocate these, in case we can.
	renderer_ensure_images_and_renderings(r, false);

	struct vk_bundle *vk = &r->c->base.vk;

	VkResult ret = comp_mirror_init( //
	    &r->mirror_to_debug_gui,     //
	    vk,                          //
	    &c->shaders,                 //
	    scratch_extent);             //
	if (ret != VK_SUCCESS) {
		COMP_ERROR(c, "comp_mirror_init: %s", vk_result_string(ret));
		assert(false && "Whelp, can't return a error. But should never really fail.");
	}
#ifdef USE_MONADO_ILLIXR_DRIVER
	create_depth_to_rg_pipeline(r);
	create_depth_to_rg_descriptors(r);
#endif
}

static void
renderer_wait_for_last_fence(struct comp_renderer *r)
{
	COMP_TRACE_MARKER();

	if (r->fenced_buffer < 0) {
		return;
	}

	struct vk_bundle *vk = &r->c->base.vk;
	VkResult ret;

	ret = vk->vkWaitForFences(vk->device, 1, &r->fences[r->fenced_buffer], VK_TRUE, UINT64_MAX);
	if (ret != VK_SUCCESS) {
		COMP_ERROR(r->c, "vkWaitForFences: %s", vk_result_string(ret));
	}

	r->fenced_buffer = -1;
}

static XRT_CHECK_RESULT VkResult
renderer_submit_queue(struct comp_renderer *r, VkCommandBuffer cmd, VkPipelineStageFlags pipeline_stage_flag)
{
	COMP_TRACE_MARKER();

	struct vk_bundle *vk = &r->c->base.vk;
	int64_t frame_id = r->c->frame.rendering.id;
	VkResult ret;

	assert(frame_id >= 0);


	/*
	 * Wait for previous frame's work to complete.
	 */

	// Wait for the last fence, if any.
	renderer_wait_for_last_fence(r);
	assert(r->fenced_buffer < 0);

	assert(r->acquired_buffer >= 0);
	ret = vk->vkResetFences(vk->device, 1, &r->fences[r->acquired_buffer]);
	VK_CHK_AND_RET(ret, "vkResetFences");


	/*
	 * Regular semaphore setup.
	 */

	// Convenience.
	struct comp_target *ct = r->c->target;
#define WAIT_SEMAPHORE_COUNT 1

	VkSemaphore wait_sems[WAIT_SEMAPHORE_COUNT] = {ct->semaphores.present_complete};
	VkPipelineStageFlags stage_flags[WAIT_SEMAPHORE_COUNT] = {pipeline_stage_flag};

	VkSemaphore *wait_sems_ptr = NULL;
	VkPipelineStageFlags *stage_flags_ptr = NULL;
	uint32_t wait_sem_count = 0;
	if (wait_sems[0] != VK_NULL_HANDLE) {
		wait_sems_ptr = wait_sems;
		stage_flags_ptr = stage_flags;
		wait_sem_count = WAIT_SEMAPHORE_COUNT;
	}

	// Next pointer for VkSubmitInfo
	const void *next = NULL;

#ifdef VK_KHR_timeline_semaphore
	assert(!comp_frame_is_invalid_locked(&r->c->frame.rendering));
	uint64_t render_complete_signal_values[WAIT_SEMAPHORE_COUNT] = {(uint64_t)frame_id};

	VkTimelineSemaphoreSubmitInfoKHR timeline_info = {
	    .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO_KHR,
	};

	if (ct->semaphores.render_complete_is_timeline) {
		timeline_info = (VkTimelineSemaphoreSubmitInfoKHR){
		    .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO_KHR,
		    .signalSemaphoreValueCount = WAIT_SEMAPHORE_COUNT,
		    .pSignalSemaphoreValues = render_complete_signal_values,
		};

		CHAIN(timeline_info, next);
	}
#endif


	VkSubmitInfo comp_submit_info = {
	    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
	    .pNext = next,
	    .pWaitDstStageMask = stage_flags_ptr,
	    .pWaitSemaphores = wait_sems_ptr,
	    .waitSemaphoreCount = wait_sem_count,
	    .commandBufferCount = 1,
	    .pCommandBuffers = &cmd,
	    .signalSemaphoreCount = 1,
	    .pSignalSemaphores = &ct->semaphores.render_complete,
	};

	// Everything prepared, now we are submitting.
	comp_target_mark_submit_begin(ct, frame_id, os_monotonic_get_ns());

	/*
	 * The renderer command buffer pool is only accessed from one thread,
	 * this satisfies the `_locked` requirement of the function. This lets
	 * us avoid taking a lot of locks. The queue lock will be taken by
	 * @ref vk_cmd_submit_locked tho.
	 */
	ret = vk_cmd_submit_locked(vk, 1, &comp_submit_info, r->fences[r->acquired_buffer]);

	// We have now completed the submit, even if we failed.
	comp_target_mark_submit_end(ct, frame_id, os_monotonic_get_ns());

	// Check after marking as submit complete.
	VK_CHK_AND_RET(ret, "vk_cmd_submit_locked");

	// This buffer now have a pending fence.
	r->fenced_buffer = r->acquired_buffer;

	return ret;
}

static void
renderer_acquire_swapchain_image(struct comp_renderer *r)
{
	COMP_TRACE_MARKER();

	uint32_t buffer_index = 0;
	VkResult ret;

	assert(r->acquired_buffer < 0);

	if (!renderer_ensure_images_and_renderings(r, false)) {
		// Not ready yet.
		return;
	}
	ret = comp_target_acquire(r->c->target, &buffer_index);

	if ((ret == VK_ERROR_OUT_OF_DATE_KHR) || (ret == VK_SUBOPTIMAL_KHR)) {
		COMP_DEBUG(r->c, "Received %s.", vk_result_string(ret));

		if (!renderer_ensure_images_and_renderings(r, true)) {
			// Failed on force recreate.
			COMP_ERROR(r->c,
			           "renderer_acquire_swapchain_image: comp_target_acquire was out of date, force "
			           "re-create image and renderings failed. Probably the target disappeared.");
			return;
		}

		/* Acquire image again to silence validation error */
		ret = comp_target_acquire(r->c->target, &buffer_index);
		if (ret != VK_SUCCESS) {
			COMP_ERROR(r->c, "comp_target_acquire: %s", vk_result_string(ret));
		}
	} else if (ret != VK_SUCCESS) {
		COMP_ERROR(r->c, "comp_target_acquire: %s", vk_result_string(ret));
	}

	r->acquired_buffer = buffer_index;
}

static void
renderer_resize(struct comp_renderer *r)
{
	if (!comp_target_check_ready(r->c->target)) {
		// Can't create images right now.
		// Just close any existing renderings.
		renderer_close_renderings_and_fences(r);
		return;
	}
	// Force recreate.
	renderer_ensure_images_and_renderings(r, true);
}

static void
renderer_present_swapchain_image(struct comp_renderer *r, uint64_t desired_present_time_ns, uint64_t present_slop_ns)
{
	COMP_TRACE_MARKER();

	VkResult ret;

	assert(!comp_frame_is_invalid_locked(&r->c->frame.rendering));
	uint64_t render_complete_signal_value = (uint64_t)r->c->frame.rendering.id;

	ret = comp_target_present(        //
	    r->c->target,                 //
	    r->c->base.vk.queue,          //
	    r->acquired_buffer,           //
	    render_complete_signal_value, //
	    desired_present_time_ns,      //
	    present_slop_ns);             //
	r->acquired_buffer = -1;

	if (ret == VK_ERROR_OUT_OF_DATE_KHR || ret == VK_SUBOPTIMAL_KHR) {
		renderer_resize(r);
		return;
	}
	if (ret != VK_SUCCESS) {
		COMP_ERROR(r->c, "vk_swapchain_present: %s", vk_result_string(ret));
	}
}

static void
renderer_fini(struct comp_renderer *r)
{
	struct vk_bundle *vk = &r->c->base.vk;

#ifdef USE_MONADO_ILLIXR_DRIVER
	if (r->depth_sampler != VK_NULL_HANDLE) {
		vk->vkDestroySampler(vk->device, r->depth_sampler, NULL);
		r->depth_sampler = VK_NULL_HANDLE;
	}

	for (uint32_t i = 0; i < 2 * OFFLOAD_BUFFER_POOL_SIZE; i++) {
		if (r->illixr_motion_vectors[i].view != VK_NULL_HANDLE) {
			vk->vkDestroyImageView(vk->device, r->illixr_motion_vectors[i].view, NULL);
			r->illixr_motion_vectors[i].view = VK_NULL_HANDLE;
		}
		if (r->illixr_motion_vectors[i].image != VK_NULL_HANDLE) {
			vk->vkDestroyImage(vk->device, r->illixr_motion_vectors[i].image, NULL);
			r->illixr_motion_vectors[i].image = VK_NULL_HANDLE;
		}
		if (r->illixr_motion_vectors[i].memory != VK_NULL_HANDLE) {
			vk->vkFreeMemory(vk->device, r->illixr_motion_vectors[i].memory, NULL);
			r->illixr_motion_vectors[i].memory = VK_NULL_HANDLE;
		}
	}
#endif
	// Command buffers
	renderer_close_renderings_and_fences(r);

	// Do before layer render just in case it holds any references.
	comp_mirror_fini(&r->mirror_to_debug_gui, vk);

	// Do this after the layer renderer.
	for (uint32_t i = 0; i < r->c->nr.view_count; i++) {
		for (uint32_t k = 0; k < COMP_SCRATCH_NUM_IMAGES; k++) {
			render_gfx_target_resources_fini(&r->scratch.views[i].targets[k]);
		}
	}

	// Do this after the layer renderer and targert resources.
	render_gfx_render_pass_fini(&r->scratch_render_pass);
}


/*
 *
 * Graphics
 *
 */

#ifdef USE_MONADO_ILLIXR_DRIVER

static void create_illixr_depth_rg_images(struct comp_renderer* r, uint32_t width, uint32_t height) {
	struct comp_compositor *c = r->c;
	struct vk_bundle *vk = &c->base.vk;

	COMP_INFO(c, "Creating ILLIXR RG depth images: %ux%u", width, height);

	for (uint32_t i = 0; i < 2 * OFFLOAD_BUFFER_POOL_SIZE; i++) {
		// Create RG8 image
		VkImageCreateInfo image_info = {
		    .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		    .imageType = VK_IMAGE_TYPE_2D,
		    .format = VK_FORMAT_R8G8_UNORM, // RG format (2 channels)
		    .extent = {width, height, 1},
		    .mipLevels = 1,
		    .arrayLayers = 1,
		    .samples = VK_SAMPLE_COUNT_1_BIT,
		    .tiling = VK_IMAGE_TILING_OPTIMAL,
		    .usage = VK_IMAGE_USAGE_STORAGE_BIT |      // For compute shader write
		             VK_IMAGE_USAGE_TRANSFER_SRC_BIT | // For export to encoder
		             VK_IMAGE_USAGE_SAMPLED_BIT,       // For reading
		    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		    .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		};

		VkExternalMemoryImageCreateInfo external_info = {
		    .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
#ifdef _WIN32
		    .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,
#else
		    .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
#endif
		};
		image_info.pNext = &external_info;

		VkResult ret = vk->vkCreateImage(vk->device, &image_info, NULL, &r->illixr_depth_rg[i].image);
		if (ret != VK_SUCCESS) {
			COMP_ERROR(c, "Failed to create RG depth image %u: %d", i, ret);
			return;
		}

		// Get memory requirements
		VkMemoryRequirements mem_reqs;
		vk->vkGetImageMemoryRequirements(vk->device, r->illixr_depth_rg[i].image, &mem_reqs);

		// Find memory type
		uint32_t memory_type_index;
		bool found =
		    vk_get_memory_type(vk, mem_reqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &memory_type_index);
		if (!found) {
			COMP_ERROR(c, "Failed to find suitable memory type for RG depth");
			return;
		}

		// Allocate exportable memory
		VkExportMemoryAllocateInfo export_alloc = {
		    .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
#ifdef _WIN32
		    .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,
#else
		    .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
#endif
		};

		VkMemoryAllocateInfo alloc_info = {
		    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		    .pNext = &export_alloc,
		    .allocationSize = mem_reqs.size,
		    .memoryTypeIndex = memory_type_index,
		};

		ret = vk->vkAllocateMemory(vk->device, &alloc_info, NULL, &r->illixr_depth_rg[i].memory);
		if (ret != VK_SUCCESS) {
			COMP_ERROR(c, "Failed to allocate RG depth memory %u: %d", i, ret);
			return;
		}

		// Bind memory
		ret = vk->vkBindImageMemory(vk->device, r->illixr_depth_rg[i].image, r->illixr_depth_rg[i].memory, 0);
		if (ret != VK_SUCCESS) {
			COMP_ERROR(c, "Failed to bind RG depth memory %u: %d", i, ret);
			return;
		}

		// Create image view
		VkImageViewCreateInfo view_info = {
		    .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		    .image = r->illixr_depth_rg[i].image,
		    .viewType = VK_IMAGE_VIEW_TYPE_2D,
		    .format = VK_FORMAT_R8G8_UNORM,
		    .subresourceRange =
		        {
		            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
		            .baseMipLevel = 0,
		            .levelCount = 1,
		            .baseArrayLayer = 0,
		            .layerCount = 1,
		        },
		};

		ret = vk->vkCreateImageView(vk->device, &view_info, NULL, &r->illixr_depth_rg[i].view);
		if (ret != VK_SUCCESS) {
			COMP_ERROR(c, "Failed to create RG depth view %u: %d", i, ret);
			return;
		}

		// Store size info
		r->illixr_depth_rg[i].memory_size = mem_reqs.size;
		r->illixr_depth_rg[i].width = width;
		r->illixr_depth_rg[i].height = height;
	}

	COMP_INFO(c, "Created %d RG depth images", 2 * OFFLOAD_BUFFER_POOL_SIZE);
}

static void
create_illixr_motion_vector_images(struct comp_renderer *r, uint32_t width, uint32_t height)
{
	struct comp_compositor *c = r->c;
	struct vk_bundle *vk = &c->base.vk;

	COMP_INFO(c, "Creating ILLIXR motion vector images: %ux%u", width, height);

	for (uint32_t i = 0; i < 2 * OFFLOAD_BUFFER_POOL_SIZE; i++) {
		VkImageCreateInfo image_info = {
		    .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		    .imageType = VK_IMAGE_TYPE_2D,
		    .format = VK_FORMAT_R16G16B16A16_SFLOAT,
		    .extent = {width, height, 1},
		    .mipLevels = 1,
		    .arrayLayers = 1,
		    .samples = VK_SAMPLE_COUNT_1_BIT,
		    .tiling = VK_IMAGE_TILING_OPTIMAL,
		    .usage =
		        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
		    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		    .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		};

		VkExternalMemoryImageCreateInfo external_info = {
		    .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
#ifdef _WIN32
		    .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,
#else
		    .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
#endif
		};
		image_info.pNext = &external_info;

		VkResult ret = vk->vkCreateImage(vk->device, &image_info, NULL, &r->illixr_motion_vectors[i].image);
		if (ret != VK_SUCCESS) {
			COMP_ERROR(c, "Failed to create motion vector image %u: %d", i, ret);
			return;
		}

		VkMemoryRequirements mem_reqs;
		vk->vkGetImageMemoryRequirements(vk->device, r->illixr_motion_vectors[i].image, &mem_reqs);

		uint32_t memory_type_index;
		if (!vk_get_memory_type(vk, mem_reqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		                        &memory_type_index)) {
			COMP_ERROR(c, "No suitable memory type for motion vector image %u", i);
			return;
		}

		VkExportMemoryAllocateInfo export_alloc = {
		    .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
#ifdef _WIN32
		    .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,
#else
		    .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
#endif
		};
		VkMemoryAllocateInfo alloc_info = {
		    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		    .pNext = &export_alloc,
		    .allocationSize = mem_reqs.size,
		    .memoryTypeIndex = memory_type_index,
		};

		ret = vk->vkAllocateMemory(vk->device, &alloc_info, NULL, &r->illixr_motion_vectors[i].memory);
		if (ret != VK_SUCCESS) {
			COMP_ERROR(c, "Failed to allocate motion vector memory %u: %d", i, ret);
			return;
		}

		ret = vk->vkBindImageMemory(vk->device, r->illixr_motion_vectors[i].image,
		                            r->illixr_motion_vectors[i].memory, 0);
		if (ret != VK_SUCCESS) {
			COMP_ERROR(c, "Failed to bind motion vector memory %u: %d", i, ret);
			return;
		}

		VkImageViewCreateInfo view_info = {
		    .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		    .image = r->illixr_motion_vectors[i].image,
		    .viewType = VK_IMAGE_VIEW_TYPE_2D,
		    .format = VK_FORMAT_R16G16B16A16_SFLOAT,
		    .subresourceRange =
		        {
		            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
		            .levelCount = 1,
		            .layerCount = 1,
		        },
		};
		ret = vk->vkCreateImageView(vk->device, &view_info, NULL, &r->illixr_motion_vectors[i].view);
		if (ret != VK_SUCCESS) {
			COMP_ERROR(c, "Failed to create motion vector image view %u: %d", i, ret);
			return;
		}

		r->illixr_motion_vectors[i].memory_size = mem_reqs.size;
		r->illixr_motion_vectors[i].width = width;
		r->illixr_motion_vectors[i].height = height;
	}
	COMP_INFO(c, "Created %d motion vector images", 2 * OFFLOAD_BUFFER_POOL_SIZE);
}

static void
create_illixr_color_downsampled_images(struct comp_renderer *r, uint32_t width, uint32_t height)
{
	struct comp_compositor *c = r->c;
	struct vk_bundle *vk = &c->base.vk;

	COMP_INFO(c, "Creating ILLIXR color downsampled images: %ux%u", width, height);

	for (uint32_t i = 0; i < 2 * OFFLOAD_BUFFER_POOL_SIZE; i++) {
		// Create image
		VkImageCreateInfo image_info = {
		    .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		    .imageType = VK_IMAGE_TYPE_2D,
		    .format = VK_FORMAT_R8G8B8A8_UNORM,
		    .extent = {width, height, 1},
		    .mipLevels = 1,
		    .arrayLayers = 1,
		    .samples = VK_SAMPLE_COUNT_1_BIT,
		    .tiling = VK_IMAGE_TILING_OPTIMAL,
		    .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT |
		             VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
		             VK_IMAGE_USAGE_SAMPLED_BIT,
		    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		    .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		};

		VkExternalMemoryImageCreateInfo external_info = {
		    .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
#ifdef _WIN32
		    .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,
#else
		    .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
#endif
		};
		image_info.pNext = &external_info;

		VkResult ret = vk->vkCreateImage(vk->device, &image_info, NULL,
		                                 &r->illixr_color_downsampled[i].image);
		if (ret != VK_SUCCESS) {
			COMP_ERROR(c, "Failed to create color downsampled image %u: %d", i, ret);
			return;
		}

		// Get memory requirements
		VkMemoryRequirements mem_reqs;
		vk->vkGetImageMemoryRequirements(vk->device, r->illixr_color_downsampled[i].image, &mem_reqs);

		// Find memory type
		uint32_t memory_type_index;
		bool found = vk_get_memory_type(vk, mem_reqs.memoryTypeBits,
		                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		                                &memory_type_index);
		if (!found) {
			COMP_ERROR(c, "Failed to find suitable memory type");
			return;
		}

		// Allocate exportable memory
		VkExportMemoryAllocateInfo export_alloc = {
		    .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
#ifdef _WIN32
		    .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,
#else
		    .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
#endif
		};

		VkMemoryAllocateInfo alloc_info = {
		    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		    .pNext = &export_alloc,
		    .allocationSize = mem_reqs.size,
		    .memoryTypeIndex = memory_type_index,
		};

		ret = vk->vkAllocateMemory(vk->device, &alloc_info, NULL,
		                           &r->illixr_color_downsampled[i].memory);
		if (ret != VK_SUCCESS) {
			COMP_ERROR(c, "Failed to allocate color memory %u: %d", i, ret);
			return;
		}

		// Bind memory
		ret = vk->vkBindImageMemory(vk->device, r->illixr_color_downsampled[i].image,
		                            r->illixr_color_downsampled[i].memory, 0);
		if (ret != VK_SUCCESS) {
			COMP_ERROR(c, "Failed to bind color memory %u: %d", i, ret);
			return;
		}

		// Create image view
		VkImageViewCreateInfo view_info = {
		    .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		    .image = r->illixr_color_downsampled[i].image,
		    .viewType = VK_IMAGE_VIEW_TYPE_2D,
		    .format = VK_FORMAT_R8G8B8A8_UNORM,
		    .subresourceRange = {
		        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
		        .baseMipLevel = 0,
		        .levelCount = 1,
		        .baseArrayLayer = 0,
		        .layerCount = 1,
		    },
		};

		ret = vk->vkCreateImageView(vk->device, &view_info, NULL,
		                            &r->illixr_color_downsampled[i].view);
		if (ret != VK_SUCCESS) {
			COMP_ERROR(c, "Failed to create color view %u: %d", i, ret);
			return;
		}

		// Store size info
		r->illixr_color_downsampled[i].memory_size = mem_reqs.size;
		r->illixr_color_downsampled[i].width = width;
		r->illixr_color_downsampled[i].height = height;
	}

	COMP_INFO(c, "Created %d color downsampled images", 2 * OFFLOAD_BUFFER_POOL_SIZE);
}

static void
create_illixr_depth_downsampled_images(struct comp_renderer *r, uint32_t width, uint32_t height)
{
	struct comp_compositor *c = r->c;
	struct vk_bundle *vk = &c->base.vk;

	COMP_INFO(c, "Creating ILLIXR depth downsampled images: %ux%u", width, height);

	for (uint32_t i = 0; i < 2 * OFFLOAD_BUFFER_POOL_SIZE; i++) {
		// Create depth image
		VkImageCreateInfo image_info = {
		    .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		    .imageType = VK_IMAGE_TYPE_2D,
		    .format = VK_FORMAT_D16_UNORM,  // 16-bit depth
		    .extent = {width, height, 1},
		    .mipLevels = 1,
		    .arrayLayers = 1,
		    .samples = VK_SAMPLE_COUNT_1_BIT,
		    .tiling = VK_IMAGE_TILING_OPTIMAL,
		    .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT |
		             VK_IMAGE_USAGE_SAMPLED_BIT,  // For compute shader input
		    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		    .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		};

		VkResult ret = vk->vkCreateImage(vk->device, &image_info, NULL,
		                                 &r->illixr_depth_downsampled[i].image);
		if (ret != VK_SUCCESS) {
			COMP_ERROR(c, "Failed to create depth downsampled image %u: %d", i, ret);
			return;
		}

		// Get memory requirements
		VkMemoryRequirements mem_reqs;
		vk->vkGetImageMemoryRequirements(vk->device, r->illixr_depth_downsampled[i].image, &mem_reqs);

		// Find memory type
		uint32_t memory_type_index;
		bool found = vk_get_memory_type(vk, mem_reqs.memoryTypeBits,
		                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		                                &memory_type_index);
		if (!found) {
			COMP_ERROR(c, "Failed to find suitable memory type for depth");
			return;
		}

		// Allocate memory
		VkMemoryAllocateInfo alloc_info = {
		    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		    .allocationSize = mem_reqs.size,
		    .memoryTypeIndex = memory_type_index,
		};

		ret = vk->vkAllocateMemory(vk->device, &alloc_info, NULL,
		                           &r->illixr_depth_downsampled[i].memory);
		if (ret != VK_SUCCESS) {
			COMP_ERROR(c, "Failed to allocate depth memory %u: %d", i, ret);
			return;
		}

		// Bind memory
		ret = vk->vkBindImageMemory(vk->device, r->illixr_depth_downsampled[i].image,
		                            r->illixr_depth_downsampled[i].memory, 0);
		if (ret != VK_SUCCESS) {
			COMP_ERROR(c, "Failed to bind depth memory %u: %d", i, ret);
			return;
		}

		// Create image view
		VkImageViewCreateInfo view_info = {
		    .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		    .image = r->illixr_depth_downsampled[i].image,
		    .viewType = VK_IMAGE_VIEW_TYPE_2D,
		    .format = VK_FORMAT_D16_UNORM,
		    .subresourceRange = {
		        .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
		        .baseMipLevel = 0,
		        .levelCount = 1,
		        .baseArrayLayer = 0,
		        .layerCount = 1,
		    },
		};

		ret = vk->vkCreateImageView(vk->device, &view_info, NULL,
		                            &r->illixr_depth_downsampled[i].view);
		if (ret != VK_SUCCESS) {
			COMP_ERROR(c, "Failed to create depth view %u: %d", i, ret);
			return;
		}

		// Store size info
		r->illixr_depth_downsampled[i].memory_size = mem_reqs.size;
		r->illixr_depth_downsampled[i].width = width;
		r->illixr_depth_downsampled[i].height = height;
	}

	COMP_INFO(c, "Created %d depth downsampled images", 2 * OFFLOAD_BUFFER_POOL_SIZE);
}
#endif

/*!
 * @pre render_gfx_init(render, &c->nr)
 */
static XRT_CHECK_RESULT VkResult
dispatch_graphics(struct comp_renderer *r,
                  struct render_gfx *render,
                  struct comp_render_scratch_state *crss,
                  enum comp_target_fov_source fov_source)
{
	COMP_TRACE_MARKER();

	struct comp_compositor *c = r->c;
	struct vk_bundle *vk = &c->base.vk;
	VkResult ret;

#ifdef USE_MONADO_ILLIXR_DRIVER
	int8_t illixr_buffer_index = -1;
#endif
	// Basics
	const struct comp_layer *layers = c->base.layer_accum.layers;
	uint32_t layer_count = c->base.layer_accum.layer_count;
	bool fast_path = c->base.frame_params.one_projection_layer_fast_path;
#ifdef USE_MONADO_ILLIXR_DRIVER
	// bool do_timewarp = illixr_offload_frames() && !c->debug.atw_off;
	bool do_timewarp = false;
#else
	bool do_timewarp = !c->debug.atw_off;
#endif


	// Resources for the distortion render target.
	struct render_gfx_target_resources *rtr = &r->rtr_array[r->acquired_buffer];

	// Consistency check.
	assert(!fast_path || c->base.layer_accum.layer_count >= 1);

	// Viewport information.
	struct render_viewport_data viewport_datas[XRT_MAX_VIEWS];
	calc_viewport_data(r, viewport_datas, render->r->view_count);

	// Vertex rotation information.
	struct xrt_matrix_2x2 vertex_rots[XRT_MAX_VIEWS];
	calc_vertex_rot_data(r, vertex_rots, render->r->view_count);

	// Device view information.
	struct xrt_fov fovs[XRT_MAX_VIEWS];
	struct xrt_pose world_poses[XRT_MAX_VIEWS];
	struct xrt_pose eye_poses[XRT_MAX_VIEWS];
	calc_pose_data(             //
	    r,                      //
	    fov_source,             //
	    fovs,                   //
	    world_poses,            //
	    eye_poses,              //
	    render->r->view_count); //

#ifdef USE_MONADO_ILLIXR_DRIVER
	// ILLIXR: Extract and update poses from the projection layer
	const struct comp_layer *proj_layer = NULL;
	for (uint32_t i = 0; i < layer_count; i++) {
		if (layers[i].data.type == XRT_LAYER_PROJECTION || layers[i].data.type == XRT_LAYER_PROJECTION_DEPTH) {
			proj_layer = &layers[i];
			break;
		}
	}
#ifdef USE_MONADO_ILLIXR_DRIVER
	if (proj_layer != NULL) {
		static bool size_logged = false;
		if (!size_logged) {
			for (uint32_t eye = 0; eye < 2; eye++) {
				struct comp_swapchain *comp_sc = (struct comp_swapchain *)proj_layer->sc_array[eye];
				if (comp_sc) {
					COMP_WARN(c, "ILLIXR Unity swapchain eye=%u: %ux%u", eye,
					          comp_sc->vkic.info.width, comp_sc->vkic.info.height);
				}
			}
			size_logged = true;
		}
	}
#endif
	// ILLIXR: Scan layers for the sentinel quad (position.z < -9998).
	// Purpose: capture g_illixr_mv_sc on the first arrival.
	// The sentinel is filtered out of the display layer list so it is not
	// passed to comp_render_gfx_dispatch (which would try to composite it).
	const struct comp_layer filtered_layers_storage[XRT_MAX_LAYERS];
	const struct comp_layer *filtered_layers = layers;
	uint32_t filtered_count = layer_count;

	{
		uint32_t n = 0;
		for (uint32_t i = 0; i < layer_count; i++) {
			// COMP_WARN(c, "Checking layer %d: type %d == %d, %.1f > 9998.0", i, (int)layers[i].data.type,
			// (int)XRT_LAYER_QUAD,
			//           layers[i].data.quad.size.x);
			if (layers[i].data.type == XRT_LAYER_QUAD &&
			    layers[i].data.quad.size.x > (ILLIXR_MV_SENTINEL_SIZE - 1.0f)) {
				// Sentinel quad: capture swapchain pointer if not yet done.
				//	COMP_WARN(c, "  set sc pointer:  %p == NULL  %p != NULL", g_illixr_mv_sc,
				//	          layers[i].sc_array[0]);
				if (g_illixr_mv_sc == NULL && layers[i].sc_array[0] != NULL) {
					g_illixr_mv_sc = (struct comp_swapchain *)layers[i].sc_array[0];
					//		COMP_WARN(c, "ILLIXR: Captured MV swapchain from sentinel: %ux%u
					//sc=%p", 		          g_illixr_mv_sc->vkic.info.width, +g_illixr_mv_sc->vkic.info.height,
					//		          (void *)g_illixr_mv_sc);
				}
				// Do NOT add to filtered list — exclude from compositing.
			} else {
				((struct comp_layer *)filtered_layers_storage)[n++] = layers[i];
			}
		}
		if (n < layer_count) {
			filtered_layers = filtered_layers_storage;
			filtered_count = n;
		}
	}

	// Read MV ring-buffer index from shared memory (written by hook DLL after
	// each xrReleaseSwapchainImage for the MV swapchain).
	if (g_illixr_shmem == NULL)
		illixr_shmem_open();

	uint32_t mv_ring_index = 0;
	bool mv_shmem_valid = false;
	// COMP_WARN(c, "ILLIXR shm:  %p   %p", (void *)g_illixr_shmem, (void *)g_illixr_mv_sc);
	if (g_illixr_shmem != NULL && g_illixr_mv_sc != NULL) {
		// Two-read protocol: seq before and after ring_index read.
		// On x86 32-bit aligned reads are naturally atomic so a single
		// read suffices, but reading seq twice is clearer.
		uint32_t seq_before = g_illixr_shmem->frame_seq;
		uint32_t ring = g_illixr_shmem->ring_index;
		uint32_t seq_after = g_illixr_shmem->frame_seq;

		// COMP_WARN(c, "    check: %d == %d  && %d != %d && %d < %d", seq_before, seq_after, seq_before,
		//           g_illixr_last_seq, ring, g_illixr_mv_sc->vkic.image_count);
		if (seq_before == seq_after                     // no torn write
		    && seq_before != g_illixr_last_seq          // new frame
		    && ring < g_illixr_mv_sc->vkic.image_count) // valid index
		{
			mv_ring_index = ring;
			mv_shmem_valid = true;
			g_illixr_last_seq = seq_before;
		}
	} else if (g_illixr_mv_sc == NULL) {
		// COMP_WARN(c, "ILLIXR: g_illixr_mv_sc still NULL - waiting for sentinel quad layer from Unity");
	}

	if (proj_layer) {
		static bool rect_logged = false;
		if (!rect_logged) {
			for (uint32_t eye = 0; eye < 2; eye++) {
				if (proj_layer->data.type == XRT_LAYER_PROJECTION) {
					struct xrt_layer_projection_view_data *vd = &proj_layer->data.proj.v[eye];
					COMP_WARN(
					    c,
					    "Unity imageRect eye=%u: offset=(%d,%d) extent=(%u,%u) "
					    "swapchain=(%u,%u)",
					    eye, vd->sub.rect.offset.w, vd->sub.rect.offset.h, vd->sub.rect.extent.w,
					    vd->sub.rect.extent.h,
					    ((struct comp_swapchain *)proj_layer->sc_array[eye])->vkic.info.width,
					    ((struct comp_swapchain *)proj_layer->sc_array[eye])->vkic.info.height);
				} else {
					struct xrt_layer_projection_view_data *vd = &proj_layer->data.depth.v[eye];
					COMP_WARN(
					    c,
					    "Unity imageRect eye=%u: offset=(%d,%d) extent=(%u,%u) "
					    "swapchain=(%u,%u)",
					    eye, vd->sub.rect.offset.w, vd->sub.rect.offset.h, vd->sub.rect.extent.w,
					    vd->sub.rect.extent.h,
					    ((struct comp_swapchain *)proj_layer->sc_array[eye])->vkic.info.width,
					    ((struct comp_swapchain *)proj_layer->sc_array[eye])->vkic.info.height);
				}
			}
			rect_logged = true;
		}

		struct xrt_pose left_pose;
		struct xrt_pose right_pose;

		if (proj_layer->data.type == XRT_LAYER_PROJECTION) {
			left_pose = proj_layer->data.proj.v[0].pose;
			right_pose = proj_layer->data.proj.v[1].pose;
		} else { // XRT_LAYER_PROJECTION_DEPTH
			left_pose = proj_layer->data.depth.v[0].pose;
			right_pose = proj_layer->data.depth.v[1].pose;
		}

		illixr_tw_update_uniforms(left_pose, right_pose);
	} else {
		COMP_WARN(c, "ILLIXR: No projection layer found for pose update");
	}
#endif
	// The arguments for the dispatch function.
	struct comp_render_dispatch_data data;
	comp_render_gfx_initial_init( //
	    &data,                    // data
	    rtr,                      // rtr
	    fast_path,                // fast_path
	    do_timewarp);             // do_timewarp
	for (uint32_t i = 0; i < render->r->view_count; i++) {
		// Which image of the scratch images for this view are we using.
		uint32_t scratch_index = crss->views[i].index;

		// The set of scratch images we are using for this view.
		struct comp_scratch_single_images *scratch_view = &c->scratch.views[i];

		// The render target resources for the scratch images.
		struct render_gfx_target_resources *rsci_rtr = &r->scratch.views[i].targets[scratch_index];

		// Scratch color image.
		struct render_scratch_color_image *rsci = &scratch_view->images[scratch_index];

		// Use the whole scratch image.
		struct render_viewport_data layer_viewport_data = {
		    .x = 0,
		    .y = 0,
		    .w = scratch_view->info.width,
		    .h = scratch_view->info.height,
		};

		// Scratch image covers the whole image.
		struct xrt_normalized_rect layer_norm_rect = {.x = 0.0f, .y = 0.0f, .w = 1.0f, .h = 1.0f};

		comp_render_gfx_add_view( //
		    &data,                //
		    &world_poses[i],      //
		    &eye_poses[i],        //
		    &fovs[i],             //
		    rsci_rtr,             //
		    &layer_viewport_data, //
		    &layer_norm_rect,     //
		    rsci->image,          //
		    rsci->srgb_view,      //
		    &vertex_rots[i],      //
		    &viewport_datas[i]);  // target_viewport_data

		if (layer_count == 0) {
			crss->views[i].used = false;
		} else {
			crss->views[i].used = !fast_path;
		}
	}

	// Start the graphics pipeline.
	render_gfx_begin(render);

#ifdef USE_MONADO_ILLIXR_DRIVER
	if (fast_path && proj_layer != NULL) {
		// Unity uses a texture array swapchain (both eyes in one VkImage,
		// array_index 0=left, 1=right). The mesh shader expects a plain
		// sampler2D and cannot bind an array image view directly.
		// Blit each eye's array layer into its per-eye 2D scratch image,
		// then composite through the identity distortion mesh from there.
		for (uint32_t eye = 0; eye < 2; eye++) {
			uint32_t image_index = (proj_layer->data.type == XRT_LAYER_PROJECTION)
			                           ? proj_layer->data.proj.v[eye].sub.image_index
			                           : proj_layer->data.depth.v[eye].sub.image_index;
			uint32_t array_index = (proj_layer->data.type == XRT_LAYER_PROJECTION)
			                           ? proj_layer->data.proj.v[eye].sub.array_index
			                           : proj_layer->data.depth.v[eye].sub.array_index;

			struct comp_swapchain *comp_sc = (struct comp_swapchain *)proj_layer->sc_array[eye];
			VkImage src_image = comp_sc->vkic.images[image_index].handle;
			uint32_t src_w = comp_sc->vkic.info.width;
			uint32_t src_h = comp_sc->vkic.info.height;

			uint32_t scratch_index = crss->views[eye].index;
			struct comp_scratch_single_images *scratch_view = &c->scratch.views[eye];
			struct render_scratch_color_image *scratch_image = &scratch_view->images[scratch_index];
			uint32_t dst_w = scratch_view->info.width;
			uint32_t dst_h = scratch_view->info.height;

			// Unity swapchain: SHADER_READ_ONLY → TRANSFER_SRC (one array layer).
			VkImageMemoryBarrier barrier = {
			    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			    .srcAccessMask = VK_ACCESS_SHADER_READ_BIT,
			    .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
			    .oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			    .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			    .image = src_image,
			    .subresourceRange =
			        {
			            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			            .baseMipLevel = 0,
			            .levelCount = 1,
			            .baseArrayLayer = array_index,
			            .layerCount = 1,
			        },
			};
			vk->vkCmdPipelineBarrier(render->r->cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &barrier);

			// Scratch image: COLOR_ATTACHMENT_OPTIMAL → TRANSFER_DST_OPTIMAL.
			barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			barrier.image = scratch_image->image;
			barrier.subresourceRange.baseArrayLayer = 0;
			vk->vkCmdPipelineBarrier(render->r->cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &barrier);

			// Blit the correct array layer into the plain 2D scratch image.
			VkImageBlit blit = {
			    .srcSubresource =
			        {
			            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			            .mipLevel = 0,
			            .baseArrayLayer = array_index,
			            .layerCount = 1,
			        },
			    .srcOffsets = {{0, 0, 0}, {src_w, src_h, 1}},
			    .dstSubresource =
			        {
			            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			            .mipLevel = 0,
			            .baseArrayLayer = 0,
			            .layerCount = 1,
			        },
			    .dstOffsets = {{0, 0, 0}, {dst_w, dst_h, 1}},
			};
			vk->vkCmdBlitImage(render->r->cmd, src_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			                   scratch_image->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
			                   VK_FILTER_LINEAR);

			// Unity swapchain back to SHADER_READ_ONLY.
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			barrier.image = src_image;
			barrier.subresourceRange.baseArrayLayer = array_index;
			vk->vkCmdPipelineBarrier(render->r->cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
			                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL, 0, NULL, 1,
			                         &barrier);

			// Scratch image to SHADER_READ_ONLY for the distortion pass.
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			barrier.image = scratch_image->image;
			barrier.subresourceRange.baseArrayLayer = 0;
			vk->vkCmdPipelineBarrier(render->r->cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
			                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL, 0, NULL, 1,
			                         &barrier);

			crss->views[eye].used = true;
		}

		// Composite from the scratch images through the identity distortion
		// mesh. Bypasses the layer squasher and its pose-based MVP warp.
		comp_render_gfx_distortion_from_scratch(render, &data);
		goto illixr_gfx_dispatch_done;
	}
#endif
	comp_render_gfx_dispatch(render, filtered_layers, filtered_count, &data);

#ifdef USE_MONADO_ILLIXR_DRIVER
illixr_gfx_dispatch_done:;
#endif

#ifdef USE_MONADO_ILLIXR_DRIVER
	if (strcmp(r->c->xdev->str, "ILLIXR") == 0 && !illixr_offload_frames()) {
		// COMP_INFO(c, "ILLIXR: Acquiring buffer for encoding");

		illixr_buffer_index = illixr_src_acquire();

		if (illixr_buffer_index < 0) {
			COMP_WARN(c, "ILLIXR: No buffer available, skipping frame");
		} else {
			// COMP_INFO(c, "ILLIXR: Acquired buffer index: %d", illixr_buffer_index);
			//  Create downsampled images if not already created
			if (!r->illixr_downsampled_created) {
				// Target encoding resolution (no scaling)
				uint32_t target_width = r->c->xdev->hmd->views[0].display.w_pixels;
				uint32_t target_height = r->c->xdev->hmd->views[0].display.h_pixels;

				uint32_t depth_width = MOTION_VECTOR_WIDTH;
				uint32_t depth_height = MOTION_VECTOR_HEIGHT;

				// Create color downsampled images
				create_illixr_color_downsampled_images(r, target_width, target_height);

				// Create depth downsampled images
				create_illixr_depth_downsampled_images(r, depth_width, depth_height);

				// Create RG depth images
				create_illixr_depth_rg_images(r, depth_width, depth_height);

				// Create depth-to-RG pipeline and descriptors
				create_depth_to_rg_pipeline(r);
				create_depth_to_rg_descriptors(r);

				// Pre-populate ALL framebuffer slots for color and depth immediately
				// after image creation.  nvenc_import_buffer_pool_images runs once on
				// the first src_release, which may arrive before every buffer slot has
				// been rendered into.  Slots whose fb->image is still VK_NULL_HANDLE are
				// skipped entirely (the continue guard in the import loop), so those
				// encoder indices stay at -1 and later encode calls warn "not imported".
				for (int i = 0; i < 2 * OFFLOAD_BUFFER_POOL_SIZE; i++) {
					// COLOR
					r->illixr_framebuffers[i].image = r->illixr_color_downsampled[i].image;
					r->illixr_framebuffers[i].memory = r->illixr_color_downsampled[i].memory;
					r->illixr_framebuffers[i].view = r->illixr_color_downsampled[i].view;
					r->illixr_framebuffers[i].image_size =
					    r->illixr_color_downsampled[i].memory_size;
					r->illixr_framebuffers[i].image_offset = 0;
					r->illixr_framebuffers[i].image_extent.width =
					    r->illixr_color_downsampled[i].width;
					r->illixr_framebuffers[i].image_extent.height =
					    r->illixr_color_downsampled[i].height;
					// DEPTH (RG image, always valid when depth is enabled)
					r->illixr_framebuffers[i].depth_image = r->illixr_depth_rg[i].image;
					r->illixr_framebuffers[i].depth_memory = r->illixr_depth_rg[i].memory;
					r->illixr_framebuffers[i].depth_view = r->illixr_depth_rg[i].view;
					r->illixr_framebuffers[i].depth_size = r->illixr_depth_rg[i].memory_size;
					r->illixr_framebuffers[i].depth_offset = 0;
					r->illixr_framebuffers[i].depth_extent.width = r->illixr_depth_rg[i].width;
					r->illixr_framebuffers[i].depth_extent.height = r->illixr_depth_rg[i].height;
				}

				// Create motion vector images eagerly so that all buffer-pool
				// slots have valid VkImage handles before the server calls
				// nvenc_import_buffer_pool_images() on the first src_release.
				// Previously these were lazy-created on the first blit, which
				// meant the one-shot import ran before any sentinel quad had
				// been submitted and saw VK_NULL_HANDLE for every slot.
				// MOTION_VECTOR_WIDTH/HEIGHT are compile-time constants so we
				// do not need Unity's swapchain dimensions here.
				create_illixr_motion_vector_images(r, MOTION_VECTOR_WIDTH, MOTION_VECTOR_HEIGHT);
				for (int i = 0; i < 2 * OFFLOAD_BUFFER_POOL_SIZE; i++) {
					r->illixr_framebuffers[i].motion_vec_image = r->illixr_motion_vectors[i].image;
					r->illixr_framebuffers[i].motion_vec_memory =
					    r->illixr_motion_vectors[i].memory;
					r->illixr_framebuffers[i].motion_vec_view = r->illixr_motion_vectors[i].view;
					r->illixr_framebuffers[i].motion_vec_size =
					    r->illixr_motion_vectors[i].memory_size;
					r->illixr_framebuffers[i].motion_vec_offset = 0;
					r->illixr_framebuffers[i].motion_vec_extent.width = MOTION_VECTOR_WIDTH;
					r->illixr_framebuffers[i].motion_vec_extent.height = MOTION_VECTOR_HEIGHT;
				}

				r->illixr_downsampled_created = true;
			}
			// Fast path: read directly from the Unity projection layer swapchain image
			// (in SHADER_READ_ONLY_OPTIMAL after the distortion pass sampled it).
			// Slow path: read from the scratch image written by the layer squasher
			// (in COLOR_ATTACHMENT_OPTIMAL).
			for (uint32_t eye = 0; eye < 2; eye++) {
				int fb_idx = illixr_buffer_index * 2 + eye;

				// Determine source image and its current layout.
				VkImage src_image;
				uint32_t src_width, src_height;
				VkAccessFlags src_access_before;
				VkAccessFlags src_access_after;
				VkImageLayout src_layout_before;
				VkImageLayout src_layout_after;
				VkPipelineStageFlags src_stage_before;
				VkPipelineStageFlags src_stage_after;

				if (fast_path && proj_layer != NULL) {
					// Unity swapchain image — already in SHADER_READ_ONLY after
					// the distortion pass sampled it.
					uint32_t image_index;
					if (proj_layer->data.type == XRT_LAYER_PROJECTION) {
						image_index = proj_layer->data.proj.v[eye].sub.image_index;
					} else {
						image_index = proj_layer->data.depth.v[eye].sub.image_index;
					}
					struct comp_swapchain *comp_sc =
					    (struct comp_swapchain *)proj_layer->sc_array[eye];
					src_image = comp_sc->vkic.images[image_index].handle;
					src_width = comp_sc->vkic.info.width;
					src_height = comp_sc->vkic.info.height;
					src_access_before = VK_ACCESS_SHADER_READ_BIT;
					src_access_after = VK_ACCESS_SHADER_READ_BIT;
					src_layout_before = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
					src_layout_after = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
					src_stage_before = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
					src_stage_after = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
				} else {
					// Scratch image written by the layer squasher.
					uint32_t scratch_index = crss->views[eye].index;
					struct comp_scratch_single_images *scratch_view = &c->scratch.views[eye];
					struct render_scratch_color_image *scratch_image =
					    &scratch_view->images[scratch_index];
					src_image = scratch_image->image;
					src_width = scratch_view->info.width;
					src_height = scratch_view->info.height;
					src_access_before = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
					src_access_after = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
					src_layout_before = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
					src_layout_after = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
					src_stage_before = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
					src_stage_after = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
				}

				// For array swapchains (fast path), read the correct layer.
				// For non-array sources (slow path) this is always 0.
				uint32_t enc_array_index = (fast_path && proj_layer != NULL)
				                               ? ((proj_layer->data.type == XRT_LAYER_PROJECTION)
				                                      ? proj_layer->data.proj.v[eye].sub.array_index
				                                      : proj_layer->data.depth.v[eye].sub.array_index)
				                               : 0;

				// Transition source to TRANSFER_SRC
				VkImageMemoryBarrier barrier = {
				    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
				    .srcAccessMask = src_access_before,
				    .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
				    .oldLayout = src_layout_before,
				    .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				    .image = src_image,
				    .subresourceRange =
				        {
				            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				            .levelCount = 1,
				            .baseArrayLayer = enc_array_index,
				            .layerCount = 1,
				        },
				};

				vk->vkCmdPipelineBarrier(render->r->cmd, src_stage_before,
				                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1,
				                         &barrier);

				// Transition downsampled to TRANSFER_DST
				barrier.srcAccessMask = 0;
				barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
				barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
				barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
				barrier.image = r->illixr_color_downsampled[fb_idx].image;

				vk->vkCmdPipelineBarrier(render->r->cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
				                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1,
				                         &barrier);

				// Blit (downsample) source → downsampled
				VkImageBlit blit = {
				    .srcSubresource =
				        {
				            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				            .baseArrayLayer = enc_array_index,
				            .layerCount = 1,
				        },
				    .srcOffsets =
				        {
				            {0, 0, 0},
				            {src_width, src_height, 1},
				        },
				    .dstSubresource =
				        {
				            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				            .layerCount = 1,
				        },
				    .dstOffsets =
				        {
				            {0, 0, 0},
				            {r->illixr_color_downsampled[fb_idx].width,
				             r->illixr_color_downsampled[fb_idx].height, 1},
				        },
				};

				vk->vkCmdBlitImage(render->r->cmd, src_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				                   r->illixr_color_downsampled[fb_idx].image,
				                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

				// Transition source back to its original layout
				barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
				barrier.dstAccessMask = src_access_after;
				barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
				barrier.newLayout = src_layout_after;
				barrier.image = src_image;
				barrier.subresourceRange.baseArrayLayer = enc_array_index;

				vk->vkCmdPipelineBarrier(render->r->cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
				                         src_stage_after, 0, 0, NULL, 0, NULL, 1, &barrier);
				/*
				// Source: Scratch image for this eye (COLOR)
				uint32_t scratch_index = crss->views[eye].index;
				struct comp_scratch_single_images *scratch_view = &c->scratch.views[eye];
				struct render_scratch_color_image *scratch_image = &scratch_view->images[scratch_index];

				int fb_idx = illixr_buffer_index * 2 + eye;

				// Transition scratch to TRANSFER_SRC
				VkImageMemoryBarrier barrier = {
				    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
				    .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
				    .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
				    .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
				    .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				    .image = scratch_image->image,
				    .subresourceRange = {
				        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				        .levelCount = 1,
				        .layerCount = 1,
				    },
				};

				vk->vkCmdPipelineBarrier(render->r->cmd,
				                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
				                         VK_PIPELINE_STAGE_TRANSFER_BIT,
				                         0, 0, NULL, 0, NULL, 1, &barrier);

				// Transition downsampled to TRANSFER_DST
				barrier.srcAccessMask = 0;
				barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
				barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
				barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
				barrier.image = r->illixr_color_downsampled[fb_idx].image;

				vk->vkCmdPipelineBarrier(render->r->cmd,
				                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
				                         VK_PIPELINE_STAGE_TRANSFER_BIT,
				                         0, 0, NULL, 0, NULL, 1, &barrier);

				// Blit (downsample) scratch → downsampled
				VkImageBlit blit = {
				    .srcSubresource = {
				        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				        .layerCount = 1,
				    },
				    .srcOffsets = {
				        {0, 0, 0},
				        {scratch_view->info.width, scratch_view->info.height, 1},
				    },
				    .dstSubresource = {
				        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				        .layerCount = 1,
				    },
				    .dstOffsets = {
				        {0, 0, 0},
				        {r->illixr_color_downsampled[fb_idx].width,
				            r->illixr_color_downsampled[fb_idx].height, 1},
				    },
				};

				vk->vkCmdBlitImage(render->r->cmd,
				                   scratch_image->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				                   r->illixr_color_downsampled[fb_idx].image,
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

				// Transition scratch back
				barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
				barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
				barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
				barrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
				barrier.image = scratch_image->image;

				vk->vkCmdPipelineBarrier(render->r->cmd,
				                         VK_PIPELINE_STAGE_TRANSFER_BIT,
				                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
				                         0, 0, NULL, 0, NULL, 1, &barrier);
				                         */
				// Transition downsampled to SHADER_READ (for NVENC)
				barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
				barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
				barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
				barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				barrier.image = r->illixr_color_downsampled[fb_idx].image;

				vk->vkCmdPipelineBarrier(render->r->cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
				                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL, 0, NULL, 1,
				                         &barrier);

				// Populate COLOR fields
				r->illixr_framebuffers[fb_idx].image = r->illixr_color_downsampled[fb_idx].image;
				r->illixr_framebuffers[fb_idx].memory = r->illixr_color_downsampled[fb_idx].memory;
				r->illixr_framebuffers[fb_idx].view = r->illixr_color_downsampled[fb_idx].view;
				r->illixr_framebuffers[fb_idx].image_extent.width =
				    r->illixr_color_downsampled[fb_idx].width;
				r->illixr_framebuffers[fb_idx].image_extent.height =
				    r->illixr_color_downsampled[fb_idx].height;
				r->illixr_framebuffers[fb_idx].image_size =
				    r->illixr_color_downsampled[fb_idx].memory_size;
				r->illixr_framebuffers[fb_idx].image_offset = 0;

				// COMP_INFO(c, "ILLIXR: Eye=%d, scratch_view: %ux%u, scratch_image handle: %p", eye,
				//           scratch_view->info.width, scratch_view->info.height,
				//           (void *)scratch_image->image);

				// Populate DEPTH fields from Unity's submitted layer
				if (proj_layer != NULL && proj_layer->data.type == XRT_LAYER_PROJECTION_DEPTH) {
					uint32_t depth_sc_index = 2 + eye;
					struct xrt_swapchain *depth_swapchain = proj_layer->sc_array[depth_sc_index];

					if (depth_swapchain != NULL) {
						uint32_t depth_image_index =
						    proj_layer->data.depth.d[eye].sub.array_index;
						struct comp_swapchain *comp_sc =
						    (struct comp_swapchain *)depth_swapchain;

						if (depth_image_index < depth_swapchain->image_count) {
							VkImage unity_depth_src =
							    comp_sc->vkic.images[depth_image_index].handle;

							// Step 1: Downsample Unity's depth to our depth buffer
							VkImageMemoryBarrier barrier = {
							    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
							    .srcAccessMask = 0,
							    .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
							    .oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
							    .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
							    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
							    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
							    .image = unity_depth_src,
							    .subresourceRange =
							        {
							            .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
							            .baseMipLevel = 0,
							            .levelCount = 1,
							            .baseArrayLayer = 0,
							            .layerCount = 1,
							        },
							};

							vk->vkCmdPipelineBarrier(render->r->cmd,
							                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
							                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
							                         NULL, 0, NULL, 1, &barrier);

							// Transition our downsampled depth to TRANSFER_DST
							barrier.srcAccessMask = 0;
							barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
							barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
							barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
							barrier.image = r->illixr_depth_downsampled[fb_idx].image;

							vk->vkCmdPipelineBarrier(render->r->cmd,
							                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
							                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
							                         NULL, 0, NULL, 1, &barrier);

							// Blit depth (downsample with NEAREST filter).
							// Source extents come from Unity's depth swapchain, which is
							// full render resolution and unrelated to scratch or color
							// downsampled size. Destination extents come from
							// illixr_depth_downsampled, which is sized to
							// MOTION_VECTOR_WIDTH x MOTION_VECTOR_HEIGHT.
							VkImageBlit depth_blit = {
							    .srcSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1},
							    .srcOffsets = {{0, 0, 0},
							                   {comp_sc->vkic.info.width,
							                    comp_sc->vkic.info.height, 1}},
							    .dstSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1},
							    .dstOffsets = {{0, 0, 0},
							                   {r->illixr_depth_downsampled[fb_idx].width,
							                    r->illixr_depth_downsampled[fb_idx].height,
							                    1}},
							};

							vk->vkCmdBlitImage(render->r->cmd, unity_depth_src,
							                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
							                   r->illixr_depth_downsampled[fb_idx].image,
							                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
							                   &depth_blit, VK_FILTER_NEAREST);

							// Step 2: Convert depth to RG with compute shader
							// Transition depth to shader read
							barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
							barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
							barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
							barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
							barrier.image = r->illixr_depth_downsampled[fb_idx].image;

							vk->vkCmdPipelineBarrier(render->r->cmd,
							                         VK_PIPELINE_STAGE_TRANSFER_BIT,
							                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
							                         0, 0, NULL, 0, NULL, 1, &barrier);

							// Transition RG output to general
							barrier.srcAccessMask = 0;
							barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
							barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
							barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
							barrier.image = r->illixr_depth_rg[fb_idx].image;
							barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

							vk->vkCmdPipelineBarrier(render->r->cmd,
							                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
							                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
							                         0, 0, NULL, 0, NULL, 1, &barrier);

							// Dispatch compute shader (depth16 → RG8)
							vk->vkCmdBindPipeline(render->r->cmd,
							                      VK_PIPELINE_BIND_POINT_COMPUTE,
							                      r->depth_to_rg_pipeline);
							vk->vkCmdBindDescriptorSets(
							    render->r->cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
							    r->depth_to_rg_layout, 0, 1,
							    &r->depth_to_rg_desc_sets[fb_idx], 0, NULL);

							uint32_t group_x = (r->illixr_depth_rg[fb_idx].width + 15) / 16;
							uint32_t group_y =
							    (r->illixr_depth_rg[fb_idx].height + 15) / 16;
							vk->vkCmdDispatch(render->r->cmd, group_x, group_y, 1);

							// Transition RG to transfer src for export
							barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
							barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
							barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
							barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
							barrier.image = r->illixr_depth_rg[fb_idx].image;

							vk->vkCmdPipelineBarrier(render->r->cmd,
							                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
							                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
							                         NULL, 0, NULL, 1, &barrier);

							// Step 3: Export RG depth to framebuffers (for ILLIXR encoder)
							r->illixr_framebuffers[fb_idx].depth_image =
							    r->illixr_depth_rg[fb_idx].image;
							r->illixr_framebuffers[fb_idx].depth_memory =
							    r->illixr_depth_rg[fb_idx].memory;
							r->illixr_framebuffers[fb_idx].depth_view =
							    r->illixr_depth_rg[fb_idx].view;
							r->illixr_framebuffers[fb_idx].depth_extent.width =
							    r->illixr_depth_rg[fb_idx].width;
							r->illixr_framebuffers[fb_idx].depth_extent.height =
							    r->illixr_depth_rg[fb_idx].height;
							r->illixr_framebuffers[fb_idx].depth_size =
							    r->illixr_depth_rg[fb_idx].memory_size;
							r->illixr_framebuffers[fb_idx].depth_offset = 0;

							// Projection clip planes from XrCompositionLayerDepthInfoKHR.
							// Unity writes these into the depth layer; forward them so the
							// decoder can linearise depth values back into view-space
							// metres.
							r->illixr_framebuffers[fb_idx].near_z =
							    proj_layer->data.depth.d[eye].near_z;
							r->illixr_framebuffers[fb_idx].far_z =
							    proj_layer->data.depth.d[eye].far_z;

							// Calculate buffer_idx for logging
							uint32_t buffer_idx = fb_idx / 2;
							// COMP_INFO(c, "ILLIXR: Processed depth for buffer %d eye %d
							// (D16→RG)", buffer_idx, eye);
						}
					}
				} else {
					// No depth layer - clear depth fields
					r->illixr_framebuffers[fb_idx].depth_image = VK_NULL_HANDLE;
					r->illixr_framebuffers[fb_idx].depth_memory = VK_NULL_HANDLE;
					r->illixr_framebuffers[fb_idx].depth_view = VK_NULL_HANDLE;
					r->illixr_framebuffers[fb_idx].depth_size = 0;
					r->illixr_framebuffers[fb_idx].depth_offset = 0;
					r->illixr_framebuffers[fb_idx].depth_extent.width = 0;
					r->illixr_framebuffers[fb_idx].depth_extent.height = 0;
					r->illixr_framebuffers[fb_idx].near_z = 0.0f;
					r->illixr_framebuffers[fb_idx].far_z = 0.0f;

					if (eye == 0) { // Only log once
						COMP_WARN(c, "ILLIXR: No depth layer (type=%d)",
						          proj_layer ? proj_layer->data.type : -1);
					}
				}

				// ILLIXR: Extract motion vectors from the MV swapchain via
				// shared memory ring index.
				if (mv_shmem_valid && g_illixr_mv_sc != NULL) {
					// if (eye == 0) {
					// COMP_WARN(c, "ILLIXR: MV shmem valid: ring=%u seq=%u", mv_ring_index,
					//           g_illixr_last_seq);

					//}
					{
						uint32_t mv_img_idx = mv_ring_index;
						struct comp_swapchain *mv_comp_sc = g_illixr_mv_sc;
						struct xrt_swapchain *mv_sc = &mv_comp_sc->base.base;

						if (mv_img_idx < mv_sc->image_count) {
							VkImage unity_mv_src =
							    mv_comp_sc->vkic.images[mv_img_idx].handle;
							uint32_t mv_w = mv_comp_sc->vkic.info.width / 2; // per-eye half
							uint32_t mv_h = mv_comp_sc->vkic.info.height;

							// Motion vector images are created eagerly in the
							// illixr_downsampled_created block above; no lazy-create
							// needed.

							// Transition Unity's swapchain image → TRANSFER_SRC
							VkImageMemoryBarrier mv_barrier = {
							    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
							    .srcAccessMask = VK_ACCESS_NONE,
							    .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
							    .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
							    .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
							    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
							    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
							    .image = unity_mv_src,
							    .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
							};
							vk->vkCmdPipelineBarrier(render->r->cmd,
							                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
							                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
							                         NULL, 0, NULL, 1, &mv_barrier);

							// Transition our motion vector image → TRANSFER_DST
							mv_barrier.srcAccessMask = 0;
							mv_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
							mv_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
							mv_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
							mv_barrier.image = r->illixr_motion_vectors[fb_idx].image;
							vk->vkCmdPipelineBarrier(render->r->cmd,
							                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
							                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
							                         NULL, 0, NULL, 1, &mv_barrier);

							// Copy the eye-half out of the side-by-side Unity texture
							VkImageBlit mv_blit = {
							    .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
							    .srcOffsets =
							        {
							            {(int32_t)(eye * (int32_t)mv_w), 0, 0},
							            {(int32_t)(eye * (int32_t)mv_w) + (int32_t)mv_w,
							             (int32_t)mv_h, 1},
							        },
							    .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
							    .dstOffsets =
							        {
							            {0, 0, 0},
							            {MOTION_VECTOR_WIDTH, MOTION_VECTOR_HEIGHT, 1},
							        },
							};

							vk->vkCmdBlitImage(render->r->cmd, unity_mv_src,
							                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
							                   r->illixr_motion_vectors[fb_idx].image,
							                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
							                   &mv_blit, VK_FILTER_LINEAR);

							// Transition our motion vector image → SHADER_READ for
							// downstream use
							mv_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
							mv_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
							mv_barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
							mv_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
							mv_barrier.image = r->illixr_motion_vectors[fb_idx].image;
							vk->vkCmdPipelineBarrier(render->r->cmd,
							                         VK_PIPELINE_STAGE_TRANSFER_BIT,
							                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
							                         0, 0, NULL, 0, NULL, 1, &mv_barrier);

							// Populate framebuffer struct for downstream consumers
							r->illixr_framebuffers[fb_idx].motion_vec_image =
							    r->illixr_motion_vectors[fb_idx].image;
							r->illixr_framebuffers[fb_idx].motion_vec_memory =
							    r->illixr_motion_vectors[fb_idx].memory;
							r->illixr_framebuffers[fb_idx].motion_vec_view =
							    r->illixr_motion_vectors[fb_idx].view;
							r->illixr_framebuffers[fb_idx].motion_vec_extent.width =
							    MOTION_VECTOR_WIDTH;
							r->illixr_framebuffers[fb_idx].motion_vec_extent.height =
							    MOTION_VECTOR_HEIGHT;
							r->illixr_framebuffers[fb_idx].motion_vec_size =
							    r->illixr_motion_vectors[fb_idx].memory_size;
							r->illixr_framebuffers[fb_idx].motion_vec_offset = 0;
						}
					}
				} else {
					if (eye == 0) {
						// COMP_WARN(c, "ILLIXR: NO MV data (shmem_valid=%d sc=%p)",
						//           (int)mv_shmem_valid, (void *)g_illixr_mv_sc);
					}
				}
				// Note: when mv_layer is NULL (no sentinel quad this frame) the
				// motion_vec_image handle is left intact.  The VkImage was
				// allocated once in the illixr_downsampled_created block and is
				// valid for the lifetime of the renderer; nulling it here would
				// prevent nvenc_import_buffer_pool_images from importing it on
				// frames that precede Unity's first sentinel quad submission.
				// COMP_DEBUG(c, "ILLIXR: Framebuffer %d - color=%p, depth=%p", fb_idx,
				//           (void *)scratch_image->image,
				//           (void *)r->illixr_framebuffers[fb_idx].depth_image);
			}
		}
	}
#endif

	// Make the command buffer submittable.
	render_gfx_end(render);

	// Everything is ready, submit to the queue.
	ret = renderer_submit_queue(r, render->r->cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
	VK_CHK_AND_RET(ret, "renderer_submit_queue");
#ifdef USE_MONADO_ILLIXR_DRIVER
	/*
	 * ILLIXR: Wait for GPU and release buffer to encoder
	 */
	if (illixr_buffer_index >= 0 && strcmp(r->c->xdev->str, "ILLIXR") == 0) {
		// COMP_INFO(c, "ILLIXR: Waiting for GPU to complete all work");

		// Wait for ALL GPU work to complete before releasing to encoder
		vk->vkQueueWaitIdle(vk->queue);

		/* for (int i = 0; i < unity_save_index; i++) {
		        if (!unity_raw_saves[i].pending)
		                continue;

		        uint32_t width = unity_raw_saves[i].width;
		        uint32_t height = unity_raw_saves[i].height;
		        VkDeviceSize buffer_size = width * height * 4;

		        void *data;
		        vk->vkMapMemory(vk->device, unity_raw_saves[i].memory, 0, buffer_size, 0, &data);

		        // Save to PPM in CURRENT WORKING DIRECTORY
		        char filename[256];
		        snprintf(filename, sizeof(filename), "D:/unity_raw_%03d_eye%d.ppm", unity_save_count,
		                 unity_raw_saves[i].eye);

		        FILE *f = fopen(filename, "wb");
		        if (f) {
		                fprintf(f, "P6\n%d %d\n255\n", width, height);

		                // Convert RGBA to RGB
		                uint8_t *pixels = (uint8_t *)data;
		                for (uint32_t j = 0; j < width * height; j++) {
		                        fwrite(&pixels[j * 4], 1, 3, f); // R,G,B (skip A)
		                }

		                fclose(f);
		                COMP_INFO(c, "ILLIXR: Saved Unity raw image to %s", filename);
		        } else {
		                COMP_ERROR(c, "ILLIXR: Failed to open %s for writing", filename);
		        }

		        vk->vkUnmapMemory(vk->device, unity_raw_saves[i].memory);

		        // Cleanup
		        vk->vkDestroyBuffer(vk->device, unity_raw_saves[i].buffer, NULL);
		        vk->vkFreeMemory(vk->device, unity_raw_saves[i].memory, NULL);

		        unity_raw_saves[i].pending = false;
		}

		if (unity_save_index > 0) {
		        unity_save_count++;
		        unity_save_index = 0; // Reset for next frame
		}*/

		// COMP_INFO(c, "ILLIXR: GPU work complete, releasing buffer for encoding");

		// Extract poses from projection layer
		struct xrt_pose left_pose = {.orientation = {.x = 0, .y = 0, .z = 0, .w = 1}};
		struct xrt_pose right_pose = {.orientation = {.x = 0, .y = 0, .z = 0, .w = 1}};

		// Find projection layer and extract poses
		const struct comp_layer *proj_layer = NULL;
		for (uint32_t i = 0; i < c->base.layer_accum.layer_count; i++) {
			if (c->base.layer_accum.layers[i].data.type == XRT_LAYER_PROJECTION ||
			    c->base.layer_accum.layers[i].data.type == XRT_LAYER_PROJECTION_DEPTH) {
				proj_layer = &c->base.layer_accum.layers[i];
				break;
			}
		}

		if (proj_layer) {
			if (proj_layer->data.type == XRT_LAYER_PROJECTION) {
				left_pose = proj_layer->data.proj.v[0].pose;
				right_pose = proj_layer->data.proj.v[1].pose;
			} else { // XRT_LAYER_PROJECTION_DEPTH
				left_pose = proj_layer->data.depth.v[0].pose;
				right_pose = proj_layer->data.depth.v[1].pose;
			}
		} else {
			COMP_WARN(c, "ILLIXR: No projection layer found for pose update");
		}

		// Release buffer to encoder (now safe - GPU is done)
		illixr_src_release(illixr_buffer_index, left_pose, right_pose);

		// COMP_INFO(c, "ILLIXR: Buffer %d released successfully", illixr_buffer_index);
	}
#endif
	return ret;
}


/*
 *
 * Compute
 *
 */

/*!
 * @pre render_compute_init(render, &c->nr)
 */
static XRT_CHECK_RESULT VkResult
dispatch_compute(struct comp_renderer *r,
                 struct render_compute *render,
                 struct comp_render_scratch_state *crss,
                 enum comp_target_fov_source fov_source)
{
	COMP_TRACE_MARKER();

	struct comp_compositor *c = r->c;
	struct vk_bundle *vk = &c->base.vk;
	VkResult ret;

	// Basics
	const struct comp_layer *layers = c->base.layer_accum.layers;
	uint32_t layer_count = c->base.layer_accum.layer_count;
	bool fast_path = c->base.frame_params.one_projection_layer_fast_path;
#ifdef USE_MONADO_ILLIXR_DRIVER
	bool do_timewarp = false;
#else
	bool do_timewarp = !c->debug.atw_off;
#endif

	// Device view information.
	struct xrt_fov fovs[XRT_MAX_VIEWS];
	struct xrt_pose world_poses[XRT_MAX_VIEWS];
	struct xrt_pose eye_poses[XRT_MAX_VIEWS];
	calc_pose_data(             //
	    r,                      //
	    fov_source,             //
	    fovs,                   //
	    world_poses,            //
	    eye_poses,              //
	    render->r->view_count); //

	// Target Vulkan resources..
	VkImage target_image = r->c->target->images[r->acquired_buffer].handle;
	VkImageView target_image_view = r->c->target->images[r->acquired_buffer].view;

	// Target view information.
	struct render_viewport_data views[XRT_MAX_VIEWS];
	calc_viewport_data(r, views, render->r->view_count);

	// The arguments for the dispatch function.
	struct comp_render_dispatch_data data;
	comp_render_cs_initial_init( //
	    &data,                   // data
	    target_image,            // target_image
	    target_image_view,       // target_unorm_view
	    fast_path,               // fast_path
	    do_timewarp);            // do_timewarp

	for (uint32_t i = 0; i < render->r->view_count; i++) {
		// Which image of the scratch images for this view are we using.
		uint32_t scratch_index = crss->views[i].index;

		// The set of scratch images we are using for this view.
		struct comp_scratch_single_images *scratch_view = &c->scratch.views[i];

		// Scratch color image.
		struct render_scratch_color_image *rsci = &scratch_view->images[scratch_index];

		// Use the whole scratch image.
		struct render_viewport_data layer_viewport_data = {
		    .x = 0,
		    .y = 0,
		    .w = scratch_view->info.width,
		    .h = scratch_view->info.height,
		};

		// Scratch image covers the whole image.
		struct xrt_normalized_rect layer_norm_rect = {.x = 0.0f, .y = 0.0f, .w = 1.0f, .h = 1.0f};

		comp_render_cs_add_view(  //
		    &data,                //
		    &world_poses[i],      //
		    &eye_poses[i],        //
		    &fovs[i],             //
		    &layer_viewport_data, //
		    &layer_norm_rect,     //
		    rsci->image,          //
		    rsci->srgb_view,      //
		    rsci->unorm_view,     //
		    &views[i]);           // target_viewport_data

		if (layer_count == 0) {
			crss->views[i].used = false;
		} else {
			crss->views[i].used = !fast_path;
		}
	}

	// Start the compute pipeline.
	render_compute_begin(render);

	// Build the command buffer.
	comp_render_cs_dispatch( //
	    render,              //
	    layers,              //
	    layer_count,         //
	    &data);              //

	// Make the command buffer submittable.
	render_compute_end(render);

	// Everything is ready, submit to the queue.
	ret = renderer_submit_queue(r, render->r->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	VK_CHK_AND_RET(ret, "renderer_submit_queue");

	return ret;
}


/*
 *
 * Interface functions.
 *
 */

XRT_CHECK_RESULT xrt_result_t
comp_renderer_draw(struct comp_renderer *r)
{
	COMP_TRACE_MARKER();

	struct comp_target *ct = r->c->target;
	struct comp_compositor *c = r->c;

	// Check that we don't have any bad data.
	assert(!comp_frame_is_invalid_locked(&c->frame.waited));
	assert(comp_frame_is_invalid_locked(&c->frame.rendering));

	// Move waited frame to rendering frame, clear waited.
	comp_frame_move_and_clear_locked(&c->frame.rendering, &c->frame.waited);

	// Tell the target we are starting to render, for frame timing.
	comp_target_mark_begin(ct, c->frame.rendering.id, os_monotonic_get_ns());

	// Are we ready to render? No - skip rendering.
	if (!comp_target_check_ready(r->c->target)) {
		// Need to emulate rendering for the timing.
		//! @todo This should be discard.
		comp_target_mark_submit_begin(ct, c->frame.rendering.id, os_monotonic_get_ns());
		comp_target_mark_submit_end(ct, c->frame.rendering.id, os_monotonic_get_ns());

		// Clear the rendering frame.
		comp_frame_clear_locked(&c->frame.rendering);
		return XRT_SUCCESS;
	}

	comp_target_flush(ct);

	comp_target_update_timings(ct);

	if (r->acquired_buffer < 0) {
		// Ensures that renderings are created.
		renderer_acquire_swapchain_image(r);
	}

	comp_target_update_timings(ct);

	// Hardcoded for now.
	const uint32_t view_count = c->nr.view_count;
	enum comp_target_fov_source fov_source = COMP_TARGET_FOV_SOURCE_DISTORTION;
#ifdef USE_MONADO_ILLIXR_DRIVER
	if (strcmp(c->xdev->str, "ILLIXR") == 0) {
		fov_source = COMP_TARGET_FOV_SOURCE_DEVICE_VIEWS;
	}
#endif

	// For scratch image debugging.
	struct comp_render_scratch_state crss;
	scratch_get_init(&crss, r, view_count);

	bool use_compute = r->settings->use_compute;
	struct render_gfx render_g = {0};
	struct render_compute render_c = {0};

	VkResult res = VK_SUCCESS;
	if (use_compute) {
		render_compute_init(&render_c, &c->nr);
		res = dispatch_compute(r, &render_c, &crss, fov_source);
	} else {
		render_gfx_init(&render_g, &c->nr);
		res = dispatch_graphics(r, &render_g, &crss, fov_source);
	}
	if (res != VK_SUCCESS) {
		return XRT_ERROR_VULKAN;
	}

#ifdef XRT_FEATURE_WINDOW_PEEK
	if (c->peek) {
		switch (comp_window_peek_get_eye(c->peek)) {
		case COMP_WINDOW_PEEK_EYE_LEFT: {
			struct comp_scratch_single_images *view = &c->scratch.views[0];
			comp_window_peek_blit(                       //
			    c->peek,                                 //
			    view->images[crss.views[0].index].image, //
			    view->info.width,                        //
			    view->info.height);                      //
		} break;
		case COMP_WINDOW_PEEK_EYE_RIGHT: {
			struct comp_scratch_single_images *view = &c->scratch.views[1];
			comp_window_peek_blit(                       //
			    c->peek,                                 //
			    view->images[crss.views[1].index].image, //
			    view->info.width,                        //
			    view->info.height);                      //
		} break;
		case COMP_WINDOW_PEEK_EYE_BOTH:
			/* TODO: display the undistorted image */
			comp_window_peek_blit(c->peek, c->target->images[r->acquired_buffer].handle, c->target->width,
			                      c->target->height);
			break;
		}
	}
#endif

	renderer_present_swapchain_image(r, c->frame.rendering.desired_present_time_ns,
	                                 c->frame.rendering.present_slop_ns);

	// Save for timestamps below.
	uint64_t frame_id = c->frame.rendering.id;
	uint64_t desired_present_time_ns = c->frame.rendering.desired_present_time_ns;
	uint64_t predicted_display_time_ns = c->frame.rendering.predicted_display_time_ns;

	// Clear the rendered frame.
	comp_frame_clear_locked(&c->frame.rendering);

	xrt_result_t xret = XRT_SUCCESS;
	comp_mirror_fixup_ui_state(&r->mirror_to_debug_gui, c);
	if (comp_mirror_is_ready_and_active(&r->mirror_to_debug_gui, c, predicted_display_time_ns)) {

		struct comp_scratch_single_images *view = &c->scratch.views[0];
		struct render_scratch_color_image *rsci = &view->images[crss.views[0].index];
		VkExtent2D extent = {view->info.width, view->info.width};

		// Used for both, want clamp to edge to no bring in black.
		VkSampler clamp_to_edge = c->nr.samplers.clamp_to_edge;

		// Covers the whole view.
		struct xrt_normalized_rect rect = {0, 0, 1.0f, 1.0f};

		xret = comp_mirror_do_blit(    //
		    &r->mirror_to_debug_gui,   //
		    &c->base.vk,               //
		    frame_id,                  //
		    predicted_display_time_ns, //
		    rsci->image,               //
		    rsci->srgb_view,           //
		    clamp_to_edge,             //
		    extent,                    //
		    rect);                     //
	}

	/*
	 * This fixes a lot of validation issues as it makes sure that the
	 * command buffer has completed and all resources referred by it can
	 * now be manipulated.
	 *
	 * This is done after a swap so isn't time critical.
	 */
	renderer_wait_queue_idle(r);
#ifdef USE_MONADO_ILLIXR_DRIVER
	//flush_scratch_image_saves(r);
#endif
	// Finalize the scratch images, send to debug UI if active.
	scratch_get_fini(&crss, r, view_count);

	// Check timestamps.
	if (xret == XRT_SUCCESS) {
		/*
		 * Get timestamps of GPU work (if available).
		 */

		uint64_t gpu_start_ns, gpu_end_ns;
		if (render_resources_get_timestamps(&c->nr, &gpu_start_ns, &gpu_end_ns)) {
			uint64_t now_ns = os_monotonic_get_ns();
			comp_target_info_gpu(ct, frame_id, gpu_start_ns, gpu_end_ns, now_ns);
		}
	}


	/*
	 * Free resources.
	 */

	if (use_compute) {
		render_compute_fini(&render_c);
	} else {
		render_gfx_fini(&render_g);
	}


	/*
	 * For direct mode this makes us wait until the last frame has been
	 * actually shown to the user, this avoids us missing that we have
	 * missed a frame and miss-predicting the next frame.
	 *
	 * Only do this if we are ready.
	 */
	if (comp_target_check_ready(r->c->target)) {
		// For estimating frame misses.
		uint64_t then_ns = os_monotonic_get_ns();

		// Do the acquire
		renderer_acquire_swapchain_image(r);

		// How long did it take?
		uint64_t now_ns = os_monotonic_get_ns();

		/*
		 * Make sure we at least waited 1ms before warning. Then check
		 * if we are more then 1ms behind when we wanted to present.
		 */
		if (then_ns + U_TIME_1MS_IN_NS < now_ns && //
		    desired_present_time_ns + U_TIME_1MS_IN_NS < now_ns) {
			uint64_t diff_ns = now_ns - desired_present_time_ns;
			double diff_ms_f = time_ns_to_ms_f(diff_ns);
			COMP_WARN(c, "Compositor probably missed frame by %.2fms", diff_ms_f);
		}
	}

	comp_target_update_timings(ct);

	return xret;
}

struct comp_renderer *
comp_renderer_create(struct comp_compositor *c, VkExtent2D scratch_extent)
{
	struct comp_renderer *r = U_TYPED_CALLOC(struct comp_renderer);

	renderer_init(r, c, scratch_extent);

	return r;
}

void
comp_renderer_destroy(struct comp_renderer **ptr_r)
{
	if (ptr_r == NULL) {
		return;
	}

	struct comp_renderer *r = *ptr_r;
	if (r == NULL) {
		return;
	}

	renderer_fini(r);

	free(r);
	*ptr_r = NULL;
}

void
comp_renderer_add_debug_vars(struct comp_renderer *self)
{
	struct comp_renderer *r = self;

	comp_mirror_add_debug_vars(&r->mirror_to_debug_gui, r->c);
}
