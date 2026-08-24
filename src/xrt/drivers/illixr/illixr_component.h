// Copyright 2020-2021, The Board of Trustees of the University of Illinois.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  ILLIXR plugin
 * @author RSIM Group <illixr@cs.illinois.edu>
 * @ingroup drv_illixr
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

#include "illixr_framebuffer.h"

#include "util/u_string_list.h"
#include "xrt/xrt_defines.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef USING_OPENXR
/*
 *
 * Hand-joint tracking constants and structures
 *
 */

/**
 * @brief Number of joints per hand (matches OpenXR XR_HAND_JOINT_COUNT_EXT)
 */
#define ILLIXR_HAND_JOINT_COUNT 26

/**
 * @brief Joint tracking data for both hands.
 */
struct illixr_hand_tracking_data
{
	struct xrt_hand_joint_set left_hand;  //!< Left hand joint data
	struct xrt_hand_joint_set right_hand; //!< Right hand joint data
	bool valid;                           //!< Whether this struct contains valid data
};
/**
 * @brief A single hand interaction pose with its accompanying scalar inputs.
 *
 * Used for AIM, GRIP, PINCH, and POKE pose types.  @c value and @c ready are
 * meaningful only for AIM, GRIP, and PINCH; they are always 0 / false for POKE.
 */
struct illixr_interaction_pose
{
	struct xrt_space_relation relation;
	float value; //!< Gesture-strength scalar in [0, 1]
	bool ready;  //!< Whether the gesture is currently activatable
	bool valid;  //!< Whether this pose is valid
};

#define ILLIXR_INTERACTION_AIM 0   /*!< /input/aim/pose + aim_activate_ext scalars   */
#define ILLIXR_INTERACTION_GRIP 1  /*!< /input/grip/pose + grasp_ext scalars          */
#define ILLIXR_INTERACTION_PINCH 2 /*!< /input/pinch_ext/pose + pinch_ext scalars     */
#define ILLIXR_INTERACTION_POKE 3  /*!< /input/poke_ext/pose (no scalars)             */
#define ILLIXR_NUM_INTERACTION_POSES 4

/**
 * @brief All four interaction poses for one hand, indexed by ILLIXR_INTERACTION_*.
 */
struct illixr_hand_interaction_data
{
	struct illixr_interaction_pose poses[ILLIXR_NUM_INTERACTION_POSES];
	bool valid; //!< Whether any pose data is available for this hand
};
#endif
/*
 *
 * Plugin lifecycle functions
 *
 */

/**
 * @brief Create the ILLIXR plugin and register with the phonebook.
 * @param pb Phonebook pointer
 * @return Plugin instance pointer
 */
void *
illixr_monado_create_plugin(void *pb);

/**
 * @brief Block until ILLIXR Vulkan display-service initialization is complete.
 */
void
illixr_monado_wait_for_init(void);

/*
 *
 * Pose tracking functions
 *
 */

/**
 * @brief Read the current head pose from ILLIXR
 * @return Current head pose
 */
struct xrt_pose
illixr_read_pose(void);

#ifdef USING_OPENXR
/*
 *
 * Hand-tracking functions (XR_EXT_hand_tracking)
 *
 */

/**
 * @brief Returns true if hand-joint tracking data is available on the switchboard.
 */
bool
illixr_hand_tracking_supported(void);

/**
 * @brief Read joint tracking data for both hands.
 * @param out_data Output struct to populate; out_data->valid set to false on failure
 * @return true if valid data was written
 */
bool
illixr_read_hand_tracking(struct illixr_hand_tracking_data *out_data);

/**
 * @brief Read joint tracking data for one hand.
 * @param hand      0 for left, 1 for right
 * @param out_hand  Output struct to populate
 * @return true if valid data was written
 */
bool
illixr_read_single_hand(int hand, struct xrt_hand_joint_set *out_hand);

/*
 *
 * Palm pose functions (XR_EXT_palm_pose)
 *
 * @param hand  0 = left, 1 = right
 */

/**
 * @brief Read palm poses for both hands from the switchboard.
 * @param out_poses Output struct to populate; out_poses->valid set to false on failure
 * @return true if valid data was written
 */
bool
illixr_read_palm_pose(int hand, struct xrt_space_relation *out_pose);

#endif

struct xrt_space_relation
illixr_read_head_relation(int64_t at_timestamp_ns);

/*
 *
 * Hand interaction functions (XR_EXT_hand_interaction)
 *
 * @param hand  0 = left, 1 = right
 */

#ifdef USING_OPENXR
/**
 * @brief Read hand-interaction poses for both hands from the switchboard.
 * @param out_interactions Output struct to populate; out_interactions->valid set to false on failure
 * @return true if valid data was written
 */
bool
illixr_read_hand_interaction(int hand, struct illixr_hand_interaction_data *out_data);

#endif

/*
 *
 * Vulkan/display functions
 *
 */

void
illixr_initialize_vulkan_display_service(VkInstance instance,
                                         VkPhysicalDevice physical_device,
                                         VkDevice device,
                                         VkQueue queue,
                                         uint32_t queue_family_index,
                                         struct u_string_list *enabled_instance_extensions,
                                         struct u_string_list *enabled_device_extensions);

void
illixr_initialize_timewarp(VkRenderPass render_pass,
                           uint32_t subpass,
                           VkExtent2D extent,
                           VkImage *image,
                           VkImageView *image_view,
                           VkDeviceMemory *device_memory,
                           VkDeviceSize *size,
                           VkDeviceSize *offset,
                           uint32_t num_buffers_per_eye,
                           struct illixr_framebuffer *framebuffer_array);

int8_t
illixr_src_acquire();
void
illixr_src_release(int8_t buffer_ind, struct xrt_pose l_pose, struct xrt_pose r_pose);
void
illixr_destroy_timewarp(void);
bool
illixr_offload_frames();
int
illixr_sleep_time();
void
illixr_tw_update_uniforms(struct xrt_pose l_pose, struct xrt_pose r_pose);
void
illixr_tw_record_command_buffer(VkCommandBuffer commandBuffer, VkFramebuffer framebuffer, int buffer_ind, int left);
void
illixr_publish_vsync_estimate(uint64_t display_time_ns);

VkExtent2D
illixr_get_extent(void);

#ifdef __cplusplus
}
#endif
