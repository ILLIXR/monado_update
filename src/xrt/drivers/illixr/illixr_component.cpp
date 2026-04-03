// Copyright 2020-2021, The Board of Trustees of the University of Illinois.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  ILLIXR plugin
 * @author RSIM Group <illixr@cs.illinois.edu>
 * @ingroup drv_illixr
 */

#define VMA_IMPLEMENTATION

#include <atomic>
#include <cassert>
#include <chrono>

#include "xrt/xrt_device.h"
#include "util/u_string_list.h"
#include "main/comp_renderer.h"

#include <memory>
#include <vulkan/vulkan.h>

#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#ifndef USING_OPENXR
#define USING_OPENXR
#endif
#include "illixr/phonebook.hpp"
#include "illixr/plugin.hpp"
#include "illixr/switchboard.hpp"
#include "illixr/data_format/pose_prediction.hpp"
#include "illixr/data_format/latency_data.hpp"
#include "illixr/data_format/poses/hand_interaction_pose.hpp"
#include "illixr/data_format/poses/hand_pose.hpp"
#include "illixr/data_format/poses/palm_pose.hpp"
#include "illixr/vk/display_provider.hpp"
#include "illixr/vk/render_pass.hpp"
#include "illixr/vk/vulkan_objects.hpp"

#include "illixr_component.h"

#include <cstdlib>
// Debug file logger - static global
#ifdef BUILD_WITH_LOGGING
static FILE *g_debug_log3 = NULL;

static void
init_debug_log(void)
{
	if (g_debug_log3 == NULL) {
		g_debug_log3 = fopen("D:\\illixr_comp_debug.log", "w");
		if (g_debug_log3) {
			fprintf(g_debug_log3, "=== Hand Tracking Debug Log Started ===\n");
			fflush(g_debug_log3);
		}
	}
}

static void
log_debug(const char *format, ...)
{
	if (g_debug_log3 == NULL) {
		init_debug_log();
	}

	if (g_debug_log3) {
		// Get timestamp
		time_t now = time(NULL);
		struct tm *tm_info = localtime(&now);
		char time_buf[64];
		strftime(time_buf, sizeof(time_buf), "%H:%M:%S", tm_info);

		// Write timestamp
		fprintf(g_debug_log3, "[%s] ", time_buf);

		// Write actual message
		va_list args;
		va_start(args, format);
		vfprintf(g_debug_log3, format, args);
		va_end(args);

		fprintf(g_debug_log3, "\n");
		fflush(g_debug_log3); // CRITICAL - force write immediately
	}
}
#endif

using namespace ILLIXR;
using namespace ILLIXR::vulkan;
using namespace ILLIXR::data_format;

const std::string PREFIX = "\e[0;32m[Monado ILLIXR]\e[0m ";

class monado_vulkan_display_provider : public display_provider
{
};

class monado_compositor_app : public app
{
};

static std::atomic<bool> _ds_ready = false;

// Simulated plugin class for an instance during phonebook registration
class illixr_plugin : public plugin
{
public:
	illixr_plugin(const std::string &name_, phonebook *pb_)
	    : plugin{name_, pb_}
	    , pb{pb_}
	    , sb{phonebook_->lookup_impl<switchboard>()}
	    , sb_pose{phonebook_->lookup_impl<pose_prediction>()}
	    , sb_clock{phonebook_->lookup_impl<relative_clock>()}
	    , ds{std::make_shared<monado_vulkan_display_provider>()}
	    , _m_vsync{sb->get_writer<switchboard::event_wrapper<time_point>>("vsync_estimate")}
	    , hand_pose_reader_{sb->get_reader<pose::hand_joint_poses_pair>("hand_poses")}
	    , hand_interaction_reader_{sb->get_reader<pose::hand_interaction_poses_pair>("hand_interactions")}
	    , palm_pose_reader_{sb->get_reader<pose::palm_poses_pair>("palm_poses")}
	    , latency_reader_{sb->get_reader<latency_ping>("latency_ping")} 
	{
		sb_timewarp = pb_->lookup_impl<timewarp>();

		if (std::getenv("ILLIXR_OFFLOAD_FRAMES") != nullptr) {
			offload_frames = std::stoi(std::getenv("ILLIXR_OFFLOAD_FRAMES"));
		}

		if (std::getenv("ILLIXR_COMPOSITOR_SLEEP_NS") != nullptr) {
			sleep_time = std::stoi(std::getenv("ILLIXR_COMPOSITOR_SLEEP_NS"));
		}

		// Check if hand tracking is enabled
		if (std::getenv("ILLIXR_USE_HAND_TRACKING") != nullptr) {
			std::string val = std::getenv("ILLIXR_USE_HAND_TRACKING");
			hand_tracking_enabled_ = (val == "1" || val == "true" || val == "TRUE");
		} else {
			// Default to enabled if offloading frames
			hand_tracking_enabled_ = offload_frames;
		}
		
		if (std::getenv("ILLIXR_USE_PALM_POSES") != nullptr) {
			std::string val = std::getenv("ILLIXR_USE_PALM_POSES");
			palm_poses_enabled_ = (val == "1" || val == "true" || val == "TRUE");
        }
        
		if (std::getenv("ILLIXR_USE_HAND_INTERACTIONS") != nullptr) {
			std::string val = std::getenv("ILLIXR_USE_HAND_INTERACTIONS");
			hand_interactions_enabled_ = (val == "1" || val == "true" || val == "TRUE");
        }
#ifdef BUILD_WITH_LOGGING
		log_debug("Hand tracking %s", (hand_tracking_enabled_ ? "enabled" : "disabled"));
#endif
	}

	std::atomic<bool> ready = false;

	bool offload_frames             = false;
	int  sleep_time                 = -1;
	bool hand_tracking_enabled_     = false;
	bool palm_poses_enabled_        = false;
	bool hand_interactions_enabled_ = false;

	phonebook *pb;
	const std::shared_ptr<switchboard>      sb;
	const std::shared_ptr<pose_prediction>  sb_pose;
	const std::shared_ptr<relative_clock>   sb_clock;
	std::shared_ptr<timewarp>               sb_timewarp;
	std::shared_ptr<vulkan::buffer_pool<BUFFER_TYPE>> buffer_pool;

	std::shared_ptr<display_provider> ds;
	switchboard::writer<switchboard::event_wrapper<time_point>> _m_vsync;

	// Pose data readers — topics published by offload_rendering_server
	switchboard::reader<pose::hand_joint_poses_pair>       hand_pose_reader_;
	switchboard::reader<pose::hand_interaction_poses_pair> hand_interaction_reader_;
	switchboard::reader<pose::palm_poses_pair>             palm_pose_reader_;
	switchboard::reader<data_format::latency_ping>         latency_reader_;

	BUFFER_TYPE last_pose;
};

static illixr_plugin *illixr_plugin_obj = nullptr;

extern "C" void *
illixr_monado_create_plugin(void *phonebook_)
{
	illixr_plugin_obj = new illixr_plugin{"illixr_plugin", static_cast<phonebook *>(phonebook_)};
	illixr_plugin_obj->start();
	return static_cast<void *>(illixr_plugin_obj);
}

extern "C" void
illixr_monado_wait_for_init(void)
{
	while (!_ds_ready) {
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}
}

extern "C" struct xrt_space_relation
illixr_read_head_relation(int64_t at_timestamp_ns)
{
	assert(illixr_plugin_obj && "illixr_plugin_obj must be initialized first.");

	struct xrt_space_relation relation = {};

	return illixr_plugin_obj->sb_pose->get_fast_pose(at_timestamp_ns);
}

/*
 *
 * Hand-joint tracking functions
 *
 */

extern "C" bool
illixr_hand_tracking_supported(void)
{
	if (!illixr_plugin_obj) {
		return false;
	}
	return illixr_plugin_obj->hand_tracking_enabled_;
}

/**
 * @brief Convert ILLIXR hand_joint_pose to illixr_hand_joint
 
static void
convert_joint(const pose::hand_joint_pose &src, struct illixr_hand_joint *dst)
{
	dst->position.x = src.pose.position.x;
	dst->position.y = src.pose.position.y;
	dst->position.z = src.pose.position.z;

	dst->orientation.x = src.pose.orientation.x;
	dst->orientation.y = src.pose.orientation.y;
	dst->orientation.z = src.pose.orientation.z;
	dst->orientation.w = src.pose.orientation.w;

	dst->radius = src.radius;

	dst->linear_velocity.x = src.linearVelocity.x;
	dst->linear_velocity.y = src.linearVelocity.y;
	dst->linear_velocity.z = src.linearVelocity.z;

	dst->angular_velocity.x = src.angularVelocity.x;
	dst->angular_velocity.y = src.angularVelocity.y;
	dst->angular_velocity.z = src.angularVelocity.z;

	dst->location_flags = 0;
	if (src.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT)
		dst->location_flags |= XRT_SPACE_RELATION_ORIENTATION_VALID_BIT;
	if (src.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT)
		dst->location_flags |= XRT_SPACE_RELATION_POSITION_VALID_BIT;
	if (src.locationFlags & XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT)
		dst->location_flags |= XRT_SPACE_RELATION_ORIENTATION_TRACKED_BIT;
	if (src.locationFlags & XR_SPACE_LOCATION_POSITION_TRACKED_BIT)
		dst->location_flags |= XRT_SPACE_RELATION_POSITION_TRACKED_BIT;
	if (src.velocityFlags & XR_SPACE_VELOCITY_LINEAR_VALID_BIT)
		dst->location_flags |= XRT_SPACE_RELATION_LINEAR_VELOCITY_VALID_BIT;
	if (src.velocityFlags & XR_SPACE_VELOCITY_ANGULAR_VALID_BIT)
		dst->location_flags |= XRT_SPACE_RELATION_ANGULAR_VELOCITY_VALID_BIT;
}*/

/**
 * @brief Convert ILLIXR single_hand_state to illixr_single_hand
 
static void
convert_single_hand(const pose::hand_joint_poses &src, struct illixr_single_hand *dst)
{
	dst->confidence = src.confidence;

	for (size_t i = 0; i < pose::HAND_JOINT_COUNT; ++i) {
		convert_joint(src.joints[i], &dst->joints[i]);
	}
}*/

extern "C" bool
illixr_read_single_hand(int hand, struct xrt_hand_joint_set *out_hand)
{
	assert(illixr_plugin_obj && "illixr_plugin_obj must be initialized first.");
	assert(out_hand && "out_hand must not be null.");
	assert(hand == 0 || hand == 1 && "hand must be 0 (left) or 1 (right).");

	if (!illixr_plugin_obj->hand_tracking_enabled_) {
		out_hand->is_active = false;
		return false;
	}

	// Try to get hand tracking data from the switchboard
	std::shared_ptr<const pose::hand_joint_poses_pair> hand_data =
	    illixr_plugin_obj->hand_pose_reader_.get_ro_nullable();

	if (!hand_data) {
		out_hand->is_active = false;
		return false;
	}

	*out_hand = (hand == 0) ? hand_data->hands.at(pose::LEFT) : hand_data->hands.at(pose::RIGHT);
	//const pose::hand_joint_poses &src = (hand == 0) ? hand_data->hands.at(pose::LEFT) : hand_data->hands.at(pose::RIGHT);

	//if (!src.confidence > 0.) {
	//	out_hand->is_active = false;
	//	return false;
	//}

	//convert_single_hand(src, out_hand);
	return true;
}

/*
 *
 * Palm-pose functions
 *
 */

extern "C" bool
illixr_read_palm_pose(int hand, struct xrt_space_relation *out_pose)
{
	assert(illixr_plugin_obj && "illixr_plugin_obj must be initialized first.");
	assert(out_pose && "out_pose must not be null.");
	assert((hand == 0 || hand == 1) && "hand must be 0 (left) or 1 (right).");

	//out_pose->is_active = false;

	if (!illixr_plugin_obj->hand_tracking_enabled_) {
		return false;
	}

	std::shared_ptr<const pose::palm_poses_pair> palm_data =
	    illixr_plugin_obj->palm_pose_reader_.get_ro_nullable();

	if (!palm_data || !palm_data->is_valid()) {
		return false;
	}

	const ILLIXR::data_format::pose::hand side =
	    (hand == 0) ? ILLIXR::data_format::pose::LEFT : ILLIXR::data_format::pose::RIGHT;

	*out_pose = palm_data->hands.at(side);

	if (out_pose->relation_flags == 0) {
		return false;
	}

	//out_pose->position.x    = src.position.x;
	//out_pose->position.y    = src.position.y;
	//out_pose->position.z    = src.position.z;
	//out_pose->orientation.x = src.orientation.x;
	//out_pose->orientation.y = src.orientation.y;
	//out_pose->orientation.z = src.orientation.z;
	//out_pose->orientation.w = src.orientation.w;
	//out_pose->valid         = true;
	return true;
}

/*
 *
 * Hand-interaction pose functions
 *
 */
extern "C" bool
illixr_read_hand_interaction(int hand, struct illixr_hand_interaction_data *out_data)
{
	assert(illixr_plugin_obj && "illixr_plugin_obj must be initialized first.");
	assert(out_data && "out_data must not be null.");
	assert((hand == 0 || hand == 1) && "hand must be 0 (left) or 1 (right).");

	out_data->valid = false;

	if (!illixr_plugin_obj->hand_tracking_enabled_) {
		return false;
	}

	std::shared_ptr<const pose::hand_interaction_poses_pair> interaction_data =
	    illixr_plugin_obj->hand_interaction_reader_.get_ro_nullable();

	if (!interaction_data || !interaction_data->is_valid()) {
		return false;
	}

	const ILLIXR::data_format::pose::hand side =
	    (hand == 0) ? ILLIXR::data_format::pose::LEFT : ILLIXR::data_format::pose::RIGHT;

	const pose::hand_interaction_poses &src = interaction_data->hands.at(side);

	// Helper to copy one interaction pose
	/* auto copy_pose = [](const pose::hand_interaction_pose &s,
	                    struct illixr_interaction_pose *d) {
		d->position.x    = s.position.x;
		d->position.y    = s.position.y;
		d->position.z    = s.position.z;
		d->orientation.x = s.orientation.x;
		d->orientation.y = s.orientation.y;
		d->orientation.z = s.orientation.z;
		d->orientation.w = s.orientation.w;
		d->value         = s.value;
		d->ready         = s.ready;
		d->valid         = s.valid();
	};

	copy_pose(src.at(pose::AIM),   &out_data->poses[ILLIXR_INTERACTION_AIM]);
	copy_pose(src.at(pose::GRIP),  &out_data->poses[ILLIXR_INTERACTION_GRIP]);
	copy_pose(src.at(pose::PINCH), &out_data->poses[ILLIXR_INTERACTION_PINCH]);
	copy_pose(src.at(pose::POKE),  &out_data->poses[ILLIXR_INTERACTION_POKE]);
	*/
	out_data->poses[ILLIXR_INTERACTION_AIM].relation = src.at(pose::AIM);
	out_data->poses[ILLIXR_INTERACTION_GRIP].relation = src.at(pose::GRIP);
	out_data->poses[ILLIXR_INTERACTION_PINCH].relation = src.at(pose::PINCH);
	out_data->poses[ILLIXR_INTERACTION_POKE].relation = src.at(pose::POKE);
	out_data->valid = src.is_valid();
	return out_data->valid;
}

/*
 *
 * Vulkan display service functions
 *
 */

extern "C" void
illixr_initialize_vulkan_display_service(VkInstance instance,
                                         VkPhysicalDevice physical_device,
                                         VkDevice device,
                                         VkQueue queue,
                                         uint32_t queue_family_index,
                                         struct u_string_list *enabled_instance_extensions,
                                         struct u_string_list *enabled_device_extensions)
{
	auto ds = std::make_shared<monado_vulkan_display_provider>();
	ds->vk_instance_ = instance;
	ds->vk_physical_device_ = physical_device;
	ds->vk_device_ = device;
	ds->queues_[queue::GRAPHICS] = {queue, queue_family_index, queue::GRAPHICS, std::make_shared<std::mutex>()};

	const char *const *exts      = u_string_list_get_data(enabled_instance_extensions);
	uint32_t           ext_count = u_string_list_get_size(enabled_instance_extensions);
	for (uint32_t i = 0; i < ext_count; i++) {
		ds->enabled_instance_extensions_.push_back(exts[i]);
	}

	const char *const *dev_exts      = u_string_list_get_data(enabled_device_extensions);
	uint32_t           dev_ext_count = u_string_list_get_size(enabled_device_extensions);
	for (uint32_t i = 0; i < dev_ext_count; i++) {
		ds->enabled_device_extensions_.push_back(dev_exts[i]);
	}

	_ds_ready = true;

	illixr_plugin_obj->pb->register_impl<display_provider>(std::static_pointer_cast<display_provider>(ds));
	illixr_plugin_obj->ds = ds;
}

extern "C" void
illixr_destroy_timewarp()
{
	assert(illixr_plugin_obj && "illixr_plugin_obj must be initialized first.");
	illixr_plugin_obj->sb_timewarp->destroy();
}

extern "C" void
illixr_initialize_timewarp(VkRenderPass render_pass,
                           uint32_t subpass,
                           VkExtent2D extent,
                           VkImage *image,
                           VkImageView *image_view,
                           VkDeviceMemory *device_memory,
                           VkDeviceSize *size,
                           VkDeviceSize *offset,
                           uint32_t num_buffers_per_eye,
                           struct illixr_framebuffer *framebuffer_array)
{
	assert(illixr_plugin_obj && "illixr_plugin_obj must be initialized first.");

	// Create empty buffer pool - images will be imported later
	std::vector<std::array<vulkan::vk_image, 2>> image_pool(num_buffers_per_eye);
	std::vector<std::array<vulkan::vk_image, 2>> depth_image_pool(num_buffers_per_eye);

	auto buffer_pool = std::make_shared<vulkan::buffer_pool<BUFFER_TYPE>>(image_pool, depth_image_pool);
	illixr_plugin_obj->buffer_pool = buffer_pool;

	// Pass EVERYTHING through setup() - the only communication channel to the plugin
	illixr_plugin_obj->sb_timewarp->setup(render_pass, subpass, buffer_pool, true,
	                                      framebuffer_array, // Plugin stores this pointer
	                                      extent             // Plugin uses this for encoder dimensions
	);

	fprintf(stderr, "[ILLIXR] Timewarp setup complete - extent=%ux%u, fb_array=%p\n", extent.width, extent.height,
	        (void *)framebuffer_array);	
}

extern "C" int8_t
illixr_src_acquire()
{
	assert(illixr_plugin_obj && "illixr_plugin_obj must be initialized first.");
	return illixr_plugin_obj->buffer_pool->src_acquire_image();
}

extern "C" void
illixr_src_release(int8_t buffer_ind, struct xrt_pose l_pose, struct xrt_pose r_pose)
{
	assert(illixr_plugin_obj && "illixr_plugin_obj must be initialized first.");

	auto now = illixr_plugin_obj->sb_clock->now();
#ifdef USING_OPENXR
	//std::array<xrt_pose, 2> input_poses{l_pose, r_pose};
	/* std::array<XrPosef, 2> poses;
	for (int i = 0; i < 2; i++) {
		poses[i].orientation.w = input_poses[i].orientation.w;
		poses[i].orientation.x = input_poses[i].orientation.x;
		poses[i].orientation.y = input_poses[i].orientation.y;
		poses[i].orientation.z = input_poses[i].orientation.z;
		poses[i].position.x = input_poses[i].position.x;
		poses[i].position.y = input_poses[i].position.y;
		poses[i].position.z = input_poses[i].position.z;
	}*/

	illixr_plugin_obj->buffer_pool->src_release_image(buffer_ind, {l_pose, r_pose});
#else
	pose::head_pose_type pose{now,
				  Eigen::Vector3f{(l_pose.position.x + r_pose.position.x) / 2.f,
						  (l_pose.position.y + r_pose.position.y) / 2.f,
						  (l_pose.position.z + r_pose.position.z) / 2.f},
				  Eigen::Quaternionf{l_pose.orientation.w, l_pose.orientation.x,
						     l_pose.orientation.y, l_pose.orientation.z}};
	
	illixr_plugin_obj->buffer_pool->src_release_image(buffer_ind, pose::fast_head_pose_type{pose, {}, {}});
#endif
}

extern "C" bool
illixr_offload_frames()
{
	assert(illixr_plugin_obj && "illixr_plugin_obj must be initialized first.");
	return illixr_plugin_obj->offload_frames;
}

extern "C" int
illixr_sleep_time()
{
	assert(illixr_plugin_obj && "illixr_plugin_obj must be initialized first.");
	return illixr_plugin_obj->sleep_time;
}

extern "C" void
illixr_tw_update_uniforms(xrt_pose l_pose, xrt_pose r_pose)
{
	assert(illixr_plugin_obj && "illixr_plugin_obj must be initialized first.");

	if (!illixr_plugin_obj->offload_frames) {
#ifdef USING_OPENXR
		std::array<xrt_pose, 2> input_poses{l_pose, r_pose};
		std::array<xrt_pose, 2> pose;
		for (int i = 0; i < 2; i++) {
			pose[i].orientation.w = input_poses[i].orientation.w;
			pose[i].orientation.x = input_poses[i].orientation.x;
			pose[i].orientation.y = input_poses[i].orientation.y;
			pose[i].orientation.z = input_poses[i].orientation.z;
			pose[i].position.x = input_poses[i].position.x;
			pose[i].position.y = input_poses[i].position.y;
			pose[i].position.z = input_poses[i].position.z;
		}

#else
		pose::head_pose_type pose{time_point{},
					  Eigen::Vector3f{(l_pose.position.x + r_pose.position.x) / 2.f,
							  (l_pose.position.y + r_pose.position.y) / 2.f,
							  (l_pose.position.z + r_pose.position.z) / 2.f},
					  Eigen::Quaternionf{l_pose.orientation.w, l_pose.orientation.x,
							     l_pose.orientation.y, l_pose.orientation.z}};
#endif
		illixr_plugin_obj->last_pose = pose;
	}
}

extern "C" void
illixr_tw_record_command_buffer(VkCommandBuffer commandBuffer, VkFramebuffer framebuffer, int buffer_ind, int left)
{
	assert(illixr_plugin_obj && "illixr_plugin_obj must be initialized first.");

	if (!illixr_plugin_obj->offload_frames) {
		illixr_plugin_obj->sb_timewarp->update_uniforms(illixr_plugin_obj->last_pose);
		illixr_plugin_obj->sb_timewarp->record_command_buffer(commandBuffer, framebuffer, buffer_ind, left);
	}
}

extern "C" void
illixr_publish_vsync_estimate(uint64_t display_time_ns)
{
	assert(illixr_plugin_obj && "illixr_plugin_obj must be initialized first.");

	if (!illixr_plugin_obj->offload_frames) {
		auto relative_time = time_point{time_point{std::chrono::nanoseconds(display_time_ns)} -
		                                illixr_plugin_obj->sb_clock->start_time()};
		illixr_plugin_obj->_m_vsync.put(illixr_plugin_obj->_m_vsync.allocate(relative_time));
	}
}
