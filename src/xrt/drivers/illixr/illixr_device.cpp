// Copyright 2020-2021, The Board of Trustees of the University of Illinois.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  ILLIXR HMD device and XR_EXT_hand_interaction device
 * @author RSIM Group <illixr@cs.illinois.edu>
 * @ingroup drv_illixr
 */

#include <assert.h>
#include <chrono>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <string>
#include <string.h>

// Platform-specific includes
#ifdef _WIN32
#include <malloc.h> // for alloca on Windows
#include <io.h>     // Windows I/O functions
                    // Windows doesn't have unistd.h or dlfcn.h
                    // Dynamic library loading is handled by ILLIXR::dynamic_lib
#else
#include <alloca.h>
#include <dlfcn.h>
#include <unistd.h>
#endif

#include "math/m_api.h"
#include "util/u_debug.h"
#include "util/u_device.h"
#include "util/u_distortion_mesh.h"
#include "util/u_misc.h"
#include "util/u_time.h"
#include "util/u_var.h"
#include "xrt/xrt_device.h"

// Include os_time.h for timestamp utilities if available
// This provides os_monotonic_get_ns() on supported platforms
#ifdef XRT_HAVE_TIMESPEC
#include "os/os_time.h"
#endif

#include "illixr/dynamic_lib.hpp"
#include "illixr/global_module_defs.hpp"
#include "illixr/runtime.hpp"

#include "illixr_component.h"

/*
 *
 * Structs and defines.
 *
 */

// Debug file logger - static global
/*
static FILE *g_debug_log = NULL;

static void
init_debug_log(void)
{
	if (g_debug_log == NULL) {
		g_debug_log = fopen("D:\\illixr_device_debug.log", "w");
		if (g_debug_log) {
			fprintf(g_debug_log, "=== Hand Tracking Debug Log Started ===\n");
			fflush(g_debug_log);
		}
	}
}

static void
ht_log(const char *format, ...)
{
	if (g_debug_log == NULL) {
		init_debug_log();
	}

	if (g_debug_log) {
		// Get timestamp
		time_t now = time(NULL);
		struct tm *tm_info = localtime(&now);
		char time_buf[64];
		strftime(time_buf, sizeof(time_buf), "%H:%M:%S", tm_info);

		// Write timestamp
		fprintf(g_debug_log, "[%s] ", time_buf);

		// Write actual message
		va_list args;
		va_start(args, format);
		vfprintf(g_debug_log, format, args);
		va_end(args);

		fprintf(g_debug_log, "\n");
		fflush(g_debug_log); // CRITICAL - force write immediately
	}
}*/

struct illixr_hmd
{
	struct xrt_device base;

	struct xrt_pose pose;

	bool print_spew;
	bool print_debug;

	const char *path;
	const char *comp;
	ILLIXR::dynamic_lib *runtime_lib;
	ILLIXR::runtime *runtime;

	// Hand-joint tracking support
	bool hand_tracking_supported;
};

/*
 *
 * Helper Functions
 *
 */

/**
 * @brief Get current monotonic time in nanoseconds (portable implementation)
 */
static uint64_t
get_timestamp_ns(void)
{
#if defined(XRT_HAVE_TIMESPEC) && !defined(_WIN32)
	// Use Monado's utility if available on non-Windows
	return os_monotonic_get_ns();
#else
	// Portable fallback using C++ chrono
	auto now = std::chrono::steady_clock::now();
	auto ns  = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch());
	return static_cast<uint64_t>(ns.count());
#endif
}

/*
 *
 * Functions
 *
 */

static inline struct illixr_hmd *
illixr_hmd(struct xrt_device *xdev)
{
	return (struct illixr_hmd *)xdev;
}

DEBUG_GET_ONCE_BOOL_OPTION(illixr_spew, "ILLIXR_PRINT_SPEW", false)
DEBUG_GET_ONCE_BOOL_OPTION(illixr_debug, "ILLIXR_PRINT_DEBUG", false)

#define DH_SPEW(dh, ...)                                                                                               \
	do {                                                                                                           \
		if (dh->print_spew) {                                                                                  \
			fprintf(stderr, "%s - ", __func__);                                                            \
			fprintf(stderr, __VA_ARGS__);                                                                  \
			fprintf(stderr, "\n");                                                                         \
		}                                                                                                      \
	} while (false)

#define DH_DEBUG(dh, ...)                                                                                              \
	do {                                                                                                           \
		if (dh->print_debug) {                                                                                 \
			fprintf(stderr, "%s - ", __func__);                                                            \
			fprintf(stderr, __VA_ARGS__);                                                                  \
			fprintf(stderr, "\n");                                                                         \
		}                                                                                                      \
	} while (false)

#define DH_ERROR(dh, ...)                                                                                              \
	do {                                                                                                           \
		fprintf(stderr, "%s - ", __func__);                                                                    \
		fprintf(stderr, __VA_ARGS__);                                                                          \
		fprintf(stderr, "\n");                                                                                 \
	} while (false)

static void
illixr_hmd_destroy(struct xrt_device *xdev)
{
	struct illixr_hmd *dh = illixr_hmd(xdev);
	dh->runtime->stop();
	delete dh->runtime;
	delete dh->runtime_lib;

	// Remove the variable tracking.
	u_var_remove_root(dh);

	u_device_free(&dh->base);
}

static xrt_result_t
illixr_hmd_update_inputs(struct xrt_device *xdev)
{
	return XRT_SUCCESS;
}

static xrt_result_t
illixr_hmd_get_tracked_pose(struct xrt_device *xdev,
                            enum xrt_input_name name,
                            int64_t at_timestamp_ns,
                            struct xrt_space_relation *out_relation)
{
	(void)at_timestamp_ns;
	if (name != XRT_INPUT_GENERIC_HEAD_POSE) {
		DH_ERROR(illixr_hmd(xdev), "unknown input name");
		return XRT_ERROR_INPUT_UNSUPPORTED;
	}

	// illixr_read_head_relation populates pose, linear/angular velocity,
	// and all relation flags (including velocity valid bits) from head_pose_type.
	*out_relation = illixr_read_head_relation();
	return XRT_SUCCESS;
}


/**
 * @brief Convert illixr_hand_joint to xrt_hand_joint_value
 */
static void
convert_illixr_joint_to_xrt(const struct illixr_hand_joint *src, struct xrt_hand_joint_value *dst)
{
	// Position and orientation
	dst->relation.pose.position.x = src->position.x;
	dst->relation.pose.position.y = src->position.y;
	dst->relation.pose.position.z = src->position.z;

	dst->relation.pose.orientation.x = src->orientation.x;
	dst->relation.pose.orientation.y = src->orientation.y;
	dst->relation.pose.orientation.z = src->orientation.z;
	dst->relation.pose.orientation.w = src->orientation.w;

	// Radius
	dst->radius = src->radius;

	// Build relation flags from ILLIXR location_flags, whose bit layout now
	// matches Monado's xrt_space_relation_flags directly:
	//   Bit 0 (0x01): Orientation valid
	//   Bit 1 (0x02): Position valid
	//   Bit 2 (0x04): Linear velocity valid
	//   Bit 3 (0x08): Angular velocity valid
	//   Bit 4 (0x10): Orientation tracked (live sensor data, not extrapolated)
	//   Bit 5 (0x20): Position tracked (live sensor data, not extrapolated)
	enum xrt_space_relation_flags flags = (enum xrt_space_relation_flags)0;

	if (src->location_flags & 0x01) { // Orientation valid
		flags = (enum xrt_space_relation_flags)(flags | XRT_SPACE_RELATION_ORIENTATION_VALID_BIT);
	}
	if (src->location_flags & 0x02) { // Position valid
		flags = (enum xrt_space_relation_flags)(flags | XRT_SPACE_RELATION_POSITION_VALID_BIT);
	}
	if (src->location_flags & 0x10) { // Orientation tracked
		flags = (enum xrt_space_relation_flags)(flags | XRT_SPACE_RELATION_ORIENTATION_TRACKED_BIT);
	}
	if (src->location_flags & 0x20) { // Position tracked
		flags = (enum xrt_space_relation_flags)(flags | XRT_SPACE_RELATION_POSITION_TRACKED_BIT);
	}

	// If no pose flags were set but we have data, assume valid and tracked
	if (flags == 0) {
		flags = (enum xrt_space_relation_flags)(
		    XRT_SPACE_RELATION_POSITION_VALID_BIT | XRT_SPACE_RELATION_ORIENTATION_VALID_BIT |
		    XRT_SPACE_RELATION_POSITION_TRACKED_BIT | XRT_SPACE_RELATION_ORIENTATION_TRACKED_BIT);
	}

	// Velocities
	dst->relation.linear_velocity.x = src->linear_velocity.x;
	dst->relation.linear_velocity.y = src->linear_velocity.y;
	dst->relation.linear_velocity.z = src->linear_velocity.z;

	dst->relation.angular_velocity.x = src->angular_velocity.x;
	dst->relation.angular_velocity.y = src->angular_velocity.y;
	dst->relation.angular_velocity.z = src->angular_velocity.z;

	// Set velocity validity flags from location_flags bits 2 and 3
	if (src->location_flags & 0x04) { // Linear velocity valid
		flags = (enum xrt_space_relation_flags)(flags | XRT_SPACE_RELATION_LINEAR_VELOCITY_VALID_BIT);
	}
	if (src->location_flags & 0x08) { // Angular velocity valid
		flags = (enum xrt_space_relation_flags)(flags | XRT_SPACE_RELATION_ANGULAR_VELOCITY_VALID_BIT);
	}

	dst->relation.relation_flags = flags;
}

/**
 * @brief Get hand-joint tracking data from ILLIXR and populate an xrt_hand_joint_set.
 *
 * Converts all 26 joint poses (position, orientation, radius, linear velocity,
 * angular velocity) and their validity flags from the ILLIXR representation
 * into Monado's xrt_hand_joint_set.  The OpenXR runtime then exposes this data
 * to applications via XR_EXT_hand_tracking.
 */
static void
illixr_hmd_get_hand_tracking(struct xrt_device *xdev,
                             enum xrt_input_name name,
                             int64_t desired_timestamp_ns,
                             struct xrt_hand_joint_set *out_value,
                             int64_t *out_timestamp_ns)
{
	(void)xdev;
	(void)desired_timestamp_ns;

	//ht_log("Hand tracking called for input %d", name);

	// Check if hand tracking is supported
	if (!illixr_hand_tracking_supported()) {
		//ht_log("Hand tracking not supported");
		out_value->is_active = false;
		return;
	}

	// Determine which hand
	int hand_index = -1;
	if (name == XRT_INPUT_GENERIC_HAND_TRACKING_LEFT) {
		hand_index = 0;
	} else if (name == XRT_INPUT_GENERIC_HAND_TRACKING_RIGHT) {
		hand_index = 1;
	} else {
		//ht_log("Unknown hand tracking input name: %d", name);
		out_value->is_active = false;
		return;
	}

	// Fetch hand data from ILLIXR switchboard
	struct illixr_single_hand hand_data;
	if (!illixr_read_single_hand(hand_index, &hand_data)) {
		//ht_log("illixr_read_single_hand returned false for hand %d", hand_index);
		out_value->is_active = false;
		return;
	}

	// Set active state
	out_value->is_active = hand_data.is_active;

	if (!hand_data.is_active) {
		//ht_log("Hand %d not active", hand_index);
		return;
	}

	// Convert all joints: position, orientation, radius, linear velocity,
	// angular velocity, and validity flags
	for (int i = 0; i < XRT_HAND_JOINT_COUNT && i < ILLIXR_HAND_JOINT_COUNT; i++) {
		convert_illixr_joint_to_xrt(&hand_data.joints[i],
		                            &out_value->values.hand_joint_set_default[i]);
	}
	// Return current time
	*out_timestamp_ns = (int64_t)get_timestamp_ns();

	//ht_log("Hand %d tracking data returned: wrist=(%.4f, %.4f, %.4f) flags=0x%x", hand_index,
	//       hand_data.joints[1].position.x, hand_data.joints[1].position.y, hand_data.joints[1].position.z,
	//       hand_data.joints[1].location_flags);
}

static void
illixr_hmd_get_view_poses(struct xrt_device *xdev,
                          const struct xrt_vec3 *default_eye_relation,
                          int64_t at_timestamp_ns,
                          uint32_t view_count,
                          struct xrt_space_relation *out_head_relation,
                          struct xrt_fov *out_fovs,
                          struct xrt_pose *out_poses)
{
	u_device_get_view_poses(xdev, default_eye_relation, at_timestamp_ns, view_count, out_head_relation, out_fovs,
	                        out_poses);
}

static std::vector<std::string>
split(const std::string &s, char delimiter)
{
	std::vector<std::string> tokens;
	std::string token;
	std::istringstream tokenStream{s};
	while (std::getline(tokenStream, token, delimiter)) {
		tokens.push_back(token);
	}
	return tokens;
}

static uint32_t
get_server_width()
{
	if (std::getenv("ILLIXR_SERVER_WIDTH") == nullptr) {
		printf("[Monado] Display width not specified, defaulting to %d pixels.\n",
		       ILLIXR::display_params::width_pixels);
		return ILLIXR::display_params::width_pixels;
	}
	return std::stoi(std::getenv("ILLIXR_SERVER_WIDTH"));
}

static uint32_t
get_server_height()
{
	if (std::getenv("ILLIXR_SERVER_HEIGHT") == nullptr) {
		printf("[Monado] Display height not specified, defaulting to %d pixels.\n",
		       ILLIXR::display_params::height_pixels);
		return ILLIXR::display_params::height_pixels;
	}
	return std::stoi(std::getenv("ILLIXR_SERVER_HEIGHT"));
}

static int
illixr_rt_launch(struct illixr_hmd *dh, const char *path, const char *comp)
{
	dh->runtime_lib = new ILLIXR::dynamic_lib{ILLIXR::dynamic_lib::create(std::string{path})};
	dh->runtime = dh->runtime_lib->get<ILLIXR::runtime *(*)()>("runtime_factory")();
#if defined(_WIN32) || defined(_WIN64)
	dh->runtime->load_so(split(std::string{comp}, ';'));
#else
	dh->runtime->load_so(split(std::string{comp}, ':'));
#endif
	dh->runtime->load_plugin_factory((ILLIXR::plugin_factory)illixr_monado_create_plugin);

	return 0;
}

extern "C" struct xrt_device *
illixr_hmd_create(const char *path_in, const char *comp_in)
{
	struct illixr_hmd *dh;
	enum u_device_alloc_flags flags =
	    (enum u_device_alloc_flags)(U_DEVICE_ALLOC_HMD | U_DEVICE_ALLOC_TRACKING_NONE);
	dh = U_DEVICE_ALLOCATE(struct illixr_hmd, flags, 3, 0);

	// Standard xrt_device callbacks
	dh->base.update_inputs = illixr_hmd_update_inputs;
	dh->base.get_tracked_pose = illixr_hmd_get_tracked_pose;
	dh->base.get_view_poses = illixr_hmd_get_view_poses;
	dh->base.get_hand_tracking = illixr_hmd_get_hand_tracking;
	dh->base.destroy = illixr_hmd_destroy;
	dh->base.name = XRT_DEVICE_GENERIC_HMD;
	dh->base.device_type = XRT_DEVICE_TYPE_HMD;
	dh->base.orientation_tracking_supported = true;
	dh->base.position_tracking_supported = true;
	dh->base.hand_tracking_supported = true;

	// Read framerate from environment variable
	if (std::getenv("ILLIXR_OFFLOAD_RENDERING_FRAMERATE") != nullptr) {
		dh->base.hmd->screens[0].nominal_frame_interval_ns =
		    1000000000 / std::stoi(std::getenv("ILLIXR_OFFLOAD_RENDERING_FRAMERATE"));
	} else {
		dh->base.hmd->screens[0].nominal_frame_interval_ns = 1000000000 / 90;
	}

	size_t idx = 0;
	dh->base.hmd->blend_modes[idx++] = XRT_BLEND_MODE_OPAQUE;
	dh->base.hmd->blend_mode_count = idx;

	dh->pose.orientation.w = 1.0f; // All other values set to zero.
	dh->print_spew = debug_get_bool_option_illixr_spew();
	dh->print_debug = debug_get_bool_option_illixr_debug();
	dh->path = path_in;
	dh->comp = comp_in;

	// Print name.
	snprintf(dh->base.str, XRT_DEVICE_NAME_LEN, "ILLIXR");
	snprintf(dh->base.serial, XRT_DEVICE_NAME_LEN, "ILLIXR");

	// Setup inputs: head pose + hand tracking
	dh->base.inputs[0].name = XRT_INPUT_GENERIC_HEAD_POSE;
	dh->base.inputs[1].name = XRT_INPUT_GENERIC_HAND_TRACKING_LEFT;
	dh->base.inputs[2].name = XRT_INPUT_GENERIC_HAND_TRACKING_RIGHT;

	// Setup info.
	struct u_device_simple_info info;
	info.display.w_pixels = get_server_width();
	info.display.h_pixels = get_server_height();
	info.display.w_meters = 0.122f;
	info.display.h_meters = 0.07f;
	info.lens_horizontal_separation_meters = 0.13f / 2.0f;
	info.lens_vertical_position_meters = 0.07f / 2.0f;
	info.fov[0] = 85.0f * (M_PI / 180.0f);
	info.fov[1] = 85.0f * (M_PI / 180.0f);

	if (!u_device_setup_split_side_by_side(&dh->base, &info)) {
		DH_ERROR(dh, "Failed to setup basic device info");
		illixr_hmd_destroy(&dh->base);
		return NULL;
	}

	// Read ILLIXR_OVERSCAN from environment variable
	float scale = 1.0f;
	if (std::getenv("ILLIXR_OVERSCAN") != nullptr) {
		scale = std::stof(std::getenv("ILLIXR_OVERSCAN"));
	}

	// The server may render at a different FOV than the client.
	for (int eye = 0; eye < 2; eye++) {
		float fov_left = scale * ILLIXR::server_params::fov_left[eye];
		float fov_right = scale * ILLIXR::server_params::fov_right[eye];
		float fov_up = scale * ILLIXR::server_params::fov_up[eye];
		float fov_down = scale * ILLIXR::server_params::fov_down[eye];

		dh->base.hmd->distortion.fov[eye].angle_left = fov_left;
		dh->base.hmd->distortion.fov[eye].angle_right = fov_right;
		dh->base.hmd->distortion.fov[eye].angle_up = fov_up;
		dh->base.hmd->distortion.fov[eye].angle_down = fov_down;
	}

	// Setup variable tracker.
	u_var_add_root(dh, "ILLIXR", true);
	u_var_add_pose(dh, &dh->pose, "pose");
	u_var_add_bool(dh, &dh->hand_tracking_supported, "hand_tracking_supported");

	if (dh->base.hmd->distortion.preferred == XRT_DISTORTION_MODEL_NONE) {
		// Setup the distortion mesh.
		u_distortion_mesh_set_none(&dh->base);
	}

	// start ILLIXR runtime
	if (illixr_rt_launch(dh, dh->path, dh->comp) != 0) {
		DH_ERROR(dh, "Failed to load ILLIXR Runtime");
		illixr_hmd_destroy(&dh->base);
		return NULL;
	}

	// DON'T wait here - Vulkan isn't initialized yet!
	// Instead, hand tracking will be checked at runtime in the callback

	// Check environment variables for INTENT to use hand tracking
	bool ht_enabled = false;

	const char* ht_env = std::getenv("ILLIXR_USE_HAND_TRACKING");
	if (ht_env != nullptr) {
		std::string val(ht_env);
		ht_enabled = (val == "1" || val == "true" || val == "TRUE" || val == "yes" || val == "YES");
	} else {
		// Fall back to ILLIXR_OFFLOAD_FRAMES
		const char* offload_env = std::getenv("ILLIXR_OFFLOAD_FRAMES");
		if (offload_env != nullptr) {
			ht_enabled = (std::stoi(offload_env) != 0);
		}
	}

	dh->hand_tracking_supported = ht_enabled;
	dh->base.hand_tracking_supported = ht_enabled;

	printf("[ILLIXR] ==========================================\n");
	printf("[ILLIXR] HMD device created successfully\n");
	printf("[ILLIXR]   ILLIXR_USE_HAND_TRACKING=%s\n",
	       std::getenv("ILLIXR_USE_HAND_TRACKING") ? std::getenv("ILLIXR_USE_HAND_TRACKING") : "(not set)");
	printf("[ILLIXR]   ILLIXR_OFFLOAD_FRAMES=%s\n",
	       std::getenv("ILLIXR_OFFLOAD_FRAMES") ? std::getenv("ILLIXR_OFFLOAD_FRAMES") : "(not set)");
	printf("[ILLIXR]   Hand tracking:          %s\n",    dh->hand_tracking_supported ? "ENABLED" : "DISABLED");
	printf("[ILLIXR]   get_hand_tracking:       %p\n",   (void *)dh->base.get_hand_tracking);
	//printf("[ILLIXR]   get_palm_poses:          %p\n",   (void *)dh->get_palm_poses);
	//printf("[ILLIXR]   get_hand_interactions:   %p\n",   (void *)dh->get_hand_interactions);
	printf("[ILLIXR]   get_tracked_pose:        %p\n",   (void *)dh->base.get_tracked_pose);
	printf("[ILLIXR]   update_inputs:           %p\n",   (void *)dh->base.update_inputs);
	printf("[ILLIXR]   destroy:                 %p\n",   (void *)dh->base.destroy);
	printf("[ILLIXR] ==========================================\n");
	printf("[ILLIXR] Returning device at address: %p\n", (void *)&dh->base);

	return &dh->base;
}

/*
 * =========================================================================
 * XR_EXT_hand_interaction device
 *
 * One instance is created per hand (left = 0, right = 1).  Both instances
 * share the same xrt_input_name values; Monado resolves which device to use
 * for a given subaction path (/user/hand/left vs /user/hand/right) by the
 * device_type (LEFT_HAND_CONTROLLER vs RIGHT_HAND_CONTROLLER) assigned at
 * creation time.
 *
 * Inputs served:
 *   0  XRT_INPUT_GENERIC_PALM_POSE          - XR_EXT_palm_pose
 *   1  XRT_INPUT_HAND_AIM_POSE              - /input/aim/pose
 *   2  XRT_INPUT_HAND_GRIP_POSE             - /input/grip/pose
 *   3  XRT_INPUT_HAND_PINCH_POSE            - /input/pinch_ext/pose
 *   4  XRT_INPUT_HAND_POKE_POSE             - /input/poke_ext/pose
 *   5  XRT_INPUT_HAND_AIM_ACTIVATE_VALUE    - /input/aim_activate_ext/value
 *   6  XRT_INPUT_HAND_GRASP_VALUE           - /input/grasp_ext/value
 *   7  XRT_INPUT_HAND_PINCH_VALUE           - /input/pinch_ext/value
 *   8  XRT_INPUT_HAND_AIM_ACTIVATE_READY    - /input/aim_activate_ext/ready_ext
 *   9  XRT_INPUT_HAND_GRASP_READY           - /input/grasp_ext/ready_ext
 *   10 XRT_INPUT_HAND_PINCH_READY           - /input/pinch_ext/ready_ext
 * =========================================================================
 */

// clang-format off
enum illixr_hand_input_index {
	ILLIXR_HAND_INPUT_PALM_POSE          = 0,
	ILLIXR_HAND_INPUT_AIM_POSE           = 1,
	ILLIXR_HAND_INPUT_GRIP_POSE          = 2,
	ILLIXR_HAND_INPUT_PINCH_POSE         = 3,
	ILLIXR_HAND_INPUT_POKE_POSE          = 4,
	ILLIXR_HAND_INPUT_AIM_ACTIVATE_VALUE = 5,
	ILLIXR_HAND_INPUT_GRASP_VALUE        = 6,
	ILLIXR_HAND_INPUT_PINCH_VALUE        = 7,
	ILLIXR_HAND_INPUT_AIM_ACTIVATE_READY = 8,
	ILLIXR_HAND_INPUT_GRASP_READY        = 9,
	ILLIXR_HAND_INPUT_PINCH_READY        = 10,
	ILLIXR_HAND_INPUT_COUNT              = 11,
};
// clang-format on

struct illixr_hand_device
{
	struct xrt_device base;
	int hand;        //!< 0 = left, 1 = right
	bool print_spew;
	bool print_debug;
};

static inline struct illixr_hand_device *
illixr_hand_device(struct xrt_device *xdev)
{
	return (struct illixr_hand_device *)xdev;
}

#define HD_ERROR(hd, ...)                                                                                              \
	do {                                                                                                           \
		fprintf(stderr, "%s [hand=%d] - ", __func__, hd->hand);                                                \
		fprintf(stderr, __VA_ARGS__);                                                                          \
		fprintf(stderr, "\n");                                                                                 \
	} while (false)

static void
illixr_hand_device_destroy(struct xrt_device *xdev)
{
	// The ILLIXR runtime is owned by the HMD device; do not stop it here.
	u_var_remove_root(xdev);
	u_device_free(xdev);
}

/**
 * @brief Write float and boolean scalar inputs from the switchboard.
 *
 * Called by Monado once per xrSyncActions before it dispatches
 * xrGetActionStateFloat / xrGetActionStateBoolean to applications.
 * Pose inputs are NOT written here; they are fetched on demand in
 * get_tracked_pose because they require a per-call timestamp.
 */
static xrt_result_t
illixr_hand_device_update_inputs(struct xrt_device *xdev)
{
	struct illixr_hand_device *hd = illixr_hand_device(xdev);
	int64_t now = (int64_t)get_timestamp_ns();

	struct illixr_hand_interaction_data data = {};
	const bool have = illixr_read_hand_interaction(hd->hand, &data);

#define SET_FLOAT(idx, v)                                        \
	hd->base.inputs[idx].active       = have;               \
	hd->base.inputs[idx].timestamp    = now;                \
	hd->base.inputs[idx].value.vec1.x = (v)

#define SET_BOOL(idx, v)                                         \
	hd->base.inputs[idx].active        = have;              \
	hd->base.inputs[idx].timestamp     = now;               \
	hd->base.inputs[idx].value.boolean = (v)

	SET_FLOAT(ILLIXR_HAND_INPUT_AIM_ACTIVATE_VALUE, data.poses[ILLIXR_INTERACTION_AIM].value);
	SET_FLOAT(ILLIXR_HAND_INPUT_GRASP_VALUE,        data.poses[ILLIXR_INTERACTION_GRIP].value);
	SET_FLOAT(ILLIXR_HAND_INPUT_PINCH_VALUE,        data.poses[ILLIXR_INTERACTION_PINCH].value);

	SET_BOOL(ILLIXR_HAND_INPUT_AIM_ACTIVATE_READY, data.poses[ILLIXR_INTERACTION_AIM].ready);
	SET_BOOL(ILLIXR_HAND_INPUT_GRASP_READY,        data.poses[ILLIXR_INTERACTION_GRIP].ready);
	SET_BOOL(ILLIXR_HAND_INPUT_PINCH_READY,        data.poses[ILLIXR_INTERACTION_PINCH].ready);

#undef SET_FLOAT
#undef SET_BOOL

	return XRT_SUCCESS;
}

/**
 * @brief Return the requested pose for this hand.
 *
 * An invalid or unavailable pose is signalled by setting relation_flags to 0,
 * which causes xrLocateSpace to return the position and orientation valid bits
 * clear - the correct spec-compliant response for an unlocatable space.
 */
static xrt_result_t
illixr_hand_device_get_tracked_pose(struct xrt_device *xdev,
                                    enum xrt_input_name name,
                                    int64_t at_timestamp_ns,
                                    struct xrt_space_relation *out_relation)
{
	(void)at_timestamp_ns;

	struct illixr_hand_device *hd = illixr_hand_device(xdev);

	const enum xrt_space_relation_flags full_flags = (enum xrt_space_relation_flags)(
	    XRT_SPACE_RELATION_ORIENTATION_VALID_BIT | XRT_SPACE_RELATION_ORIENTATION_TRACKED_BIT |
	    XRT_SPACE_RELATION_POSITION_VALID_BIT    | XRT_SPACE_RELATION_POSITION_TRACKED_BIT);

	// ---- Palm pose -------------------------------------------------------
	if (name == XRT_INPUT_GENERIC_PALM_POSE) {
		struct illixr_palm_pose palm = {};
		if (!illixr_read_palm_pose(hd->hand, &palm) || !palm.valid) {
			out_relation->relation_flags = (enum xrt_space_relation_flags)0;
			return XRT_SUCCESS;
		}
		out_relation->pose.position.x    = palm.position.x;
		out_relation->pose.position.y    = palm.position.y;
		out_relation->pose.position.z    = palm.position.z;
		out_relation->pose.orientation.x = palm.orientation.x;
		out_relation->pose.orientation.y = palm.orientation.y;
		out_relation->pose.orientation.z = palm.orientation.z;
		out_relation->pose.orientation.w = palm.orientation.w;
		out_relation->relation_flags     = full_flags;
		return XRT_SUCCESS;
	}

	// ---- Hand interaction poses ------------------------------------------
	int pose_slot = -1;
	switch (name) {
	case XRT_INPUT_HAND_AIM_POSE:   pose_slot = ILLIXR_INTERACTION_AIM;   break;
	case XRT_INPUT_HAND_GRIP_POSE:  pose_slot = ILLIXR_INTERACTION_GRIP;  break;
	case XRT_INPUT_HAND_PINCH_POSE: pose_slot = ILLIXR_INTERACTION_PINCH; break;
	case XRT_INPUT_HAND_POKE_POSE:  pose_slot = ILLIXR_INTERACTION_POKE;  break;
	default:
		HD_ERROR(hd, "unknown input name %d", (int)name);
		return XRT_ERROR_INPUT_UNSUPPORTED;
	}

	struct illixr_hand_interaction_data data = {};
	if (!illixr_read_hand_interaction(hd->hand, &data)) {
		out_relation->relation_flags = (enum xrt_space_relation_flags)0;
		return XRT_SUCCESS;
	}

	const struct illixr_interaction_pose &src = data.poses[pose_slot];
	if (!src.valid) {
		out_relation->relation_flags = (enum xrt_space_relation_flags)0;
		return XRT_SUCCESS;
	}

	out_relation->pose.position.x    = src.position.x;
	out_relation->pose.position.y    = src.position.y;
	out_relation->pose.position.z    = src.position.z;
	out_relation->pose.orientation.x = src.orientation.x;
	out_relation->pose.orientation.y = src.orientation.y;
	out_relation->pose.orientation.z = src.orientation.z;
	out_relation->pose.orientation.w = src.orientation.w;
	out_relation->relation_flags     = full_flags;
	return XRT_SUCCESS;
}

extern "C" struct xrt_device *
illixr_hand_device_create(int hand)
{
	assert(hand == 0 || hand == 1);

	struct illixr_hand_device *hd =
	    U_DEVICE_ALLOCATE(struct illixr_hand_device, U_DEVICE_ALLOC_TRACKING_NONE,
	                      ILLIXR_HAND_INPUT_COUNT, 0);

	hd->hand        = hand;
	hd->print_spew  = debug_get_bool_option_illixr_spew();
	hd->print_debug = debug_get_bool_option_illixr_debug();

	hd->base.update_inputs    = illixr_hand_device_update_inputs;
	hd->base.get_tracked_pose = illixr_hand_device_get_tracked_pose;
	hd->base.destroy          = illixr_hand_device_destroy;

	hd->base.name        = XRT_DEVICE_EXT_HAND_INTERACTION;
	hd->base.device_type = (hand == 0)
	                           ? XRT_DEVICE_TYPE_LEFT_HAND_CONTROLLER
	                           : XRT_DEVICE_TYPE_RIGHT_HAND_CONTROLLER;

	hd->base.orientation_tracking_supported = true;
	hd->base.position_tracking_supported    = true;
	hd->base.hand_tracking_supported        = false; // hand skeleton is on the HMD device

	snprintf(hd->base.str,    XRT_DEVICE_NAME_LEN, "ILLIXR Hand (%s)", hand == 0 ? "Left" : "Right");
	snprintf(hd->base.serial, XRT_DEVICE_NAME_LEN, "ILLIXR-HAND-%s",   hand == 0 ? "L"    : "R");

	// ---- Input table ----
	hd->base.inputs[ILLIXR_HAND_INPUT_PALM_POSE].name          = XRT_INPUT_GENERIC_PALM_POSE;
	hd->base.inputs[ILLIXR_HAND_INPUT_AIM_POSE].name           = XRT_INPUT_HAND_AIM_POSE;
	hd->base.inputs[ILLIXR_HAND_INPUT_GRIP_POSE].name          = XRT_INPUT_HAND_GRIP_POSE;
	hd->base.inputs[ILLIXR_HAND_INPUT_PINCH_POSE].name         = XRT_INPUT_HAND_PINCH_POSE;
	hd->base.inputs[ILLIXR_HAND_INPUT_POKE_POSE].name          = XRT_INPUT_HAND_POKE_POSE;
	hd->base.inputs[ILLIXR_HAND_INPUT_AIM_ACTIVATE_VALUE].name = XRT_INPUT_HAND_AIM_ACTIVATE_VALUE;
	hd->base.inputs[ILLIXR_HAND_INPUT_GRASP_VALUE].name        = XRT_INPUT_HAND_GRASP_VALUE;
	hd->base.inputs[ILLIXR_HAND_INPUT_PINCH_VALUE].name        = XRT_INPUT_HAND_PINCH_VALUE;
	hd->base.inputs[ILLIXR_HAND_INPUT_AIM_ACTIVATE_READY].name = XRT_INPUT_HAND_AIM_ACTIVATE_READY;
	hd->base.inputs[ILLIXR_HAND_INPUT_GRASP_READY].name        = XRT_INPUT_HAND_GRASP_READY;
	hd->base.inputs[ILLIXR_HAND_INPUT_PINCH_READY].name        = XRT_INPUT_HAND_PINCH_READY;

	u_var_add_root(hd, hd->base.str, true);

	printf("[ILLIXR] Hand interaction device created: %s\n", hd->base.str);
	printf("[ILLIXR]   device_type:      %d\n", (int)hd->base.device_type);
	printf("[ILLIXR]   input count:      %d\n", ILLIXR_HAND_INPUT_COUNT);
	printf("[ILLIXR]   update_inputs:    %p\n", (void *)hd->base.update_inputs);
	printf("[ILLIXR]   get_tracked_pose: %p\n", (void *)hd->base.get_tracked_pose);

	return &hd->base;
}
