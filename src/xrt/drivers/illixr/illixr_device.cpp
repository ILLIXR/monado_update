// Copyright 2020-2026, The Board of Trustees of the University of Illinois.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  ILLIXR HMD device
 * @author RSIM Group <illixr@cs.illinois.edu>
 * @ingroup drv_illixr
 */
#include <assert.h>
#include <math.h>
#include <string>

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
#include "os/os_time.h"

#include "illixr/dynamic_lib.hpp"
#include "illixr/global_module_defs.hpp"
#include "illixr/runtime.hpp"

#include "illixr_component.h"
#include "illixr_device_common.h"

/*
 *
 * Structs and defines.
 *
 */

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
};

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
	if (name != XRT_INPUT_GENERIC_HEAD_POSE) {
		DH_ERROR(illixr_hmd(xdev), "unknown input name");
		return XRT_ERROR_INPUT_UNSUPPORTED;
	}

	// illixr_read_head_relation populates pose, linear/angular velocity,
	// and all relation flags (including velocity valid bits) from head_pose_type.
	*out_relation = illixr_read_head_relation(at_timestamp_ns);
	return XRT_SUCCESS;
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
	dh = U_DEVICE_ALLOCATE(struct illixr_hmd, flags, 1, 0);

	// Standard xrt_device callbacks
	dh->base.update_inputs = illixr_hmd_update_inputs;
	dh->base.get_tracked_pose = illixr_hmd_get_tracked_pose;
	dh->base.get_view_poses = illixr_hmd_get_view_poses;
	dh->base.destroy = illixr_hmd_destroy;
	dh->base.name = XRT_DEVICE_GENERIC_HMD;
	dh->base.device_type = XRT_DEVICE_TYPE_HMD;
	dh->base.orientation_tracking_supported = true;
	dh->base.position_tracking_supported = true;

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

		// Read ILLIXR_OVERSCAN from environment variable
	float scale = 1.0f;
	if (std::getenv("ILLIXR_OVERSCAN") != nullptr) {
		scale = std::stof(std::getenv("ILLIXR_OVERSCAN"));
	}
	 
	// Setup info.
	struct u_device_simple_info info;
	info.display.w_pixels = (uint32_t)(get_server_width() * scale);
	info.display.h_pixels = (uint32_t)(get_server_height() * scale);
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

	printf("[ILLIXR] ==========================================\n");
	printf("[ILLIXR] HMD device created successfully\n");
	printf("[ILLIXR]   ILLIXR_OFFLOAD_FRAMES=%s\n",
	       std::getenv("ILLIXR_OFFLOAD_FRAMES") ? std::getenv("ILLIXR_OFFLOAD_FRAMES") : "(not set)");
	printf("[ILLIXR]   get_tracked_pose:        %p\n",   (void *)dh->base.get_tracked_pose);
	printf("[ILLIXR]   update_inputs:           %p\n",   (void *)dh->base.update_inputs);
	printf("[ILLIXR]   destroy:                 %p\n",   (void *)dh->base.destroy);
	printf("[ILLIXR] ==========================================\n");
	printf("[ILLIXR] Returning device at address: %p\n", (void *)&dh->base);

	return &dh->base;
}
