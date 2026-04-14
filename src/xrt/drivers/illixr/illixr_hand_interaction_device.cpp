// Copyright 2020-2026, The Board of Trustees of the University of Illinois.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  XRT_DEVICE_EXT_HAND_INTERACTION device
 * @author RSIM Group <illixr@cs.illinois.edu>
 * @ingroup drv_illixr
 */
#include <assert.h>
#include <chrono>

#include "util/u_debug.h"
#include "util/u_device.h"
#include "util/u_time.h"
#include "util/u_var.h"
#include "xrt/xrt_device.h"

// Include os_time.h for timestamp utilities if available
// This provides os_monotonic_get_ns() on supported platforms
#ifdef XRT_HAVE_TIMESPEC
#include "os/os_time.h"
#endif

#include "illixr_component.h"
#include "illixr_device_common.h"


// Debug file logger - static global
#ifdef BUILD_WITH_LOGGING
static FILE *g_debug_log = NULL;

static void
init_debug_log(void)
{
	if (g_debug_log == NULL) {
		g_debug_log = fopen("D:\\illixr_device_debug2.log", "w");
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
}
#endif

/*
 *
 * Functions
 *
 */

DEBUG_GET_ONCE_BOOL_OPTION(illixr_spew, "ILLIXR_PRINT_SPEW", false)
DEBUG_GET_ONCE_BOOL_OPTION(illixr_debug, "ILLIXR_PRINT_DEBUG", false)

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
enum illixr_hand_interaction_input_index {
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

struct illixr_hand_interaction_device
{
	struct xrt_device base;
	int hand; //!< 0 = left, 1 = right
	bool print_spew;
	bool print_debug;
};

static inline struct illixr_hand_interaction_device *
illixr_hand_interaction_device(struct xrt_device *xdev)
{
	return (struct illixr_hand_interaction_device *)xdev;
}

#define HD_ERROR(hd, ...)                                                                                              \
	do {                                                                                                           \
		fprintf(stderr, "%s [hand=%d] - ", __func__, hd->hand);                                                \
		fprintf(stderr, __VA_ARGS__);                                                                          \
		fprintf(stderr, "\n");                                                                                 \
	} while (false)

static void
illixr_hand_interaction_device_destroy(struct xrt_device *xdev)
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
illixr_hand_interaction_device_update_inputs(struct xrt_device *xdev)
{
	struct illixr_hand_interaction_device *hd = illixr_hand_interaction_device(xdev);
	int64_t now = (int64_t)get_timestamp_ns();

	struct illixr_hand_interaction_data data = {};
	const bool have = illixr_read_hand_interaction(hd->hand, &data);

#define SET_FLOAT(idx, v)                                                                                              \
	hd->base.inputs[idx].active = true;                                                                            \
	hd->base.inputs[idx].timestamp = now;                                                                          \
	hd->base.inputs[idx].value.vec1.x = (v)

#define SET_BOOL(idx, v)                                                                                               \
	hd->base.inputs[idx].active = true;                                                                            \
	hd->base.inputs[idx].timestamp = now;                                                                          \
	hd->base.inputs[idx].value.boolean = (v)

	if (have) {
		SET_FLOAT(ILLIXR_HAND_INPUT_AIM_ACTIVATE_VALUE, data.poses[ILLIXR_INTERACTION_AIM].value);
		SET_FLOAT(ILLIXR_HAND_INPUT_GRASP_VALUE, data.poses[ILLIXR_INTERACTION_GRIP].value);
		SET_FLOAT(ILLIXR_HAND_INPUT_PINCH_VALUE, data.poses[ILLIXR_INTERACTION_PINCH].value);

		SET_BOOL(ILLIXR_HAND_INPUT_AIM_ACTIVATE_READY, data.poses[ILLIXR_INTERACTION_AIM].ready);
		SET_BOOL(ILLIXR_HAND_INPUT_GRASP_READY, data.poses[ILLIXR_INTERACTION_GRIP].ready);
		SET_BOOL(ILLIXR_HAND_INPUT_PINCH_READY, data.poses[ILLIXR_INTERACTION_PINCH].ready);

#ifdef BUILD_WITH_LOGGING
		ht_log(
		    "update_inputs [hand=%d]: aim_val=%.3f grip_val=%.3f pinch_val=%.3f "
		    "aim_rdy=%d grip_rdy=%d pinch_rdy=%d",
		    hd->hand, data.poses[ILLIXR_INTERACTION_AIM].value, data.poses[ILLIXR_INTERACTION_GRIP].value,
		    data.poses[ILLIXR_INTERACTION_PINCH].value, (int)data.poses[ILLIXR_INTERACTION_AIM].ready,
		    (int)data.poses[ILLIXR_INTERACTION_GRIP].ready, (int)data.poses[ILLIXR_INTERACTION_PINCH].ready);
#endif
	} else {
		// Mark all scalar inputs inactive without zeroing the last known values.
		// Monado will report isActive=false to the application until data returns.
		for (int i = ILLIXR_HAND_INPUT_AIM_ACTIVATE_VALUE; i <= ILLIXR_HAND_INPUT_PINCH_READY; i++) {
			hd->base.inputs[i].active = false;
		}
#ifdef BUILD_WITH_LOGGING
		ht_log("update_inputs [hand=%d]: no interaction data", hd->hand);
#endif
	}

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
illixr_hand_interaction_device_get_tracked_pose(struct xrt_device *xdev,
                                                enum xrt_input_name name,
                                                int64_t at_timestamp_ns,
                                                struct xrt_space_relation *out_relation)
{
	(void)at_timestamp_ns;
	struct illixr_hand_interaction_device *hd = illixr_hand_interaction_device(xdev);

	// ---- Palm pose -------------------------------------------------------
	if (name == XRT_INPUT_GENERIC_PALM_POSE) {
		//struct illixr_palm_pose palm = {};
		if (!illixr_read_palm_pose(hd->hand, out_relation)) {
			out_relation->relation_flags = (enum xrt_space_relation_flags)0;
			return XRT_SUCCESS;
		}
		return XRT_SUCCESS;
	}

	// ---- Hand interaction poses ------------------------------------------
	int pose_slot = -1;
	switch (name) {
	case XRT_INPUT_HAND_AIM_POSE:
		// This is what the hand dots in JumpSim are based on
		pose_slot = ILLIXR_INTERACTION_AIM;
		break;
	case XRT_INPUT_HAND_GRIP_POSE:
		pose_slot = ILLIXR_INTERACTION_GRIP;
		break;
	case XRT_INPUT_HAND_PINCH_POSE:
		// Pinch does not seem to flow through right
		pose_slot = ILLIXR_INTERACTION_PINCH;
		break;
	case XRT_INPUT_HAND_POKE_POSE:
		pose_slot = ILLIXR_INTERACTION_POKE;
		break;
	default:
		HD_ERROR(hd, "unknown input name %d", (int)name);
		return XRT_ERROR_INPUT_UNSUPPORTED;
	}

	struct illixr_hand_interaction_data data = {};
	if (!illixr_read_hand_interaction(hd->hand, &data)) {
		out_relation->relation_flags = (enum xrt_space_relation_flags)0;
#ifdef BUILD_WITH_LOGGING
		ht_log("No hand interaction data");
#endif
		return XRT_SUCCESS;
	}

	const struct illixr_interaction_pose &src = data.poses[pose_slot];
	if (!src.valid) {
#ifdef BUILD_WITH_LOGGING
		ht_log("Hand interaction data not valid");
#endif
		out_relation->relation_flags = (enum xrt_space_relation_flags)0;
		return XRT_SUCCESS;
	}
	*out_relation = src.relation;
#ifdef BUILD_WITH_LOGGING
	ht_log("Hand interaction tracking data returned: (%.4f, %.4f, %.4f) flags=0x%x", src.relation.pose.position.x,
	       src.relation.pose.position.y, src.relation.pose.position.z, src.relation.relation_flags);
#endif
	return XRT_SUCCESS;
}



extern "C" struct xrt_device *
illixr_hand_interaction_device_create(int hand, struct xrt_tracking_origin *origin)
{
	assert(hand == 0 || hand == 1);

	struct illixr_hand_interaction_device *hd = U_DEVICE_ALLOCATE(
	    struct illixr_hand_interaction_device, U_DEVICE_ALLOC_TRACKING_NONE, ILLIXR_HAND_INPUT_COUNT, 0);

	hd->hand = hand;
	hd->print_spew = debug_get_bool_option_illixr_spew();
	hd->print_debug = debug_get_bool_option_illixr_debug();

	hd->base.update_inputs = illixr_hand_interaction_device_update_inputs;
	hd->base.get_tracked_pose = illixr_hand_interaction_device_get_tracked_pose;
	hd->base.destroy = illixr_hand_interaction_device_destroy;

	hd->base.name = XRT_DEVICE_EXT_HAND_INTERACTION;
	hd->base.device_type =
	    (hand == 0) ? XRT_DEVICE_TYPE_LEFT_HAND_CONTROLLER : XRT_DEVICE_TYPE_RIGHT_HAND_CONTROLLER;

	// Share the HMD's tracking origin so Monado applies the same
	// LOCAL-space transform to interaction poses as it does to the head.
	// Without this, Monado uses a garbage or identity origin and the
	// world-space aim pose gets a wrong additional offset applied.
	hd->base.tracking_origin = origin;

	hd->base.orientation_tracking_supported = true;
	hd->base.position_tracking_supported = true;

	snprintf(hd->base.str, XRT_DEVICE_NAME_LEN, "ILLIXR Hand Interaction (%s)", hand == 0 ? "Left" : "Right");
	snprintf(hd->base.serial, XRT_DEVICE_NAME_LEN, "ILLIXR-HAND-INTERACTION-%s", hand == 0 ? "L" : "R");

	// ---- Input table ----
	hd->base.inputs[ILLIXR_HAND_INPUT_PALM_POSE].name = XRT_INPUT_GENERIC_PALM_POSE;
	hd->base.inputs[ILLIXR_HAND_INPUT_AIM_POSE].name = XRT_INPUT_HAND_AIM_POSE;
	hd->base.inputs[ILLIXR_HAND_INPUT_GRIP_POSE].name = XRT_INPUT_HAND_GRIP_POSE;
	hd->base.inputs[ILLIXR_HAND_INPUT_PINCH_POSE].name = XRT_INPUT_HAND_PINCH_POSE;
	hd->base.inputs[ILLIXR_HAND_INPUT_POKE_POSE].name = XRT_INPUT_HAND_POKE_POSE;
	hd->base.inputs[ILLIXR_HAND_INPUT_AIM_ACTIVATE_VALUE].name = XRT_INPUT_HAND_AIM_ACTIVATE_VALUE;
	hd->base.inputs[ILLIXR_HAND_INPUT_GRASP_VALUE].name = XRT_INPUT_HAND_GRASP_VALUE;
	hd->base.inputs[ILLIXR_HAND_INPUT_PINCH_VALUE].name = XRT_INPUT_HAND_PINCH_VALUE;
	hd->base.inputs[ILLIXR_HAND_INPUT_AIM_ACTIVATE_READY].name = XRT_INPUT_HAND_AIM_ACTIVATE_READY;
	hd->base.inputs[ILLIXR_HAND_INPUT_GRASP_READY].name = XRT_INPUT_HAND_GRASP_READY;
	hd->base.inputs[ILLIXR_HAND_INPUT_PINCH_READY].name = XRT_INPUT_HAND_PINCH_READY;
	u_var_add_root(hd, hd->base.str, true);

	printf("[ILLIXR] Hand interaction device created: %s\n", hd->base.str);

	printf("[ILLIXR]   device_type:      %d\n", (int)hd->base.device_type);
	printf("[ILLIXR]   input count:      %d\n", ILLIXR_HAND_INPUT_COUNT);
	printf("[ILLIXR]   update_inputs:    %p\n", (void *)hd->base.update_inputs);
	printf("[ILLIXR]   get_tracked_pose: %p\n", (void *)hd->base.get_tracked_pose);
	printf("[ILLIXR] Returning device at address: %p\n", (void *)&hd->base);

	return &hd->base;
}
