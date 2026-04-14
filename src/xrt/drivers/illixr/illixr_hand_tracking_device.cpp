// Copyright 2020-2026, The Board of Trustees of the University of Illinois.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  XRT_DEVICE_TYPE_HAND_TRACKER device
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
		g_debug_log = fopen("D:\\illixr_hand_debug.log", "w");
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

// clang-format on

struct illixr_hand_tracking_device
{
	struct xrt_device base;
	int hand; //!< 0 = left, 1 = right
	bool print_spew;
	bool print_debug;
};

static inline struct illixr_hand_tracking_device *
illixr_hand_tracking_device(struct xrt_device *xdev)
{
	return (struct illixr_hand_tracking_device *)xdev;
}

#define HD_ERROR(hd, ...)                                                                                              \
	do {                                                                                                           \
		fprintf(stderr, "%s [hand=%d] - ", __func__, hd->hand);                                                \
		fprintf(stderr, __VA_ARGS__);                                                                          \
		fprintf(stderr, "\n");                                                                                 \
	} while (false)

static void
illixr_hand_tracking_device_destroy(struct xrt_device *xdev)
{
	// The ILLIXR runtime is owned by the HMD device; do not stop it here.
	u_var_remove_root(xdev);
	u_device_free(xdev);
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
illixr_hand_tracking_device_get_tracking(struct xrt_device *xdev,
                                         enum xrt_input_name name,
					 int64_t desired_timestamp_ns,
					 struct xrt_hand_joint_set *out_value,
					 int64_t *out_timestamp_ns)
{
	(void)desired_timestamp_ns;
	// Return current time
	*out_timestamp_ns = (int64_t)get_timestamp_ns();

	struct illixr_hand_tracking_device *hd = illixr_hand_tracking_device(xdev);

#ifdef BUILD_WITH_LOGGING
	ht_log("Hand tracking called for input - %d", name);
#endif

	// Check if hand tracking is supported
	if (!illixr_hand_tracking_supported()) {
#ifdef BUILD_WITH_LOGGING
		ht_log("Hand tracking not supported");
#endif
		out_value->is_active = false;
		return;
	}

	// Validate the name matches this device's hand
	xrt_input_name expected =
	    (hd->hand == 0) ? XRT_INPUT_GENERIC_HAND_TRACKING_LEFT : XRT_INPUT_GENERIC_HAND_TRACKING_RIGHT;

	if (name != expected) {
#ifdef BUILD_WITH_LOGGING
		ht_log("unexpected hand tracking input name %d", (int)name);
#endif
		out_value->is_active = false;
		return;
	}

	if (!illixr_read_single_hand(hd->hand, out_value)) {
#ifdef BUILD_WITH_LOGGING
		ht_log("illixr_read_single_hand returned false for hand %d", hd->hand);
#endif
		out_value->is_active = false;
		return;
	}

	if (!out_value->is_active) {
#ifdef BUILD_WITH_LOGGING
		ht_log("Hand %d not active", hd->hand);
#endif
		return;
	}

	// ILLIXR joints are already in world-space so the hand_pose.pose should be identity
	out_value->hand_pose.pose.orientation.w = 1.0f;
	out_value->hand_pose.pose.orientation.x = 0.0f;
	out_value->hand_pose.pose.orientation.y = 0.0f;
	out_value->hand_pose.pose.orientation.z = 0.0f;
	out_value->hand_pose.relation_flags = (enum xrt_space_relation_flags)(
	    XRT_SPACE_RELATION_ORIENTATION_VALID_BIT | XRT_SPACE_RELATION_ORIENTATION_TRACKED_BIT |
	    XRT_SPACE_RELATION_POSITION_VALID_BIT | XRT_SPACE_RELATION_POSITION_TRACKED_BIT);

#ifdef BUILD_WITH_LOGGING
	ht_log("Hand %d tracking data returned: wrist=(%.4f, %.4f, %.4f) flags=0x%x", hd->hand,
	       out_value->values.hand_joint_set_default[1].relation.pose.position.x,
	       out_value->values.hand_joint_set_default[1].relation.pose.position.y,
	       out_value->values.hand_joint_set_default[1].relation.pose.position.z,
	       out_value->values.hand_joint_set_default[1].relation.relation_flags);
#endif
}

static xrt_result_t
illixr_hand_tracking_get_pose(struct xrt_device *xdev,
                              enum xrt_input_name name,
                              int64_t at_timestamp_ns,
                              struct xrt_space_relation *out_relation)
{
	out_relation->pose.orientation.w = 1.0f;
	out_relation->pose.orientation.x = 0.0f;
	out_relation->pose.orientation.y = 0.0f;
	out_relation->pose.orientation.z = 0.0f;
	out_relation->pose.position.x = 0.0f;
	out_relation->pose.position.y = 0.0f;
	out_relation->pose.position.z = 0.0f;
	out_relation->relation_flags = (enum xrt_space_relation_flags)(
	    XRT_SPACE_RELATION_ORIENTATION_VALID_BIT | XRT_SPACE_RELATION_ORIENTATION_TRACKED_BIT |
	    XRT_SPACE_RELATION_POSITION_VALID_BIT | XRT_SPACE_RELATION_POSITION_TRACKED_BIT);
	return XRT_SUCCESS;
}

static xrt_result_t
illixr_hand_tracking_update_inputs(struct xrt_device *xdev)
{
	return XRT_SUCCESS;
}
extern "C" struct xrt_device *
illixr_hand_tracking_device_create(int hand, struct xrt_tracking_origin *origin)
{
	assert(hand == 0 || hand == 1);

	struct illixr_hand_tracking_device *hd =
	    U_DEVICE_ALLOCATE(struct illixr_hand_tracking_device, U_DEVICE_ALLOC_TRACKING_NONE, 1, 0);

	hd->hand = hand;
	hd->print_spew = debug_get_bool_option_illixr_spew();
	hd->print_debug = debug_get_bool_option_illixr_debug();

	hd->base.update_inputs = illixr_hand_tracking_update_inputs;
	hd->base.get_tracked_pose = illixr_hand_tracking_get_pose;
	hd->base.get_hand_tracking = illixr_hand_tracking_device_get_tracking;
	hd->base.destroy = illixr_hand_tracking_device_destroy;

	hd->base.name = XRT_DEVICE_HAND_TRACKER;
	hd->base.device_type = XRT_DEVICE_TYPE_HAND_TRACKER;

	// Share the HMD's tracking origin so Monado applies the same
	// LOCAL-space transform to interaction poses as it does to the head.
	// Without this, Monado uses a garbage or identity origin and the
	// world-space aim pose gets a wrong additional offset applied.
	hd->base.tracking_origin = origin;

	hd->base.orientation_tracking_supported = true;
	hd->base.hand_tracking_supported = true;

	snprintf(hd->base.str, XRT_DEVICE_NAME_LEN, "ILLIXR Hand (%s)", hand == 0 ? "Left" : "Right");
	snprintf(hd->base.serial, XRT_DEVICE_NAME_LEN, "ILLIXR-HAND-%s", hand == 0 ? "L" : "R");

	// ---- Input table ----
	hd->base.inputs[0].name =
	    (hand == 0) ? XRT_INPUT_GENERIC_HAND_TRACKING_LEFT : XRT_INPUT_GENERIC_HAND_TRACKING_RIGHT;
	u_var_add_root(hd, hd->base.str, true);

	printf("[ILLIXR] Hand tracking device created: %s\n", hd->base.str);
	printf("[ILLIXR]   ILLIXR_USE_HAND_TRACKING=%s\n",
	       std::getenv("ILLIXR_USE_HAND_TRACKING") ? std::getenv("ILLIXR_USE_HAND_TRACKING") : "(not set)");
	printf("[ILLIXR]   Hand tracking:          %s\n", hd->base.hand_tracking_supported ? "ENABLED" : "DISABLED");
	printf("[ILLIXR]   get_hand_tracking:       %p\n", (void *)hd->base.get_hand_tracking);

	printf("[ILLIXR]   device_type:      %d\n", (int)hd->base.device_type);
	printf("[ILLIXR]   update_inputs:    %p\n", (void *)hd->base.update_inputs);
	printf("[ILLIXR] Returning device at address: %p\n", (void *)&hd->base);

	return &hd->base;
}
