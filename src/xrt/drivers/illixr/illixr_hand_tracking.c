// Copyright 2020-2021, The Board of Trustees of the University of Illinois.
// SPDX-License-Identifier: BSL-1.0

#include "illixr_component.h"
#include "xrt/xrt_device.h"
#include <stdio.h>

static void
illixr_hmd_get_hand_tracking_impl(struct xrt_device *xdev,
                                  enum xrt_input_name name,
                                  int64_t desired_timestamp_ns,
                                  struct xrt_hand_joint_set *out_value,
                                  int64_t *out_timestamp_ns)
{
	(void)xdev;
	(void)desired_timestamp_ns;

	fprintf(stderr, "[ILLIXR] Hand tracking called for input %d\n", name);
	fflush(stderr);

	// Check if hand tracking is supported
	if (!illixr_hand_tracking_supported()) {
		fprintf(stderr, "[ILLIXR] Hand tracking not supported\n");
		fflush(stderr);
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
		fprintf(stderr, "[ILLIXR] Unknown hand tracking input: %d\n", name);
		fflush(stderr);
		out_value->is_active = false;
		return;
	}

	// Get hand data from ILLIXR
	struct illixr_single_hand hand_data;
	if (!illixr_read_single_hand(hand_index, &hand_data)) {
		out_value->is_active = false;
		return;
	}

	// Set active state
	out_value->is_active = hand_data.is_active;

	if (!hand_data.is_active) {
		return;
	}

	// Convert all joints
	for (int i = 0; i < XRT_HAND_JOINT_COUNT && i < ILLIXR_HAND_JOINT_COUNT; i++) {
		struct xrt_hand_joint_value *dst = &out_value->values.hand_joint_set_default[i];
		const struct illixr_hand_joint *src = &hand_data.joints[i];

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

		// Set relation flags
		enum xrt_space_relation_flags flags = (enum xrt_space_relation_flags)(
		    XRT_SPACE_RELATION_ORIENTATION_VALID_BIT | XRT_SPACE_RELATION_POSITION_VALID_BIT |
		    XRT_SPACE_RELATION_ORIENTATION_TRACKED_BIT | XRT_SPACE_RELATION_POSITION_TRACKED_BIT);

		dst->relation.relation_flags = flags;
	}

	// Return current time
	*out_timestamp_ns = 0; // You can implement proper timestamp here

	fprintf(stderr, "[ILLIXR] Hand tracking data returned successfully\n");
	fflush(stderr);
}

illixr_hand_tracking_fn illixr_get_hand_tracking_callback(void)
{
	return illixr_hmd_get_hand_tracking_impl;
}
