// Copyright 2020-2026, The Board of Trustees of the University of Illinois.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  ILLIXR prober
 * @author RSIM Group <illixr@cs.illinois.edu>
 * @ingroup drv_illixr
 */

#include <stdlib.h>
#include <string.h>

#include "xrt/xrt_prober.h"
#include "util/u_misc.h"
#include "util/u_debug.h"

#include "illixr_interface.h"


struct illixr_prober
{
	struct xrt_auto_prober base;
};

static inline struct illixr_prober *
illixr_prober(struct xrt_auto_prober *p)
{
	return (struct illixr_prober *)p;
}

static void
illixr_prober_destroy(struct xrt_auto_prober *p)
{
	struct illixr_prober *dp = illixr_prober(p);

	free(dp);
}

static int
illixr_prober_autoprobe(struct xrt_auto_prober *xap,
                        cJSON *attached_data,
                        bool no_hmds,
                        struct xrt_prober *xp,
                        struct xrt_device **out_xdevs)
{
	struct illixr_prober *dp = illixr_prober(xap);
	(void)dp;

	if (no_hmds) {
		return 0;
	}

	const char *illixr_path, *illixr_comp;
	illixr_path = getenv("ILLIXR_PATH");
	illixr_comp = getenv("ILLIXR_COMP");
	if (!illixr_path || !illixr_comp) {
		return 0;
	}

	// Check environment variables for INTENT to use hand tracking
	bool ht_enabled = false;

#ifdef ILLIXR_ENABLE_HAND_TRACKING
	const char *ht_env = getenv("ILLIXR_USE_HAND_TRACKING");
	if (ht_env != NULL) {
		char v1[] = "1";
		char v2[] = "true";
		char v3[] = "TRUE";
		char v4[] = "yes";
		char v5[] = "YES";

		ht_enabled = (strcmp(ht_env, v1) == 0 || strcmp(ht_env, v2) == 0 || strcmp(ht_env, v3) == 0 ||
		              strcmp(ht_env, v4) == 0 || strcmp(ht_env, v5) == 0);
	} else {
		// Fall back to ILLIXR_OFFLOAD_FRAMES
		const char *offload_env = getenv("ILLIXR_OFFLOAD_FRAMES");
		if (offload_env != NULL) {
			ht_enabled = (atoi(offload_env) != 0);
		}
	}
#endif
	// HMD device — must be created first; it launches the ILLIXR runtime
	// and registers the monado plugin that owns the switchboard readers.
	out_xdevs[0] = illixr_hmd_create(illixr_path, illixr_comp);
	if (out_xdevs[0] == NULL) {
		return 0;
	}
	if (!ht_enabled)
		return 1;
#ifdef ILLIXR_ENABLE_HAND_TRACKING
	// Hand interaction devices — created after the HMD so the runtime is
	// already running and the switchboard topics are available.
	out_xdevs[1] = illixr_hand_interaction_device_create(0, out_xdevs[0]->tracking_origin); // left
	out_xdevs[2] = illixr_hand_interaction_device_create(1, out_xdevs[0]->tracking_origin); // right
	out_xdevs[3] = illixr_hand_tracking_device_create(0, out_xdevs[0]->tracking_origin);    // left
	out_xdevs[4] = illixr_hand_tracking_device_create(1, out_xdevs[0]->tracking_origin);    // right

	if (out_xdevs[1] == NULL || out_xdevs[2] == NULL) {
		// Non-fatal: HMD still works without hand tracking devices.
		// Clean up whichever device was created and return only the HMD.
		if (out_xdevs[1] != NULL) {
			out_xdevs[1]->destroy(out_xdevs[1]);
			out_xdevs[1] = NULL;
		}
		if (out_xdevs[2] != NULL) {
			out_xdevs[2]->destroy(out_xdevs[2]);
			out_xdevs[2] = NULL;
		}
		return 1;
	}
	if (out_xdevs[3] == NULL || out_xdevs[4] == NULL) {
		// Non-fatal: HMD still works without hand interaction devices.
		// Clean up whichever device was created and return only the HMD.
		if (out_xdevs[3] != NULL) {
			out_xdevs[3]->destroy(out_xdevs[3]);
			out_xdevs[3] = NULL;
		}
		if (out_xdevs[4] != NULL) {
			out_xdevs[4]->destroy(out_xdevs[4]);
			out_xdevs[4] = NULL;
		}
		return 3;
	}

	return 5;
#endif
}

struct xrt_auto_prober *
illixr_create_auto_prober()
{
	struct illixr_prober *dp = U_TYPED_CALLOC(struct illixr_prober);
	dp->base.name = "ILLIXR";
	dp->base.destroy = illixr_prober_destroy;
	dp->base.lelo_dallas_autoprobe = illixr_prober_autoprobe;

	return &dp->base;
}
