// Copyright 2020-2026, The Board of Trustees of the University of Illinois.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  ILLIXR driver interface
 * @author RSIM Group <illixr@cs.illinois.edu>
 * @ingroup drv_illixr
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * @defgroup drv_illixr illixr driver.
 * @ingroup drv
 *
 * @brief illixr driver.
 */

/*!
 * Create a auto prober for illixr devices.
 *
 * @ingroup drv_illixr
 */
struct xrt_auto_prober *
illixr_create_auto_prober(void);

/*!
 * Create the ILLIXR HMD device.
 *
 * Also launches the ILLIXR runtime and registers the monado plugin.
 * Must be called before illixr_hand_device_create.
 *
 * @ingroup drv_illixr
 */
struct xrt_device *
illixr_hmd_create(const char *path, const char *comp);

/*!
 * Create an XR_EXT_hand_interaction device for one hand.
 *
 * Must be called after illixr_hmd_create (which starts the ILLIXR runtime
 * and sets up the switchboard readers used by this device).
 *
 * @param hand  0 for left hand, 1 for right hand
 * @ingroup drv_illixr
 */
struct xrt_device *
illixr_hand_tracking_device_create(int hand, struct xrt_tracking_origin *origin);

/*!
 * Create an XR_EXT_hand_interaction device for one hand.
 *
 * Must be called after illixr_hmd_create (which starts the ILLIXR runtime
 * and sets up the switchboard readers used by this device).
 *
 * @param hand  0 for left hand, 1 for right hand
 * @ingroup drv_illixr
 */
struct xrt_device *
illixr_hand_interaction_device_create(int hand, struct xrt_tracking_origin *origin);



#ifdef __cplusplus
}
#endif
