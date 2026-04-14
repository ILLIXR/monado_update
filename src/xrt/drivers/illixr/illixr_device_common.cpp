// Copyright 2020-2026, The Board of Trustees of the University of Illinois.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  ILLIXR driver common functions
 * @author RSIM Group <illixr@cs.illinois.edu>
 * @ingroup drv_illixr
 */
 #include "illixr_device_common.h"

#include <chrono>
#include <sstream>

// Include os_time.h for timestamp utilities if available
// This provides os_monotonic_get_ns() on supported platforms
#include "os/os_time.h"

/**
 * @brief Get current monotonic time in nanoseconds (portable implementation)
 */
uint64_t get_timestamp_ns(void)
{
	return os_monotonic_get_ns();
}
