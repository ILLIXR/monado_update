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
