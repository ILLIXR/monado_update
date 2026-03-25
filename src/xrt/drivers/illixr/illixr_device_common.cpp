#include "illixr_device_common.h"

#include <chrono>
#include <sstream>

// Include os_time.h for timestamp utilities if available
// This provides os_monotonic_get_ns() on supported platforms
#ifdef XRT_HAVE_TIMESPEC
#include "os/os_time.h"
#endif

/**
 * @brief Get current monotonic time in nanoseconds (portable implementation)
 */
uint64_t get_timestamp_ns(void)
{
#if defined(XRT_HAVE_TIMESPEC) && !defined(_WIN32)
	// Use Monado's utility if available on non-Windows
	return os_monotonic_get_ns();
#else
	// Portable fallback using C++ chrono
	auto now = std::chrono::steady_clock::now();
	auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch());
	return static_cast<uint64_t>(ns.count());
#endif
}


