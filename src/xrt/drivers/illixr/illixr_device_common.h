#pragma once

#include <stdint.h>
#include <stdio.h>

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

/**
 * @brief Get current monotonic time in nanoseconds (portable implementation)
 */
uint64_t
get_timestamp_ns(void);
