// Copyright 2026, ILLIXR
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  ILLIXR switchboard topic support.
 * @ingroup oxr_main
 */

#include "oxr_extension_support.h"

#include "xrt/xrt_device.h"
#include "util/u_time.h"

#include "oxr_objects.h"
#include "oxr_logger.h"

#ifdef OXR_HAVE_IXR_switchboard
/*
 *
 * Helpers.
 *
 */

// Envelope schema version this runtime understands.
#define OXR_IXR_ENVELOPE_VERSION 1

static XrResult
illixr_result_to_xr(struct oxr_logger *log, xrt_result_t xret, const char *func)
{
	switch (xret) {
	case XRT_SUCCESS: return XR_SUCCESS;
	case XRT_ERROR_ILLIXR_TOPIC_UNAVAILABLE:
		return oxr_error(log, XR_ERROR_TOPIC_UNAVAILABLE_IXR, "(%s) Topic is not available", func);
	case XRT_ERROR_ILLIXR_PAYLOAD_MISMATCH:
		return oxr_error(log, XR_ERROR_VALIDATION_FAILURE, "(%s) Payload type or version does not match topic",
		                 func);
	case XRT_ERROR_IPC_FAILURE: return oxr_error(log, XR_ERROR_RUNTIME_FAILURE, "(%s) IPC failure", func);
	default: return oxr_error(log, XR_ERROR_RUNTIME_FAILURE, "(%s) Unexpected error %d", func, (int)xret);
	}
}

static XrResult
verify_envelope(struct oxr_logger *log, const XrEnvelopeIXR *envelope, uint32_t size, const char *func)
{
	if (envelope->version != OXR_IXR_ENVELOPE_VERSION) {
		return oxr_error(log, XR_ERROR_VALIDATION_FAILURE, "(%s) Unsupported envelope version %u", func,
		                 envelope->version);
	}

	if (size > 0 && envelope->data == NULL) {
		return oxr_error(log, XR_ERROR_VALIDATION_FAILURE, "(%s) data is NULL with a non-zero size", func);
	}

	return XR_SUCCESS;
}

/*
 *
 * 'Exported' functions.
 *
 */


XrResult
oxr_illixr_publish(struct oxr_logger *log, struct oxr_session *session, uint32_t topicId, const XrEnvelopeIXR *envelope)
{
	struct xrt_device *xdev = oxr_system_get_illixr_device(session->sys);
	if (xdev == NULL) {
		return oxr_error(log, XR_ERROR_FUNCTION_UNSUPPORTED, "(xrPublishIXR) No ILLIXR device in system");
	}

	XrResult ret = verify_envelope(log, envelope, envelope->dataSize, "xrPublishIXR");
	if (ret != XR_SUCCESS) {
		return ret;
	}

	struct xrt_illixr_payload_info info = {
	    .data_type = envelope->dataType,
	    .data_version = envelope->dataVersion,
	    .data_capacity = 0,
	    .data_size = envelope->dataSize,
	    .sequence = 0,
	    .timestamp_ns = 0,
	};

	xrt_result_t xret = xrt_device_illixr_publish(xdev, topicId, &info, envelope->data);

	return illixr_result_to_xr(log, xret, "xrPublishIXR");
}

XrResult
oxr_illixr_read_latest(struct oxr_logger *log, struct oxr_session *session, uint32_t topicId, XrEnvelopeIXR *envelope)
{
	struct xrt_device *xdev = oxr_system_get_illixr_device(session->sys);
	if (xdev == NULL) {
		return oxr_error(log, XR_ERROR_FUNCTION_UNSUPPORTED, "(xrReadLatestIXR) No ILLIXR device in system");
	}

	XrResult ret = verify_envelope(log, envelope, envelope->dataCapacity, "xrReadLatestIXR");
	if (ret != XR_SUCCESS) {
		return ret;
	}

	struct xrt_illixr_payload_info info = {
	    .data_type = envelope->dataType,
	    .data_version = envelope->dataVersion,
	    .data_capacity = envelope->dataCapacity,
	    .data_size = 0,
	    .sequence = 0,
	    .timestamp_ns = 0,
	};

	xrt_result_t xret = xrt_device_illixr_read_latest(xdev, topicId, &info, envelope->data);
	ret = illixr_result_to_xr(log, xret, "xrReadLatestIXR");
	if (ret != XR_SUCCESS) {
		return ret;
	}

	// Always report the size, the app needs it to grow its buffer.
	envelope->dataSize = info.data_size;

	if (info.data_size > envelope->dataCapacity) {
		// Not an oxr_error, this is the normal two-call path and should not be logged as a failure.
		return XR_ERROR_SIZE_INSUFFICIENT;
	}

	envelope->sequence = info.sequence;
	envelope->timestamp = time_state_monotonic_to_ts_ns(session->sys->inst->timekeeping, info.timestamp_ns);

	return XR_SUCCESS;
}
#endif
