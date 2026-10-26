#include "openxr/openxr.h"
#include "xrt/xrt_compiler.h"

#include "math/m_api.h"

#include "util/u_debug.h"
#include "util/u_trace_marker.h"


#include "oxr_objects.h"
#include "oxr_logger.h"

#include "oxr_api_funcs.h"
#include "oxr_api_verify.h"


XRAPI_ATTR XrResult XRAPI_CALL
oxr_xrPublishIXR(XrSession session, uint32_t topicId, const XrEnvelopeIXR *envelope)
{
    OXR_TRACE_MARKER();

    struct oxr_session *sess = NULL;
    struct oxr_logger log;
    OXR_VERIFY_SESSION_AND_INIT_LOG(&log, session, sess, "xrPublishIXR");
    OXR_VERIFY_EXTENSION(&log, sess->sys->inst, IXR_switchboard);
    OXR_VERIFY_ARG_TYPE_AND_NOT_NULL(&log, envelope, XR_TYPE_ENVELOPE_IXR);

    return oxr_illixr_publish(&log, sess, topicId, envelope);
}

XRAPI_ATTR XrResult XRAPI_CALL
oxr_xrReadLatestIXR(XrSession session, uint32_t topicId, XrEnvelopeIXR *envelope) {
    OXR_TRACE_MARKER();

    struct oxr_session *sess = NULL;
    struct oxr_logger log;
    OXR_VERIFY_SESSION_AND_INIT_LOG(&log, session, sess, "xrPublishIXR");
    OXR_VERIFY_EXTENSION(&log, sess->sys->inst, IXR_switchboard);
    OXR_VERIFY_ARG_TYPE_AND_NOT_NULL(&log, envelope, XR_TYPE_ENVELOPE_IXR);

    return oxr_illixr_read_latest(&log, sess, topicId, envelope);
}
