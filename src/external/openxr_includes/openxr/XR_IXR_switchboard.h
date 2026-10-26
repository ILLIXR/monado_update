#pragma once

#include <openxr/openxr.h>

#ifdef __cplusplus
extern "C" {
#endif

#define XR_IXR_switchboard 1
#define XR_IXR_switchboard_SPEC_VERSION 1
#define XR_IXR_SWITCHBOARD_EXTENSION_NAME "XR_IXR_switchboard"

#define XR_TYPE_ENVELOPE_ILLIXR ((XrStructureType)1000999001)
#define XR_TYPE_SYSTEM_SWITCHBOARD_PROPERTIES_ILLIXR ((XrStructureType)1000999002)

XR_STRUCT_ENUM(XR_TYPE_ENVELOPE_IXR, 1000999000);
XR_STRUCT_ENUM(XR_TYPE_SYSTEM_SWITCHBOARD_PROPERTIES_IXR, 1000999001);

#define XR_ERROR_TOPIC_UNAVAILABLE_IXR ((XrResult)-1000999000)

typedef struct XrEnvelopeIXR {
    XrStructureType          type;
    const void* XR_MAY_ALIAS next;
    uint32_t                 version;
    uint32_t                 dataType;
    uint32_t                 dataVersion;
    uint32_t                 dataCapacity;
    uint32_t                 dataSize;
    uint32_t                 padding;
    uint64_t                 sequence;
    XrTime                   timestamp;
    void*                    data;
} XrEnvelopeIXR;

typedef struct XrSystemSwitchboardPropertiesIXR {
    XrStructureType       type;
    void* XR_MAY_ALIAS    next;
    XrBool32              supportsSwitchboard;
} XrSystemSwitchboardPropertiesIXR;

typedef XrResult(XRAPI_PTR* PFN_xrPublishIXR)(XrSession session, uint32_t topic, const XrEnvelopeIXR* envelope);
typedef XrResult(XRAPI_PTR* PFN_xrReadLatestIXR)(XrSession session, uint32_t topicId, XrEnvelopeIXR* envelope);

// Prototypes, only for code that links the functions directly, such as Monado itself
#ifndef XR_NO_PROTOTYPES
#ifdef XR_EXTENSION_PROTOTYPES
XRAPI_ATTR XrResult XRAPI_CALL xrPublishIXR(XrSession session, uint32_t topicId, const XrEnvelopeIXR* envelope);
XRAPI_ATTR XrResult XRAPI_CALL xrReadLatestIXR(XrSession session, uint32_t topicId,, XrEnvelopeIXR* envelope);
#endif
#endif

#ifdef __cplusplus
}
#endif
