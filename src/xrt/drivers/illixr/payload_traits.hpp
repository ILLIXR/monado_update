#pragma once

#include "topic_registry.hpp"

namespace ILLIXR {

ILLIXR_PAYLOAD_TRAITS(ixr_foo_data, IXR_DATA_TYPE_FOO, IXR_FOO_DATA_VERSION)
ILLIXR_PAYLOAD_TRAITS(ixr_audio_chunk, IXR_DATA_TYPE_AUDIO_CHUNK, IXR_AUDIO_CHUNK_DATA_VERSION)

static_assert(sizeof(ixr_audio_chunk) == 8216, "ixr_audio_chunk layout changed");
} // namespace ILLIXR
