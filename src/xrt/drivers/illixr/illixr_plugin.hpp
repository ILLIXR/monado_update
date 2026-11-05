#pragma once

#include "illixr/phonebook.hpp"
#include "illixr/plugin.hpp"
#include "illixr/switchboard.hpp"
#include "illixr/threadloop.hpp"
#include "illixr/data_format/hmd_config.hpp"
#include "illixr/data_format/pose_prediction.hpp"
#include "illixr/data_format/latency_data.hpp"
#include "illixr/data_format/misc.hpp"
#include "illixr/data_format/poses/hand_interaction_pose.hpp"
#include "illixr/data_format/poses/hand_pose.hpp"
#include "illixr/data_format/poses/palm_pose.hpp"
#include "illixr/data_format/serialization/hmd_config.hpp"

#include "illixr/vk/display_provider.hpp"
#include "illixr/vk/render_pass.hpp"
#include "illixr/vk/vulkan_objects.hpp"

#include <openxr/openxr.h>
#include "xrt/xrt_defines.h"

#include "payload_traits.hpp"
#ifdef ILLIXR_DUMP_AUDIO
#include "audio_dump.hpp"
#endif

using namespace ILLIXR;
using namespace ILLIXR::data_format;
using namespace ILLIXR::vulkan;
using namespace std::chrono_literals;

class monado_compositor_app : public app
{
};

// Simulated plugin class for an instance during phonebook registration
class MY_EXPORT_API illixr_plugin : public threadloop
{
public:
	illixr_plugin(const std::string &name_, phonebook *pb_);
	std::atomic<bool> ready = false;

	void _p_one_iteration() override;
	xrt_result_t
	publish_topic(uint32_t topic_id, const xrt_illixr_payload_info *info, const void *data) noexcept;

	xrt_result_t
	read_topic_latest(uint32_t topic_id, xrt_illixr_payload_info *info, void *data) noexcept;
	
	xrt_result_t
	read_topic_next(uint32_t topic_id, xrt_illixr_payload_info *info, void *data) noexcept;

#ifdef ILLIXR_DUMP_AUDIO
	void capture_from_server(const switchboard::ptr<const ixr_event<ixr_audio_chunk>>& event);
	void capture_from_client(const switchboard::ptr<const ixr_event<ixr_audio_chunk>>& event);
#endif
	bool offload_frames = false;
	int sleep_time = -1;
#ifdef USING_OPENXR
	bool hand_tracking_enabled_ = false;
	bool palm_poses_enabled_ = false;
	bool hand_interactions_enabled_ = false;
#endif

	phonebook *pb;
	const std::shared_ptr<switchboard> sb;
	const std::shared_ptr<pose_prediction> sb_pose;
	const std::shared_ptr<relative_clock> sb_clock;
	std::shared_ptr<timewarp> sb_timewarp;
	std::shared_ptr<vulkan::buffer_pool<BUFFER_TYPE>> buffer_pool;

	std::shared_ptr<display_provider> ds;
	switchboard::writer<switchboard::event_wrapper<time_point>> _m_vsync;
#ifdef ILLIXR_DUMP_AUDIO
	switchboard::network_writer<ixr_event<ixr_audio_chunk>> audio_writer_;
	switchboard::writer<ixr_event<ixr_audio_chunk>> audio_relay_;
#endif
	// Pose data readers - topics published by offload_rendering_server
#ifdef USING_OPENXR
	switchboard::reader<pose::hand_joint_poses_pair> hand_pose_reader_;
	switchboard::reader<pose::hand_interaction_poses_pair> hand_interaction_reader_;
	switchboard::reader<pose::palm_poses_pair> palm_pose_reader_;
#endif
	switchboard::reader<data_format::latency_ping> latency_reader_;
	topic_registry registry_;
	BUFFER_TYPE last_pose{};
	
#ifdef ILLIXR_DUMP_AUDIO
	audio_dump server_dump_;
	audio_dump client_dump_;
#endif
};
