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

using namespace ILLIXR;
using namespace ILLIXR::data_format;
using namespace ILLIXR::vulkan;
using namespace std::chrono_literals;

class monado_vulkan_display_provider : public display_provider
{
};

class monado_compositor_app : public app
{
};

// Simulated plugin class for an instance during phonebook registration
class MY_EXPORT_API illixr_plugin : public threadloop
{
public:
	illixr_plugin(const std::string &name_, phonebook *pb_);
	hmd_config get_config(const float scale);
	
	void ipd_callback(const switchboard::ptr<const data_format::ipd>& datum);
	
	void _p_thread_setup() override;
	void _p_one_iteration() override;
	float ipd() const;
	std::atomic<bool> ready = false;

	bool offload_frames             = false;
	int  sleep_time                 = -1;
	bool hand_tracking_enabled_     = false;
	bool palm_poses_enabled_        = false;
	bool hand_interactions_enabled_ = false;

	phonebook *pb;
	const std::shared_ptr<switchboard>      sb;
	const std::shared_ptr<pose_prediction>  sb_pose;
	const std::shared_ptr<relative_clock>   sb_clock;
	std::shared_ptr<timewarp>               sb_timewarp;
	std::shared_ptr<vulkan::buffer_pool<BUFFER_TYPE>> buffer_pool;

	std::shared_ptr<display_provider> ds;
	switchboard::writer<switchboard::event_wrapper<time_point>> _m_vsync;
	switchboard::network_writer<data_format::illixr_signal> signal_writer_;
	
	// Pose data readers — topics published by offload_rendering_server
	switchboard::reader<pose::hand_joint_poses_pair>       hand_pose_reader_;
	switchboard::reader<pose::hand_interaction_poses_pair> hand_interaction_reader_;
	switchboard::reader<pose::palm_poses_pair>             palm_pose_reader_;
	switchboard::reader<data_format::latency_ping>         latency_reader_;
	switchboard::reader<data_format::hmd_config_data>      config_reader_;
	
	data_format::ipd current_ipd_{64.f};
	mutable std::mutex       ipd_mutex_;
	BUFFER_TYPE last_pose{};
	data_format::hmd_config_data hmd_config_{};
	uint8_t status_{0};
};
