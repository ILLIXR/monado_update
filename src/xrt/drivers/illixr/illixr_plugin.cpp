#include "illixr_plugin.hpp"

illixr_plugin::illixr_plugin(const std::string &name_, phonebook *pb_) 
    : threadloop{name_, pb_}
      , pb{pb_}
      , sb{phonebook_->lookup_impl<switchboard>()}
      , sb_pose{phonebook_->lookup_impl<pose_prediction>()}
      , sb_clock{phonebook_->lookup_impl<relative_clock>()}
      , ds{std::make_shared<monado_vulkan_display_provider>()}
      , _m_vsync{sb->get_writer<switchboard::event_wrapper<time_point>>("vsync_estimate")}
      , signal_writer_{sb->get_network_writer<data_format::illixr_signal>("offload_signal", 
                                                                        {network::topic_config::BOOST,
                                                                         network::topic_config::UDP
                                                                        })}
      , hand_pose_reader_{sb->get_reader<pose::hand_joint_poses_pair>("hand_poses")}
      , hand_interaction_reader_{sb->get_reader<pose::hand_interaction_poses_pair>("hand_interactions")}
      , palm_pose_reader_{sb->get_reader<pose::palm_poses_pair>("palm_poses")}
      , latency_reader_{sb->get_reader<latency_ping>("latency_ping")}
      , config_reader_{sb->get_reader<data_format::hmd_config_data>("hmd_config_relay")} {
	sb_timewarp = pb_->lookup_impl<timewarp>();
		
	offload_frames = sb->get_env_bool("ILLIXR_OFFLOAD_FRAMES", "false");
	
	sleep_time = sb->get_env_int("ILLIXR_COMPOSITOR_SLEEP_NS", 0);
	
	// Check if hand tracking is enabled
	hand_tracking_enabled_ = sb->get_env_bool("ILLIXR_USE_HAND_TRACKING", (offload_frames) ? "true" : "false");

	palm_poses_enabled_ = sb->get_env_bool("ILLIXR_USE_PALM_POSES", "false");

	hand_interactions_enabled_ = sb->get_env_bool("ILLIXR_USE_HAND_INTERACTIONS", "false");
	
	signal_writer_.put(std::make_shared<data_format::illixr_signal>(AWAITING_CONFIG));
	status_ = AWAITING_CONFIG;
}

void illixr_plugin::_p_thread_setup() {
	bool have_cfg = false;
	while (!have_cfg) {
		std::shared_ptr<const data_format::hmd_config_data> cfg = config_reader_.get_ro_nullable();
				
		if (cfg != nullptr) {
			hmd_config_ = hmd_config_data(cfg->config, cfg->ipd);
			current_ipd_ = data_format::ipd(hmd_config_.ipd);
			printf("[ILLIXR] have HMD config\n");
			printf("     fov left: %f, %f\n", hmd_config_.config.fov_angle_left[0], hmd_config_.config.fov_angle_left[1]);
			printf("     fov left: %f, %f\n", hmd_config_.config.fov_angle_right[0], hmd_config_.config.fov_angle_right[1]);
			printf("     fov left: %f, %f\n", hmd_config_.config.fov_angle_up[0], hmd_config_.config.fov_angle_up[1]);
			printf("     fov left: %f, %f\n", hmd_config_.config.fov_angle_down[0], hmd_config_.config.fov_angle_down[1]);
			printf("     size: %d, %d\n", hmd_config_.config.recommended_image_width, hmd_config_.config.recommended_image_height);
			printf("     ipd: %f\n", hmd_config_.ipd);
			have_cfg = true;
		} else {
			std::this_thread::sleep_for(10ms);
		}
	}
	signal_writer_.put(std::make_shared<data_format::illixr_signal>(RUNNING));
	status_ = RUNNING;
	sb->schedule<data_format::ipd>(id_, "current_ipd", [&](const switchboard::ptr<const data_format::ipd>& datum, size_t) {
		ipd_callback(datum);
	});
}

void illixr_plugin::ipd_callback(const switchboard::ptr<const data_format::ipd> &datum) {
	const std::lock_guard<std::mutex> lock(ipd_mutex_);
	current_ipd_ = *datum;
}

float illixr_plugin::ipd() const {
	const std::lock_guard<std::mutex> lock(ipd_mutex_);
	return current_ipd_.ipd_m;
}

void illixr_plugin::_p_one_iteration() {
	std::this_thread::sleep_for(100ms);
}

hmd_config illixr_plugin::get_config(const float scale) {
	while (status_ != RUNNING)
		std::this_thread::sleep_for(10ms);
	return hmd_config_.get_cfg(scale);
}