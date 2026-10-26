#include "illixr_plugin.hpp"
#include "illixr_display_provider.hpp"

#include "illixr/data_format/poses/head_pose.hpp"

illixr_plugin::illixr_plugin(const std::string &name_, phonebook *pb_)
    : threadloop{name_, pb_}, pb{pb_}, sb{phonebook_->lookup_impl<switchboard>()},
      sb_pose{phonebook_->lookup_impl<pose_prediction>()}, sb_clock{phonebook_->lookup_impl<relative_clock>()},
      ds{std::make_shared<monado_vulkan_display_provider>()},
      _m_vsync{sb->get_writer<switchboard::event_wrapper<time_point>>("vsync_estimate")},
#ifdef USING_OPENXR
      hand_pose_reader_{sb->get_reader<pose::hand_joint_poses_pair>("hand_poses")},
      hand_interaction_reader_{sb->get_reader<pose::hand_interaction_poses_pair>("hand_interactions")},
      palm_pose_reader_{sb->get_reader<pose::palm_poses_pair>("palm_poses")}
#endif

      ,
      latency_reader_{sb->get_reader<latency_ping>("latency_ping")},
      registry_{sb}
{
    // App -> ILLIXR
    registry_.add_writer<ixr_foo_data>(IXR_TOPIC_FOO, "oxr_input");

    // ILLIXR -> App
    registry_.add_reader<ixr_foo_data>(IXR_TOPIC_FOO_ECHO, "oxr_output");
	sb_timewarp = pb_->lookup_impl<timewarp>();

	offload_frames = sb->get_env_bool("ILLIXR_OFFLOAD_FRAMES", "false");

	sleep_time = sb->get_env_int("ILLIXR_COMPOSITOR_SLEEP_NS", 0);

#ifdef USING_OPENXR
	// Check if hand tracking is enabled
	hand_tracking_enabled_ = sb->get_env_bool("ILLIXR_USE_HAND_TRACKING", (offload_frames) ? "true" : "false");

	palm_poses_enabled_ = sb->get_env_bool("ILLIXR_USE_PALM_POSES", "false");

	hand_interactions_enabled_ = sb->get_env_bool("ILLIXR_USE_HAND_INTERACTIONS", "false");
#endif
}
xrt_result_t
illixr_plugin::publish_topic(uint32_t topic_id, const xrt_illixr_payload_info *info, const void *data) noexcept
{
    if (info == nullptr) {
        return XRT_ERROR_ILLIXR_PAYLOAD_MISMATCH;
    }

    try {
        topic_endpoint* endpoint = registry_.find(topic_id);
        if (endpoint == nullptr) {
            return XRT_ERROR_ILLIXR_TOPIC_UNAVAILABLE;
        }
        return endpoint->publish(*info, data);
    } catch (const std::bad_alloc&) {
        return XRT_ERROR_ALLOCATION;
    } catch (const std::exception& e) {
        spdlog::error("publish_topic({}): {}", topic_id, e.what());
        return XRT_ERROR_ILLIXR_INTERNAL;
    } catch (...) {
        return XRT_ERROR_ILLIXR_INTERNAL;
    }
}

xrt_result_t
illixr_plugin::read_topic_latest(uint32_t topic_id, xrt_illixr_payload_info *info, void *data) noexcept
{
    if (info == nullptr) {
        return XRT_ERROR_ILLIXR_PAYLOAD_MISMATCH;
    }

    try {
        topic_endpoint* endpoint = registry_.find(topic_id);
        if (endpoint == nullptr) {
            return XRT_ERROR_ILLIXR_TOPIC_UNAVAILABLE;
        }
        return endpoint->read_latest(*info, data);
    } catch (const std::bad_alloc&) {
        return XRT_ERROR_ALLOCATION;
    } catch (const std::exception& e) {
        spdlog::error("read_topic_latest({}): {}", topic_id, e.what());
        return XRT_ERROR_ILLIXR_INTERNAL;
    } catch (...) {
        return XRT_ERROR_ILLIXR_INTERNAL;
    }
}

void illixr_plugin::_p_one_iteration()
{
	std::this_thread::sleep_for(100ms);
}