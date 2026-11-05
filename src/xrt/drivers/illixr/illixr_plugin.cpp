#include "illixr_plugin.hpp"
#include "illixr_display_provider.hpp"

#include "illixr/data_format/poses/head_pose.hpp"
#include "illixr/data_format/openxr_payloads.h"
#include "illixr/data_format/openxr_data.hpp"
#include "illixr/data_format/serialization/openxr_payloads.hpp"

illixr_plugin::illixr_plugin(const std::string &name_, phonebook *pb_)
    : threadloop{name_, pb_}, pb{pb_}, sb{phonebook_->lookup_impl<switchboard>()},
      sb_pose{phonebook_->lookup_impl<pose_prediction>()}, sb_clock{phonebook_->lookup_impl<relative_clock>()},
      ds{std::make_shared<monado_vulkan_display_provider>()},
      _m_vsync{sb->get_writer<switchboard::event_wrapper<time_point>>("vsync_estimate")},
#ifdef ILLIXR_DUMP_AUDIO
      audio_writer_{sb->get_network_writer<ixr_event<ixr_audio_chunk>>("audio_to_client", network::topic_config{network::topic_config::BOOST,
														network::topic_config::TCP})},
      audio_relay_{sb->get_writer<ixr_event<ixr_audio_chunk>>("audio_to_server")},
#endif
#ifdef USING_OPENXR
      hand_pose_reader_{sb->get_reader<pose::hand_joint_poses_pair>("hand_poses")},
      hand_interaction_reader_{sb->get_reader<pose::hand_interaction_poses_pair>("hand_interactions")},
      palm_pose_reader_{sb->get_reader<pose::palm_poses_pair>("palm_poses")}
#endif
      , latency_reader_{sb->get_reader<latency_ping>("latency_ping")},
      registry_{sb}
#ifdef ILLIXR_DUMP_AUDIO
      , server_dump_{sb->get_env("AUDIO_DUMP_DIR", ""), "server_audio"},
      client_dump_{sb->get_env("AUDIO_DUMP_DIR", ""), "client_audio"}
#endif 
{
    register_ixr_serialization();  // DO NOT REMOVE, while this is a dummy call, it forces the linker to include the serialization obj file
    
    // App -> ILLIXR
#ifdef ILLIXR_DUMP_AUDIO
    registry_.add_writer<ixr_audio_chunk>(IXR_TOPIC_AUDIO_CAPTURE, "audio_from_server");

#else
    registry_.add_network_writer<ixr_audio_chunk>(IXR_TOPIC_AUDIO_CAPTURE, "audio_to_client", network::topic_config{network::topic_config::BOOST,
    	                                                                                      network::topic_config::TCP});
#endif
    // ILLIXR -> App
#ifdef ILLIXR_DUMP_AUDIO
    registry_.add_buffered_reader<ixr_audio_chunk>(IXR_TOPIC_AUDIO_INPUT, "audio_to_server");
#else
    registry_.add_buffered_reader<ixr_audio_chunk>(IXR_TOPIC_AUDIO_INPUT, "audio_from_client");
#endif
    
#ifdef ILLIXR_DUMP_AUDIO
    sb->schedule<ixr_event<ixr_audio_chunk>>(id_, "audio_from_server",
     					 [this](const switchboard::ptr<const ixr_event<ixr_audio_chunk>>& event, std::size_t) {
    						   capture_from_server(event);
    						 });
    sb->schedule<ixr_event<ixr_audio_chunk>>(id_, "audio_from_client",
    					     [this](const switchboard::ptr<const ixr_event<ixr_audio_chunk>>& event, std::size_t) {
    					       capture_from_client(event);
    	                                     });
#endif    	                                     
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

#ifdef ILLIXR_DUMP_AUDIO
void illixr_plugin::capture_from_server(const switchboard::ptr<const ixr_event<ixr_audio_chunk>> &event) {
	server_dump_.write(event->data.samples, event->data.frame_count, event->data.channel_count, event->data.sample_rate, event->data.start_frame);
	audio_writer_.put(std::make_shared<ixr_event<ixr_audio_chunk>>(event->data, event->timestamp_ns));
}

void illixr_plugin::capture_from_client(const switchboard::ptr<const ixr_event<ixr_audio_chunk>> &event) {
	client_dump_.write(event->data.samples, event->data.frame_count, event->data.channel_count, event->data.sample_rate, event->data.start_frame);
	audio_relay_.put(std::make_shared<ixr_event<ixr_audio_chunk>>(event->data, event->timestamp_ns));
}
#endif

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

xrt_result_t
illixr_plugin::read_topic_next(uint32_t topic_id, xrt_illixr_payload_info *info, void *data) noexcept
{
	if (info == nullptr) {
		return XRT_ERROR_ILLIXR_PAYLOAD_MISMATCH;
	}
	
	try {
		topic_endpoint* endpoint = registry_.find(topic_id);
		if (endpoint == nullptr) {
			return XRT_ERROR_ILLIXR_TOPIC_UNAVAILABLE;
		}
		return endpoint->read_next(*info, data);
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