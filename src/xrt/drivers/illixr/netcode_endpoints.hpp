#pragma once

#include "topic_registry.hpp"

#include "illixr/data_format/netcode/netcode.hpp"
#include "illixr/data_format/netcode/netcode_queue.hpp"
#include "illixr/data_format/openxr_payloads.h"
#include "illixr/data_format/netcode/records.hpp"
#include "illixr/network/network_backend.hpp"
#include "illixr/switchboard.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <memory>
#include <string>

namespace ILLIXR {

/*
 * XR_IXR_switchboard endpoints for netcode (see ixr_netcode.h). Unlike add_writer/add_reader,
 * these do not wrap a plain C struct in ixr_event<Payload>: they convert between the byte layout
 * the app uses (ixr_netcode_header + payload records, ixr_netcode_control) and the switchboard
 * events netcode.rx and netcode.tx use (netcode_message, netcode_event_list, netcode_ack).
 * Register them with topic_registry::add_endpoint(); see register_netcode_endpoints() below.
 */

// The result for a control command the backend could not carry out. Replace it with a dedicated
// code if you add one; until then the app sees it as a failed publish, which is all it checks.
constexpr xrt_result_t XRT_NETCODE_COMMAND_FAILED = XRT_ERROR_ILLIXR_PAYLOAD_MISMATCH;

inline bool netcode_payload_matches(const xrt_illixr_payload_info& info, uint32_t data_type) {
	return info.data_type == data_type && info.data_version == IXR_NETCODE_DATA_VERSION;
}

// The highest sequence the app has acknowledged, shared by the ack and events endpoints.
using netcode_ack_state = std::atomic<uint64_t>;

// App -> ILLIXR: one record, sent to the peer its header names.
class netcode_send_endpoint : public topic_endpoint {
public:
	explicit netcode_send_endpoint(const std::shared_ptr<switchboard>& sb)
	    : writer_{sb->get_writer<data_format::netcode_message>(std::string(netcode::OUTBOUND_TOPIC))} { }

	xrt_result_t publish(const xrt_illixr_payload_info& info, const void* data) override {
		netcode_header header{};
		const uint8_t* payload = nullptr;
		if (!netcode_payload_matches(info, IXR_DATA_TYPE_NETCODE_RECORD) ||
		    !netcode::unpack_record(static_cast<const uint8_t*>(data), info.data_size, header, payload)) {
			return XRT_ERROR_ILLIXR_PAYLOAD_MISMATCH;
		}
		writer_.put(writer_.allocate(header, payload, header.payload_size));
		return XRT_SUCCESS;
	}

private:
	switchboard::writer<data_format::netcode_message> writer_;
};

// ILLIXR -> app: the events newer than the last ack, packed into the app's buffer.
class netcode_events_endpoint : public topic_endpoint {
public:
	netcode_events_endpoint(const std::shared_ptr<switchboard>& sb, std::shared_ptr<const netcode_ack_state> acked)
	    : reader_{sb->get_reader<data_format::netcode_event_list>(std::string(netcode::EVENTS_TOPIC))}
	      , acked_{std::move(acked)} { }

	xrt_result_t read_latest(xrt_illixr_payload_info& info, void* data) override {
		if (!netcode_payload_matches(info, IXR_DATA_TYPE_NETCODE_RECORDS)) {
			return XRT_ERROR_ILLIXR_PAYLOAD_MISMATCH;
		}

		info.data_size    = 0;
		info.sequence     = 0;
		info.timestamp_ns = 0;
		switchboard::ptr<const data_format::netcode_event_list> snapshot = reader_.get_ro_nullable();
		if (!snapshot) {
			// Nothing published yet: no events.
			return XRT_SUCCESS;
		}

		const std::size_t           capacity = data != nullptr ? info.data_capacity : 0;
		const netcode::pack_result packed =
		    netcode::pack_events(*snapshot, acked_->load(), static_cast<uint8_t*>(data), capacity);
		// As with every reader: always report the size, copy only what fits. When not even the first
		// waiting record fits, its size exceeds the capacity and the app is told to grow its buffer.
		info.data_size    = packed.count > 0 ? packed.bytes : packed.first_needed;
		info.sequence     = packed.last_sequence;
		info.timestamp_ns = packed.last_timestamp_ns;
		return XRT_SUCCESS;
	}

private:
	switchboard::reader<data_format::netcode_event_list> reader_;
	std::shared_ptr<const netcode_ack_state>             acked_;
};

// App -> ILLIXR: every event up to this sequence has been consumed.
class netcode_ack_endpoint : public topic_endpoint {
public:
	netcode_ack_endpoint(const std::shared_ptr<switchboard>& sb, std::shared_ptr<netcode_ack_state> acked)
	    : writer_{sb->get_writer<data_format::netcode_ack>(std::string(netcode::ACK_TOPIC))}
	      , acked_{std::move(acked)} { }

	xrt_result_t publish(const xrt_illixr_payload_info& info, const void* data) override {
		if (!netcode_payload_matches(info, IXR_DATA_TYPE_NETCODE_ACK) || info.data_size != sizeof(uint64_t) ||
		    data == nullptr) {
			return XRT_ERROR_ILLIXR_PAYLOAD_MISMATCH;
		}
		uint64_t sequence;
		std::memcpy(&sequence, data, sizeof(sequence));
		// The events endpoint stops sending what the app has acknowledged even before netcode.rx
		// has trimmed it.
		acked_->store(sequence);
		writer_.put(writer_.allocate(sequence));
		return XRT_SUCCESS;
	}

private:
	switchboard::writer<data_format::netcode_ack> writer_;
	std::shared_ptr<netcode_ack_state>            acked_;
};

// App -> ILLIXR: join or leave the session. Carried out before returning, so a refused listen or
// dial is the publish's result.
class netcode_control_endpoint : public topic_endpoint {
public:
	explicit netcode_control_endpoint(std::shared_ptr<network::tcp_backend> backend)
	    : backend_{std::move(backend)} { }

	xrt_result_t publish(const xrt_illixr_payload_info& info, const void* data) override {
		if (!netcode_payload_matches(info, IXR_DATA_TYPE_NETCODE_CONTROL) ||
		    info.data_size != sizeof(netcode_control) || data == nullptr) {
			return XRT_ERROR_ILLIXR_PAYLOAD_MISMATCH;
		}
		netcode_control control;
		std::memcpy(&control, data, sizeof(control));
		control.address[ILLIXR_NETCODE_ADDRESS_LENGTH - 1] = '\0';
		const std::string address{control.address};

		switch (control.command) {
		case ILLIXR_NETCODE_COMMAND_LISTEN:
			return backend_->listen_for_peers(address, static_cast<int>(control.port)) ? XRT_SUCCESS
			                                                                           : XRT_NETCODE_COMMAND_FAILED;
		case ILLIXR_NETCODE_COMMAND_CONNECT:
			return backend_->connect_to_peer(address, static_cast<int>(control.port)) ? XRT_SUCCESS
			                                                                          : XRT_NETCODE_COMMAND_FAILED;
		case ILLIXR_NETCODE_COMMAND_DISCONNECT:
			backend_->disconnect_peer(control.peer_id);
			return XRT_SUCCESS;
		case ILLIXR_NETCODE_COMMAND_LEAVE:
			backend_->leave_session();
			return XRT_SUCCESS;
		default:
			return XRT_ERROR_ILLIXR_PAYLOAD_MISMATCH;
		}
	}

private:
	std::shared_ptr<network::tcp_backend> backend_;
};

/**
 * Register the four netcode topics. Call after the TCP backend is in the phonebook, alongside the
 * add_writer/add_reader calls. The plugins netcode.rx and netcode.tx must also be loaded.
 */
inline void register_netcode_endpoints(topic_registry& registry, const std::shared_ptr<switchboard>& sb,
                           const std::shared_ptr<network::tcp_backend>& backend) {
	auto acked = std::make_shared<netcode_ack_state>(0);
	registry.add_endpoint(IXR_TOPIC_NETCODE_SEND, std::make_unique<netcode_send_endpoint>(sb));
	registry.add_endpoint(IXR_TOPIC_NETCODE_EVENTS, std::make_unique<netcode_events_endpoint>(sb, acked));
	registry.add_endpoint(IXR_TOPIC_NETCODE_ACK, std::make_unique<netcode_ack_endpoint>(sb, acked));
	registry.add_endpoint(IXR_TOPIC_NETCODE_CONTROL, std::make_unique<netcode_control_endpoint>(backend));
}

} // namespace ILLIXR
