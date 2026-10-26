#pragma once

#include "illixr/data_format/openxr_payloads.h"
#include "illixr/switchboard.hpp"
#include "xrt/xrt_device.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ILLIXR {

/*
 * Binds a C payload struct to its data type id and version. Specialize once per payload,
 * using a payload without a specialization is a compile error.
 */
template<typename Payload>
struct payload_traits;

#define ILLIXR_PAYLOAD_TRAITS(PAYLOAD, TYPE_ID, VERSION)       \
    template<>                                                 \
    struct payload_traits<PAYLOAD> {                           \
        static constexpr uint32_t type_id = (TYPE_ID);         \
        static constexpr uint32_t version = (VERSION);         \
    };

/*
 * Wraps a plain C payload so it can travel through switchboard. ILLIXR plugins that produce
 * or consume these topics use ixr_event<Payload> as the topic's event type.
 */
template<typename Payload>
struct ixr_event : public switchboard::event {
    static_assert(std::is_trivially_copyable_v<Payload>, "payloads must be plain C structs");

    Payload data;
    int64_t timestamp_ns; // Monado monotonic clock

    ixr_event(const Payload& payload, int64_t timestamp)
            : data{payload}
            , timestamp_ns{timestamp} { }
};

// Must be the same clock domain as Monado's os_monotonic_get_ns.
inline int64_t now_monotonic_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

template<typename Payload>
bool payload_matches(const xrt_illixr_payload_info& info) {
    return info.data_type == payload_traits<Payload>::type_id && info.data_version == payload_traits<Payload>::version;
}

/*
 * Type-erased endpoint, the registry stores these. The defaults reject the call, so an
 * endpoint used in the wrong direction returns an error.
 */
class topic_endpoint {
public:
    virtual ~topic_endpoint() = default;

    virtual xrt_result_t publish(const xrt_illixr_payload_info& info, const void* data) {
        (void) info;
        (void) data;
        return XRT_ERROR_ILLIXR_PAYLOAD_MISMATCH;
    }

    virtual xrt_result_t read_latest(xrt_illixr_payload_info& info, void* data) {
        (void) info;
        (void) data;
        return XRT_ERROR_ILLIXR_PAYLOAD_MISMATCH;
    }
};

// App -> ILLIXR
template<typename Payload>
class writer_endpoint : public topic_endpoint {
public:
    using event_type = ixr_event<Payload>;

    writer_endpoint(const std::shared_ptr<switchboard>& sb, const std::string& topic_name)
            : writer_{sb->get_writer<event_type>(topic_name)} { }

    xrt_result_t publish(const xrt_illixr_payload_info& info, const void* data) override {
        if (!payload_matches<Payload>(info) || info.data_size != sizeof(Payload) || data == nullptr) {
            return XRT_ERROR_ILLIXR_PAYLOAD_MISMATCH;
        }

        Payload payload;
        std::memcpy(&payload, data, sizeof(Payload)); // data may not be aligned for Payload
        writer_.put(writer_.allocate(payload, now_monotonic_ns()));
        return XRT_SUCCESS;
    }

private:
    switchboard::writer<event_type> writer_;
};

// ILLIXR -> App
template<typename Payload>
class reader_endpoint : public topic_endpoint {
public:
    using event_type = ixr_event<Payload>;

    reader_endpoint(const std::shared_ptr<switchboard>& sb, const std::string& topic_name)
            : reader_{sb->get_reader<event_type>(topic_name)} { }

    xrt_result_t read_latest(xrt_illixr_payload_info& info, void* data) override {
        if (!payload_matches<Payload>(info)) {
            return XRT_ERROR_ILLIXR_PAYLOAD_MISMATCH;
        }

        auto event = reader_.get_ro_nullable();
        if (event == nullptr) {
            // Nothing published yet, sequence 0 tells the app there is no data.
            info.data_size    = 0;
            info.sequence     = 0;
            info.timestamp_ns = 0;
            return XRT_SUCCESS;
        }

        // Always report the size, copy only if it fits.
        info.data_size = sizeof(Payload);
        if (info.data_capacity < sizeof(Payload) || data == nullptr) {
            return XRT_SUCCESS;
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            // Holding the previous event keeps its address from being reused by a new one.
            if (event != last_event_) {
                last_event_ = event;
                ++sequence_;
            }
            info.sequence = sequence_;
        }

        std::memcpy(data, &event->data, sizeof(Payload));
        info.timestamp_ns = event->timestamp_ns;
        return XRT_SUCCESS;
    }

private:
    switchboard::reader<event_type>   reader_;
    std::mutex                        mutex_;
    std::shared_ptr<const event_type> last_event_;
    uint64_t                          sequence_ = 0;
};

/*
 * Topic id -> endpoint. Filled in the plugin constructor and read-only afterward,
 * so lookups need no lock.
 */
class topic_registry {
public:
    explicit topic_registry(std::shared_ptr<switchboard> sb)
            : switchboard_{std::move(sb)} { }

    template<typename Payload>
    void add_writer(uint32_t topic_id, const std::string& topic_name) {
        set(topic_id, std::make_unique<writer_endpoint<Payload>>(switchboard_, topic_name));
    }

    template<typename Payload>
    void add_reader(uint32_t topic_id, const std::string& topic_name) {
        set(topic_id, std::make_unique<reader_endpoint<Payload>>(switchboard_, topic_name));
    }

    [[nodiscard]] topic_endpoint* find(uint32_t topic_id) const {
        return topic_id < endpoints_.size() ? endpoints_[topic_id].get() : nullptr;
    }

private:
    void set(uint32_t topic_id, std::unique_ptr<topic_endpoint> endpoint) {
        if (topic_id >= endpoints_.size() || endpoints_[topic_id] != nullptr) {
            throw std::runtime_error("invalid or duplicate topic id " + std::to_string(topic_id));
        }
        endpoints_[topic_id] = std::move(endpoint);
    }

    std::shared_ptr<switchboard>                                switchboard_;
    std::array<std::unique_ptr<topic_endpoint>, IXR_TOPIC_COUNT> endpoints_;
};

} // namespace ILLIXR
