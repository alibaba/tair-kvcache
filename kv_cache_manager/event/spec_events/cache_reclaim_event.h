#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "kv_cache_manager/event/base_event.h"

namespace kv_cache_manager {

class CacheReclaimSubmitEvent : public BaseEvent {
public:
    explicit CacheReclaimSubmitEvent(const std::string &source)
        : BaseEvent(source, "cache_reclaimer", "CacheReclaimSubmit") {}

    void SetAdditionalArgs(const std::vector<std::int64_t> &block_keys,
                           const std::vector<std::vector<std::string>> &location_ids,
                           std::uint64_t delay_us) {
        block_keys_ = block_keys;
        location_ids_ = location_ids;
        delay_us_ = delay_us;
    }

    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {
        BaseEvent::ToRapidWriter(writer);
        Put(writer, "block_keys", block_keys_);
        Put(writer, "location_ids", location_ids_);
        Put(writer, "delay_us", delay_us_);
    }

private:
    std::vector<std::int64_t> block_keys_;
    std::vector<std::vector<std::string>> location_ids_;
    std::uint64_t delay_us_{0};
};

} // namespace kv_cache_manager
