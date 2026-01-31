#pragma once

#include "kv_cache_manager/common/jsonizable.h"

namespace kv_cache_manager {

class MetaChannelConfig : public Jsonizable {
public:
    bool FromRapidValue(const rapidjson::Value &rapid_value) override {
        KVCM_JSON_GET_MACRO(rapid_value, "retry_time", retry_time_);
        KVCM_JSON_GET_MACRO(rapid_value, "connection_timeout", connection_timeout_);
        KVCM_JSON_GET_MACRO(rapid_value, "call_timeout", call_timeout_);
        return true;
    }

    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {
        Put(writer, "retry_time", retry_time_);
        Put(writer, "connection_timeout", connection_timeout_);
        Put(writer, "call_timeout", call_timeout_);
    }

    inline uint32_t retry_time() const { return retry_time_; }
    inline uint32_t connection_timeout() const { return connection_timeout_; }
    inline uint32_t call_timeout() const { return call_timeout_; }

private:
    uint32_t retry_time_ = 3;
    uint32_t connection_timeout_ = 1000; // ms
    uint32_t call_timeout_ = 100;        // ms
};

} // namespace kv_cache_manager