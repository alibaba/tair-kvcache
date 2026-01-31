#pragma once

#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <string>

#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/common/string_util.h"

namespace kv_cache_manager {
class BaseEvent : public Jsonizable {
public:
    BaseEvent(const std::string &source, const std::string &component, const std::string &type)
        : source_(source), component_(component), type_(type) {
        // TODO plus KVCacheManager ip
        id_ = StringUtil::GenerateRandomString(32);
    };
    virtual ~BaseEvent() = default;
    BaseEvent() = delete;
    const std::string &event_id() const { return id_; }
    const std::string &event_source() const { return source_; }
    const std::string &event_component() const { return component_; }
    const std::string &event_type() const { return type_; }
    int64_t event_trigger_time_us() const { return trigger_time_us_; }

    bool FromRapidValue(const rapidjson::Value &rapid_value) override { return false; }

    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {
        Put(writer, "id", id_);
        Put(writer, "source", source_);
        Put(writer, "component", component_);
        Put(writer, "type", type_);
        Put(writer, "trigger_time_us", trigger_time_us_);
    }

    void SetEventTriggerTime() {
        trigger_time_us_ =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch())
                .count();
    }

protected:
    std::string id_;             // 事件唯一ID
    std::string source_;         // 事件来源实例
    std::string component_;      // 事件所属组件
    std::string type_;           // 事件类型
    int64_t trigger_time_us_{0}; // 微秒
};

} // namespace kv_cache_manager