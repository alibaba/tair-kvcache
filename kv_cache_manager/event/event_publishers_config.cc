#include "kv_cache_manager/event/event_publishers_config.h"

namespace kv_cache_manager {

bool LogEventPublisherConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "enable", enable_, true);
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "queue_size", queue_size_, std::size_t(10000));
    return queue_size_ > 0;
}

void LogEventPublisherConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "enable", enable_);
    Put(writer, "queue_size", queue_size_);
}

bool OptimizerEventPublisherConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "enable", enable_, false);
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "queue_size", queue_size_, std::size_t(100000));
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "max_subscribers", max_subscribers_, std::size_t(4));
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "subscriber_queue_size", subscriber_queue_size_, std::size_t(10000));
    return queue_size_ > 0 && max_subscribers_ > 0 && subscriber_queue_size_ > 0;
}

void OptimizerEventPublisherConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "enable", enable_);
    Put(writer, "queue_size", queue_size_);
    Put(writer, "max_subscribers", max_subscribers_);
    Put(writer, "subscriber_queue_size", subscriber_queue_size_);
}

bool EventPublishersConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    log_ = LogEventPublisherConfig{};
    optimizer_ = OptimizerEventPublisherConfig{};
    KVCM_JSON_GET_MACRO(rapid_value, "log", log_);
    KVCM_JSON_GET_MACRO(rapid_value, "optimizer", optimizer_);
    return true;
}

void EventPublishersConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "log", log_);
    Put(writer, "optimizer", optimizer_);
}

} // namespace kv_cache_manager
