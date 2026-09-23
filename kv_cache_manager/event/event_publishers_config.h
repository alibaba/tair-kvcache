#pragma once

#include <cstddef>

#include "kv_cache_manager/common/jsonizable.h"

namespace kv_cache_manager {

class LogEventPublisherConfig : public Jsonizable {
public:
    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

    bool enable() const { return enable_; }
    std::size_t queue_size() const { return queue_size_; }

private:
    bool enable_ = true;
    std::size_t queue_size_ = 10000;
};

class OptimizerEventPublisherConfig : public Jsonizable {
public:
    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

    bool enable() const { return enable_; }
    std::size_t queue_size() const { return queue_size_; }
    std::size_t max_subscribers() const { return max_subscribers_; }
    std::size_t subscriber_queue_size() const { return subscriber_queue_size_; }

private:
    bool enable_ = false;
    std::size_t queue_size_ = 100000;
    std::size_t max_subscribers_ = 4;
    std::size_t subscriber_queue_size_ = 10000;
};

class EventPublishersConfig : public Jsonizable {
public:
    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

    bool enable_log_event_publisher() const { return log_.enable(); }
    const LogEventPublisherConfig &log_event_publisher_config() const { return log_; }
    bool enable_optimizer_event_publisher() const { return optimizer_.enable(); }
    const OptimizerEventPublisherConfig &optimizer_event_publisher_config() const { return optimizer_; }

private:
    LogEventPublisherConfig log_;
    OptimizerEventPublisherConfig optimizer_;
};

} // namespace kv_cache_manager
