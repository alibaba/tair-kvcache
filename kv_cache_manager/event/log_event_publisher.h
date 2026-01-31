#pragma once

#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "kv_cache_manager/event/event_publisher.h"

namespace kv_cache_manager {
class BaseEvent;
class CacheGetEvent;
class StartWriteCacheEvent;
class FinishWriteCacheEvent;
} // namespace kv_cache_manager

namespace kv_cache_manager {

class LogEventPublisher : public EventPublisher {
public:
    LogEventPublisher();
    ~LogEventPublisher() override;

    bool Init(const std::string &config) override;
    bool Publish(const std::shared_ptr<BaseEvent> &event) override;
    bool Stop() override;

private:
    void WorkerThread();
    void WriteEventToFile(const std::shared_ptr<BaseEvent> &event);
    std::string FormatEvent(const std::shared_ptr<BaseEvent> &event) const;

private:
    std::thread worker_;
};

} // namespace kv_cache_manager