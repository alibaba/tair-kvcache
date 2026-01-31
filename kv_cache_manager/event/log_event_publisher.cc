#include "kv_cache_manager/event/log_event_publisher.h"

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/manager/cache_location.h"

namespace kv_cache_manager {

LogEventPublisher::LogEventPublisher() = default;

LogEventPublisher::~LogEventPublisher() {
    if (running_) {
        Stop();
    }
}
// 先假设这里传的就是log文件路径
bool LogEventPublisher::Init(const std::string &config) {
    // 初始化基础队列，这里的队列长度可以通过配置传入
    InitBasicQueue();

    running_ = true;

    worker_ = std::thread(&LogEventPublisher::WorkerThread, this);

    return true;
}

bool LogEventPublisher::Publish(const std::shared_ptr<BaseEvent> &event) {
    if (!event) {
        return false;
    }

    if (!running_) {
        return false;
    }
    if (!BasicEnqueue(event)) {
        KVCM_LOG_WARN("Event queue full, dropping event");
        return false;
    }
    KVCM_LOG_DEBUG("Event enqueued successfully");
    return true;
}

bool LogEventPublisher::Stop() {
    if (!running_) {
        return true;
    }
    running_ = false;
    ClearBasicQueue();
    if (basic_queue_) {
        basic_queue_->queue_cv.notify_all();
    }

    if (worker_.joinable()) {
        worker_.join();
    }

    return true;
}

void LogEventPublisher::WorkerThread() {
    while (running_) {
        BasicWait();

        std::shared_ptr<BaseEvent> process_event;
        while (BasicDequeue(process_event)) {
            WriteEventToFile(process_event);
        }
    }
}

void LogEventPublisher::WriteEventToFile(const std::shared_ptr<BaseEvent> &event) {
    if (!event) {
        return;
    }

    KVCM_PUBLISHER_LOG(FormatEvent(event));
}

std::string LogEventPublisher::FormatEvent(const std::shared_ptr<BaseEvent> &event) const {
    if (!event) {
        return "{}";
    }
    return event->ToJsonString();
}

} // namespace kv_cache_manager