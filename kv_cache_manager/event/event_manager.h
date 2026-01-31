#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <shared_mutex>
#include <string>
#include <vector>

#include "kv_cache_manager/event/base_event.h"
#include "kv_cache_manager/event/event_publisher.h"

namespace kv_cache_manager {

class EventManager {
public:
    EventManager();
    ~EventManager();

    // 系统管理
    bool Init();
    bool Stop();

    // 发布器管理
    bool RegisterPublisher(const std::string &unique_name, std::shared_ptr<EventPublisher> publisher);
    bool RemovePublisher(const std::string &unique_name);
    std::shared_ptr<EventPublisher> GetPublisher(const std::string &unique_name) const;
    bool HasPublisher(const std::string &unique_name) const;
    std::vector<std::string> ListPublishers() const;
    void ClearPublishers();
    // 事件推送
    bool Publish(const std::shared_ptr<BaseEvent> &event);
    // 状态和统计
    size_t GetPublisherCount() const;

private:
    // 禁止拷贝
    EventManager(const EventManager &) = delete;
    EventManager &operator=(const EventManager &) = delete;

private:
    // 发布器管理
    std::map<std::string, std::shared_ptr<EventPublisher>> registered_publishers_;
    mutable std::shared_mutex publishers_mutex_; // 读写锁支持动态注册

    std::atomic<bool> initialized_{false};
};
} // namespace kv_cache_manager