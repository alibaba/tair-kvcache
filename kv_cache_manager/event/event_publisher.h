#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include "kv_cache_manager/event/base_event.h"

namespace kv_cache_manager {

class EventPublisher {
public:
    virtual ~EventPublisher() = default;

    /**
     * @brief 初始化事件发布器
     * @param config 配置信息
     * @return 是否成功
     */
    virtual bool Init(const std::string &config) = 0;

    /**
     * @brief 异步发布事件
     * @param event 待发布的事件
     * @return 是否成功
     */
    virtual bool Publish(const std::shared_ptr<BaseEvent> &event) = 0;

    /**
     * @brief 关闭事件发布器
     * @return 是否成功
     */
    virtual bool Stop() = 0;

    void set_name(const std::string &publisher_name) { name_ = publisher_name; }
    std::string &name() { return name_; }

protected:
    EventPublisher() = default;

    // 基础队列组件结构
    struct BasicQueueComponents {
        std::queue<std::shared_ptr<BaseEvent>> event_queue;
        mutable std::mutex queue_mutex;
        std::condition_variable queue_cv;
        size_t max_queue_size{10000};
        std::atomic<size_t> dropped_counts{0};
        std::atomic<size_t> queue_size{0};
    };

    // 初始化基础队列（子类可选择调用）
    void InitBasicQueue(size_t max_size = 10000) {
        if (!basic_queue_) {
            basic_queue_ = std::make_unique<BasicQueueComponents>();
            basic_queue_->max_queue_size = max_size;
        }
    }

    // 基础队列操作（子类可选择使用）
    bool BasicEnqueue(const std::shared_ptr<BaseEvent> &event) {
        if (!basic_queue_ || !event) {
            return false;
        }

        {
            std::lock_guard<std::mutex> lock(basic_queue_->queue_mutex);
            if (basic_queue_->queue_size.load(std::memory_order_relaxed) >= basic_queue_->max_queue_size) {
                basic_queue_->dropped_counts.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            basic_queue_->event_queue.push(event);
            basic_queue_->queue_size.fetch_add(1, std::memory_order_relaxed);
        }

        basic_queue_->queue_cv.notify_one();
        return true;
    }

    bool BasicDequeue(std::shared_ptr<BaseEvent> &event) {
        if (!basic_queue_) {
            return false;
        }

        std::lock_guard<std::mutex> lock(basic_queue_->queue_mutex);
        if (basic_queue_->event_queue.empty()) {
            return false;
        }
        event = basic_queue_->event_queue.front();
        basic_queue_->event_queue.pop();
        basic_queue_->queue_size.fetch_sub(1, std::memory_order_relaxed);
        return true;
    }

    void BasicWait() {
        if (!basic_queue_) {
            return;
        }

        std::unique_lock<std::mutex> lock(basic_queue_->queue_mutex);
        basic_queue_->queue_cv.wait(
            lock, [this] { return (basic_queue_ && !basic_queue_->event_queue.empty()) || !running_; });
    }

    size_t BasicQueueSize() const {
        if (!basic_queue_) {
            return 0;
        }
        return basic_queue_->queue_size.load(std::memory_order_relaxed);
    }

    size_t BasicDroppedCount() const {
        if (!basic_queue_) {
            return 0;
        }
        return basic_queue_->dropped_counts.load(std::memory_order_relaxed);
    }

    // 清空队列
    void ClearBasicQueue() {
        if (!basic_queue_) {
            return;
        }

        std::lock_guard<std::mutex> lock(basic_queue_->queue_mutex);
        while (!basic_queue_->event_queue.empty()) {
            basic_queue_->event_queue.pop();
        }
        basic_queue_->queue_size.store(0, std::memory_order_relaxed);
    }

protected:
    // 基础队列组件（可选使用）
    std::unique_ptr<BasicQueueComponents> basic_queue_;
    std::string name_;
    // 运行状态
    std::atomic<bool> running_{false};
};

} // namespace kv_cache_manager