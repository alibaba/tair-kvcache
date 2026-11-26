#pragma once

#include <autil/LockFreeThreadPool.h>
#include <memory>
#include <string>
#include <utility>

namespace kv_cache_manager {

class LockFreeThreadPool {
public:
    LockFreeThreadPool(size_t thread_num,
                       size_t queue_size,
                       const std::string &thread_name,
                       bool stop_if_has_exception = false);
    ~LockFreeThreadPool();

    bool start(const std::string &name = "");
    void stop();
    void waitFinish();
    bool isFull() const;

    template <typename Function>
    auto async(Function &&function) {
        return pool_->async(std::forward<Function>(function));
    }

private:
    std::unique_ptr<autil::LockFreeThreadPool> pool_;
};

} // namespace kv_cache_manager
