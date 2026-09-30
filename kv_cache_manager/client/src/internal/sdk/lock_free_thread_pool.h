#pragma once

#include <future>
#include <memory>

#include "kv_cache_manager/client/include/common.h"

namespace autil {
class LockFreeThreadPool;
}

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
    std::future<ClientErrorCode> async(std::function<ClientErrorCode()> &&func);

private:
    std::unique_ptr<autil::LockFreeThreadPool> pool_;
};

} // namespace kv_cache_manager