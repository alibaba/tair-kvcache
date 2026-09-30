#include "kv_cache_manager/client/src/internal/sdk/lock_free_thread_pool.h"

#include <autil/LockFreeThreadPool.h>

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

LockFreeThreadPool::LockFreeThreadPool(size_t thread_num,
                                       size_t queue_size,
                                       const std::string &thread_name,
                                       bool stop_if_has_exception)
    : pool_(new autil::LockFreeThreadPool(thread_num, queue_size, nullptr, thread_name, stop_if_has_exception)) {}
LockFreeThreadPool::~LockFreeThreadPool() { pool_.reset(); }

bool LockFreeThreadPool::start(const std::string &name) {
    if (!pool_->start(name)) {
        KVCM_LOG_WARN("start lock free thread pool %s failed", name.c_str());
        return false;
    }
    return true;
}
void LockFreeThreadPool::stop() { pool_->stop(); }
void LockFreeThreadPool::waitFinish() { pool_->waitFinish(); }
bool LockFreeThreadPool::isFull() const { return pool_->isFull(); }
std::future<ClientErrorCode> LockFreeThreadPool::async(std::function<ClientErrorCode()> &&func) {
    return pool_->async(func);
}

} // namespace kv_cache_manager
