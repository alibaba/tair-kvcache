#pragma once

#include <algorithm>
#include <atomic>
#include <autil/Thread.h>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "kv_cache_manager/common/error_code.h"

namespace kv_cache_manager {

// Each call uses independent workers; the caller also executes tasks.
// Failure stops new work on a best-effort basis; started work finishes before return.
template <typename Function>
ErrorCode RunBoundedParallel(size_t count, size_t concurrency, Function &&function) noexcept {
    if (count == 0) {
        return EC_OK;
    }
    if (concurrency == 0) {
        return EC_BADARGS;
    }

    std::atomic<size_t> next{0};
    std::atomic<ErrorCode> error{EC_OK};
    const auto record_error = [&](ErrorCode ec) {
        auto expected = EC_OK;
        error.compare_exchange_strong(expected, ec, std::memory_order_acq_rel);
    };
    const auto work = [&]() noexcept {
        while (error.load(std::memory_order_acquire) == EC_OK) {
            const size_t index = next.fetch_add(1, std::memory_order_relaxed);
            if (index >= count) {
                return;
            }
            const auto ec = function(index);
            if (ec != EC_OK) {
                record_error(ec);
            }
        }
    };

    std::vector<autil::ThreadPtr> workers;
    const size_t worker_count = std::min(count, concurrency);
    workers.reserve(worker_count - 1);
    for (size_t i = 1; i < worker_count; ++i) {
        auto worker = autil::Thread::createThread(work, "kvcm_parallel");
        if (!worker) {
            record_error(EC_ERROR);
            break;
        }
        workers.push_back(std::move(worker));
    }
    work();
    for (auto &worker : workers) {
        worker->join();
    }
    return error.load(std::memory_order_acquire);
}

} // namespace kv_cache_manager
