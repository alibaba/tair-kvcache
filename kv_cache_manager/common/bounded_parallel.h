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

namespace detail {

// Each call uses independent workers; the caller also executes tasks.
// Failure stops new work on a best-effort basis; started work finishes before return.
template <typename Function, typename ThreadFactory>
ErrorCode RunBoundedParallelWithFactory(size_t count,
                                        size_t concurrency,
                                        Function &&function,
                                        ThreadFactory &&create_thread) noexcept {
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
            try {
                const auto ec = function(index);
                if (ec != EC_OK) {
                    record_error(ec);
                }
            } catch (...) {
                record_error(EC_ERROR);
            }
        }
    };

    std::vector<autil::ThreadPtr> workers;
    try {
        const size_t worker_count = std::min(count, concurrency);
        workers.reserve(worker_count - 1);
        for (size_t i = 1; i < worker_count; ++i) {
            auto worker = create_thread(work);
            if (!worker) {
                record_error(EC_ERROR);
                break;
            }
            workers.push_back(std::move(worker));
        }
        work();
    } catch (...) {
        record_error(EC_ERROR);
    }
    for (auto &worker : workers) {
        worker->join();
    }
    return error.load(std::memory_order_acquire);
}

} // namespace detail

template <typename Function>
ErrorCode RunBoundedParallel(size_t count, size_t concurrency, Function &&function) noexcept {
    return detail::RunBoundedParallelWithFactory(
        count, concurrency, std::forward<Function>(function), [](const auto &work) {
            return autil::Thread::createThread(work, "kvcm_parallel");
        });
}

} // namespace kv_cache_manager
