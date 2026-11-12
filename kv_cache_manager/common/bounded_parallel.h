#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <thread>
#include <vector>

namespace kv_cache_manager {

// Each call uses independent workers; the caller also executes tasks.
// Failure stops new work on a best-effort basis; started work finishes before return.
template <typename Function>
bool RunBoundedParallel(size_t count, size_t concurrency, Function &&function) noexcept {
    if (count == 0) {
        return true;
    }
    if (concurrency == 0) {
        return false;
    }

    std::atomic<size_t> next{0};
    std::atomic<bool> failed{false};
    const auto work = [&]() noexcept {
        while (!failed.load(std::memory_order_acquire)) {
            const size_t index = next.fetch_add(1, std::memory_order_relaxed);
            if (index >= count) {
                return;
            }
            try {
                if (!function(index)) {
                    failed.store(true, std::memory_order_release);
                }
            } catch (...) {
                failed.store(true, std::memory_order_release);
            }
        }
    };

    std::vector<std::thread> workers;
    try {
        const size_t worker_count = std::min(count, concurrency);
        workers.reserve(worker_count - 1);
        for (size_t i = 1; i < worker_count; ++i) {
            workers.emplace_back(work);
        }
        work();
    } catch (...) {
        // Join workers even if thread creation fails partway through.
        failed.store(true, std::memory_order_release);
    }
    for (auto &worker : workers) {
        worker.join();
    }
    return !failed.load(std::memory_order_acquire);
}

} // namespace kv_cache_manager
