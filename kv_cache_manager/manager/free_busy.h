#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "kv_cache_manager/manager/delete_limits.h"
#include "kv_cache_manager/meta/cache_location.h"

namespace kv_cache_manager {

class MetricsRegistry;

// Shared by managed Free tasks in this executor, including delayed callbacks.
struct FreeBusyResources {
    explicit FreeBusyResources(uint64_t limit, std::shared_ptr<MetricsRegistry> metrics = {})
        : memory_limit(limit), metrics(std::move(metrics)) {}
    const uint64_t memory_limit;
    const std::shared_ptr<MetricsRegistry> metrics;
    std::atomic<uint64_t> reserved{0};
    std::atomic<size_t> executing_retries{0};
    std::atomic<uint64_t> generation{0};
    std::atomic<uint64_t> retry_attempts{0}, busy_replies{0};

    bool Reserve(uint64_t bytes) {
        auto used = reserved.load();
        while (bytes <= memory_limit - used) {
            if (reserved.compare_exchange_weak(used, used + bytes)) {
                return true;
            }
        }
        return false;
    }
    bool TryStartRetry(size_t workers) {
        const size_t limit = std::max<size_t>(1, workers / 2);
        auto count = executing_retries.load();
        while (count < limit) {
            if (executing_retries.compare_exchange_weak(count, count + 1)) {
                return true;
            }
        }
        return false;
    }
};

struct FreeBusyTask {
    using Clock = std::chrono::steady_clock;
    FreeBusyTask(std::shared_ptr<FreeBusyResources> resources,
                 uint64_t initial_ms,
                 uint32_t max_ms = DeleteLimits::kDefaultMaxBackoffMs)
        : resources(std::move(resources))
        , generation(this->resources->generation.load())
        , initial_ms(std::min<uint64_t>(initial_ms, max_ms))
        , max_ms(max_ms) {}
    ~FreeBusyTask() { ReleaseUnused(); }
    bool Active() const { return generation == resources->generation.load(); }

    // One initial worker divides the admission reservation among storage batches.
    bool Take(uint64_t bytes) {
        if (bytes > unassigned && !resources->Reserve(bytes - unassigned)) {
            return false;
        }
        unassigned -= std::min(bytes, unassigned);
        return true;
    }
    void ReleaseUnused() {
        resources->reserved.fetch_sub(unassigned);
        unassigned = 0;
    }
    // One individually allocated record per Location. Allow for the list node,
    // batch/continuation and allocator overhead even in a one-Location batch.
    static uint64_t IdBytes(const std::string &id, size_t instance_size, size_t storage_size) {
        return 1024 + 2 * (id.size() + instance_size + storage_size);
    }
    static uint64_t RecordBytes(const CacheLocation &loc, size_t instance_size) {
        // A batch also retains the storage name from the first URI.
        const auto &specs = loc.location_specs();
        uint64_t bytes = IdBytes(loc.id(), instance_size, specs.empty() ? 0 : specs.front().uri().size());
        for (const auto &spec : loc.location_specs()) {
            bytes += sizeof(std::string) + 32 + 2 * spec.uri().size();
        }
        return bytes;
    }
    // Installed before submission. Feedback is applied to the original admission.
    std::function<void(const std::string &, uint64_t)> on_feedback;
    void Record(const std::string &storage, uint64_t busy_ms) const {
        if (on_feedback) {
            on_feedback(storage, busy_ms);
        }
    }

    const std::shared_ptr<FreeBusyResources> resources;
    const uint64_t generation;
    const uint64_t initial_ms;
    const uint64_t max_ms;
    uint64_t unassigned{0};

};

} // namespace kv_cache_manager
