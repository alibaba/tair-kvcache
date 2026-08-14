#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace kv_cache_manager {

// Process-local, bounded executor for latency-sensitive metadata queries.
//
// The calling RPC thread always participates in ParallelFor. worker_count is
// therefore the maximum parallelism of one query, not the number of background
// threads: an executor configured with N workers owns N - 1 threads. A bounded
// queue prevents concurrent large requests from creating unbounded work; when
// admission fails, the caller and any already-admitted workers finish the
// remaining chunks themselves.
class QueryExecutor {
public:
    using RangeFunction = std::function<void(std::size_t begin, std::size_t end)>;

    QueryExecutor(std::size_t worker_count,
                  std::size_t parallel_threshold,
                  std::size_t chunk_size,
                  std::size_t queue_capacity);
    ~QueryExecutor();

    QueryExecutor(const QueryExecutor &) = delete;
    QueryExecutor &operator=(const QueryExecutor &) = delete;

    // Runs fn over disjoint half-open ranges that cover [0, count). Returns
    // false if a callback throws or the parallel work cannot be allocated or
    // scheduled. Calls made recursively from this executor's own worker thread
    // deliberately fall back to serial execution so a task can never wait for
    // the pool that is currently running it.
    bool ParallelFor(std::size_t count, const RangeFunction &fn) const noexcept;
    // Same bounded executor with a per-call range size. Large local metadata
    // scans use this to amortize shard locking without changing the global
    // projection chunk configured for other query work.
    bool ParallelForWithChunkSize(std::size_t count, std::size_t chunk_size, const RangeFunction &fn) const noexcept;

    [[nodiscard]] std::size_t worker_count() const noexcept { return worker_count_; }
    [[nodiscard]] std::size_t parallel_threshold() const noexcept { return parallel_threshold_; }
    [[nodiscard]] std::size_t chunk_size() const noexcept { return chunk_size_; }

private:
    bool ParallelForImpl(std::size_t count, std::size_t chunk_size, const RangeFunction &fn) const noexcept;
    bool TrySubmit(std::function<void()> task) const;
    void WorkerLoop();

private:
    std::size_t worker_count_ = 1;
    std::size_t parallel_threshold_ = 1;
    std::size_t chunk_size_ = 1;
    std::size_t queue_capacity_ = 1;

    mutable std::mutex mutex_;
    mutable std::condition_variable condition_;
    mutable std::deque<std::function<void()>> tasks_;
    mutable bool stopping_ = false;
    std::vector<std::thread> workers_;
};

} // namespace kv_cache_manager
