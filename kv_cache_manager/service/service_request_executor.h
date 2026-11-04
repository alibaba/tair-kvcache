#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace kv_cache_manager {

// Shared FIFO for Meta HTTP and gRPC business calls. No task runs on a
// submitting thread, and capacity limits waiting tasks only (zero is unbounded).
class ServiceRequestExecutor {
private:
    struct State;
    struct CompletionLease;

public:
    // An accepted task owns one completion lease. Adapters may retain copies
    // past the callback return, e.g. until the HTTP handler resumes on its IO
    // thread. Drain waits until the last copy has been released.
    class CompletionToken {
    public:
        CompletionToken() = default;

    private:
        friend class ServiceRequestExecutor;
        explicit CompletionToken(std::shared_ptr<CompletionLease> lease) : lease_(std::move(lease)) {}
        std::shared_ptr<CompletionLease> lease_;
    };

    struct Task {
        // Both callbacks must complete their request, including exception
        // handling. Exactly one is invoked for each accepted task.
        std::function<void(CompletionToken)> run;
        std::function<void(CompletionToken)> on_discard;
    };

    enum class SubmitResult {
        kAccepted,
        kFull,
        kClosed
    };
    enum class ShutdownMode {
        kDiscardPending,
        kDrain
    };

    struct Snapshot {
        std::size_t queued = 0;
        std::size_t running = 0;
        std::size_t discarding = 0;
        std::size_t completion_tokens = 0;
        std::size_t rejected = 0;
        std::size_t discarded = 0;
    };

    explicit ServiceRequestExecutor(std::size_t worker_count, std::size_t queue_capacity = 0);
    ~ServiceRequestExecutor();

    ServiceRequestExecutor(const ServiceRequestExecutor &) = delete;
    ServiceRequestExecutor &operator=(const ServiceRequestExecutor &) = delete;

    // Never waits for space. A rejected task invokes neither callback. Memory
    // allocation and invalid callbacks may throw before accepting the task.
    SubmitResult TrySubmit(Task task);

    // First close wins. Pending discard callbacks run on the calling shutdown
    // coordinator, outside the queue mutex; already-dequeued tasks keep running.
    void CloseAdmission(ShutdownMode mode = ShutdownMode::kDiscardPending);

    // Call these on the shutdown coordinator, never a worker, a discard
    // callback, or a thread whose progress is needed to release a token.
    // Drain does not close admission. Join closes it if necessary and joins
    // workers, but does not replace Drain's wait for external completion tokens.
    void Drain();
    void Join();

    Snapshot GetSnapshot() const;

private:
    static void WorkerLoop(const std::shared_ptr<State> &state);

    std::shared_ptr<State> state_;
    std::mutex join_mutex_;
    std::vector<std::thread> workers_;
};

} // namespace kv_cache_manager
