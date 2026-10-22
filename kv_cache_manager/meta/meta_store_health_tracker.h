#pragma once

#include <atomic>
#include <cstdint>
#include <string>

namespace kv_cache_manager {

// Tracks whether the remote store behind a persistent MetaStorageBackend (for
// example the Redis instance holding cache-location metadata) can still durably
// accept writes, and detects when it comes back.
//
// Why this exists: in memory_primary mode the local backend is authoritative and
// the remote store is only a best-effort backup, so a remote outage is silent --
// every request keeps returning EC_OK while the flushed writes are dropped. The
// metadata that must survive a restart (a GA address allocated by KVCM, or the
// deletion of one) is exactly the metadata that cannot be reconstructed later,
// so callers need an explicit signal to stop allocating and stop deleting until
// the remote store is reachable again.
//
// That signal has to describe the store *now*, not merely "no failure seen yet".
// A backend only learns from the operations it actually issues, so an instance
// that happens to be idle when the store goes down keeps serving the last
// known-good verdict forever: the allocation gate stays open and the whole burst
// that follows gets locations whose backup is then silently dropped. Backends
// that own a background thread therefore poll the store every `probe_interval_ms`
// regardless of the current verdict, and a failed poll is conclusive -- see
// RecordProbeFailure().
//
// The state machine is deliberately asymmetric: a single success clears the
// degradation immediately, while entering it requires `failure_threshold`
// consecutive *operation* failures so one transient command error does not stall
// the write path.
// All operations are lock-free and safe to call from every consumer thread.
class MetaStoreHealthTracker {
public:
    struct Config {
        // Consecutive failed store operations required to report unavailable.
        // Probes bypass it, a failed probe degrades on its own.
        int64_t failure_threshold = 3;
        // Minimum spacing between active probes. Doubles as the heartbeat period
        // while healthy, so it bounds how long an outage can go unnoticed by a
        // backend that is not issuing operations right now.
        int64_t probe_interval_ms = 1000;
    };

    // Note: no default argument for `config` -- a nested class with default
    // member initializers is still incomplete while the enclosing class body is
    // being parsed, so `Config{}` cannot appear in a default argument here.
    MetaStoreHealthTracker() { Configure(std::string(), Config()); }
    MetaStoreHealthTracker(std::string name, Config config);

    // Re-applies identity and thresholds. Backends own their tracker by value
    // (std::atomic is neither copyable nor movable) and only learn the instance
    // id and URI parameters during Init, so configuration is a separate step.
    // Must be called before the backend starts serving.
    void Configure(std::string name, Config config) noexcept;

    // Records the outcome of a real store operation (write flush or read), and of
    // a successful probe.
    void RecordSuccess() noexcept;
    void RecordFailure() noexcept;

    // Records a failed active probe. Unlike a pipelined write, a probe carries no
    // command level ambiguity -- it is an EXISTS on our own metadata key, issued
    // only because there was no recent verdict -- so it degrades immediately
    // instead of waiting for `failure_threshold` hits. Counting it like an
    // operation would multiply the detection latency by the threshold, and the
    // point of the heartbeat is to close the gate before the next allocation
    // rather than a few allocations later.
    void RecordProbeFailure() noexcept;

    // True when the caller should issue an active probe, at most once per probe
    // interval across all threads. Deliberately not conditioned on the current
    // verdict: while degraded this is the recovery poll, while healthy it is the
    // heartbeat that keeps an idle backend from going stale. One EXISTS per
    // interval is the entire cost.
    bool ShouldProbe() noexcept;

    bool IsAvailable() const noexcept { return available_.load(std::memory_order_acquire); }
    int64_t ConsecutiveFailures() const noexcept { return consecutive_failures_.load(std::memory_order_relaxed); }
    const std::string &name() const noexcept { return name_; }

private:
    void ReportTransition(bool available) noexcept;

    std::string name_;
    Config config_{};
    std::atomic<bool> available_{true};
    std::atomic<int64_t> consecutive_failures_{0};
    std::atomic<int64_t> last_probe_time_ms_{0};
};

} // namespace kv_cache_manager
