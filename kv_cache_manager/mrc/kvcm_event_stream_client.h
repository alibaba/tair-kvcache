#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "kv_cache_manager/common/service_discovery.h"
#include "kv_cache_manager/mrc/online_mrc_config.h"
#include "kv_cache_manager/protocol/protobuf/optimizer_service.pb.h"

namespace kv_cache_manager {

class MetricsRegistry;
class OnlineMrcFactRegistry;
class ServiceDiscovery;

// Optimizer-side client: discovers every KVCM node, actively connects to each
// one, sends HELLO, and then only reads CacheEventBatch frames. KVCM never
// needs to discover or configure an optimizer endpoint in this mode.
class KvcmEventStreamClient {
public:
    KvcmEventStreamClient(const OnlineMrcConfig &config,
                          std::shared_ptr<OnlineMrcFactRegistry> fact_registry,
                          std::shared_ptr<MetricsRegistry> metrics_registry);
    ~KvcmEventStreamClient();

    bool Init();
    bool Start();
    void Stop();
    void ReportMetrics();

    size_t ConnectionCount() const;

private:
    class Connection;

    struct QueuedBatch {
        proto::optimizer::CacheEventBatch batch;
        uint64_t wire_bytes = 0;
    };

    void DiscoveryLoop();
    void UpdateDesiredEndpoints(const std::vector<ServiceEndpoint> &endpoints);
    void EventLoop();
    void ApplyDesiredEndpoints();
    void WakeEventLoop();
    void DrainWakePipe();
    void BeginConnect(Connection &connection);
    void HandleConnectionEvent(Connection &connection, short revents);
    bool FinishConnect(Connection &connection);
    bool FlushHello(Connection &connection);
    bool ReadAvailableFrames(Connection &connection);
    bool DecodeBufferedFrames(Connection &connection);
    void Disconnect(Connection &connection, bool schedule_retry);
    int ComputePollTimeoutMs() const;
    bool Enqueue(proto::optimizer::CacheEventBatch batch, uint64_t wire_bytes);
    void ConsumerLoop();

    OnlineMrcConfig config_;
    std::shared_ptr<OnlineMrcFactRegistry> fact_registry_;
    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::unique_ptr<ServiceDiscovery> discovery_;
    std::string optimizer_id_;

    // desired_endpoints_ is published by the discovery thread. The event-loop
    // thread is the sole owner of connections_ and every socket state change.
    mutable std::mutex desired_endpoints_mutex_;
    std::unordered_map<std::string, ServiceEndpoint> desired_endpoints_;
    uint64_t desired_endpoints_generation_ = 0;
    uint64_t applied_endpoints_generation_ = 0;
    std::unordered_map<std::string, std::unique_ptr<Connection>> connections_;
    std::thread discovery_thread_;
    std::thread event_loop_thread_;
    std::thread consumer_thread_;
    int wake_read_fd_ = -1;
    int wake_write_fd_ = -1;
    mutable std::mutex wait_mutex_;
    std::condition_variable wait_cv_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::deque<QueuedBatch> queue_;
    // Sum of serialized on-wire frame bytes waiting in queue_. This is a
    // stable pressure signal, not an allocator/RSS measurement.
    uint64_t queue_bytes_ = 0;
    std::atomic<bool> running_{false};
    std::atomic<bool> ingress_stopped_{true};

    std::atomic<uint64_t> discovered_endpoints_{0};
    std::atomic<uint64_t> managed_connections_{0};
    std::atomic<uint64_t> active_connections_{0};
    std::atomic<uint64_t> reconnects_{0};
    std::atomic<uint64_t> decode_errors_{0};
    std::atomic<uint64_t> received_batches_{0};
    std::atomic<uint64_t> dropped_batches_{0};
    std::atomic<uint64_t> rejected_batches_{0};
};

} // namespace kv_cache_manager
