#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

#include "kv_cache_manager/event/event_publisher.h"
#include "kv_cache_manager/mrc/online_mrc_config.h"
#include "kv_cache_manager/protocol/protobuf/optimizer_service.grpc.pb.h"

namespace kv_cache_manager {

class LoopThread;
class MetricsRegistry;

struct OptimizerTraceInstanceMetadata {
    std::string instance_group;
    int32_t block_size = 0;
    int64_t bytes_per_block = 0;
};

// KVCM-side best-effort forwarder. Publish performs only filtering and a
// bounded queue insertion; serialization, batching and RPC run on its worker.
class OptimizerTraceForwarder : public EventPublisher {
public:
    using MetadataResolver =
        std::function<OptimizerTraceInstanceMetadata(const std::string &instance_id)>;

    OptimizerTraceForwarder(const OptimizerTraceForwarderConfig &config,
                            std::shared_ptr<MetricsRegistry> metrics_registry,
                            MetadataResolver metadata_resolver);
    ~OptimizerTraceForwarder() override;

    bool Init(const std::string &config) override;
    bool Publish(const std::shared_ptr<BaseEvent> &event) override;
    bool Stop() override;

private:
    void WorkerLoop();
    bool Send(proto::optimizer::ReportAccessTraceRequest &request);
    void ReportMetrics();

    OptimizerTraceForwarderConfig config_;
    std::shared_ptr<MetricsRegistry> metrics_registry_;
    MetadataResolver metadata_resolver_;
    std::unique_ptr<proto::optimizer::OptimizerService::Stub> stub_;
    std::thread worker_;
    std::shared_ptr<LoopThread> report_thread_;
    std::string source_id_;
    std::unordered_map<std::string, uint64_t> next_sequence_by_instance_;
    std::unordered_map<std::string, OptimizerTraceInstanceMetadata> metadata_by_instance_;
    std::mutex backoff_mutex_;
    std::condition_variable backoff_cv_;
    int64_t consecutive_send_failures_ = 0;

    std::atomic<uint64_t> dropped_keys_{0};
    std::atomic<uint64_t> dropped_spans_{0};
    std::atomic<uint64_t> sent_spans_{0};
    std::atomic<uint64_t> sent_keys_{0};
    std::atomic<uint64_t> send_failures_{0};
    std::atomic<uint64_t> filtered_spans_{0};
    std::atomic<uint64_t> filtered_keys_{0};
    uint64_t acknowledged_dropped_spans_ = 0;
    uint64_t acknowledged_dropped_keys_ = 0;
};

} // namespace kv_cache_manager
