#include "kv_cache_manager/optimizer/manager/lite_hit_offline_runner.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/optimizer/config/optimizer_registry_manager.h"
#include "kv_cache_manager/optimizer/liteHit/facts_csv.h"
#include "kv_cache_manager/optimizer/liteHit/lite_hit.h"
#include "kv_cache_manager/optimizer/liteHit/request_preprocess.h"
#include "kv_cache_manager/optimizer/manager/online_runtime/online_optimizer_manager.h"
#include "kv_cache_manager/optimizer/trace_loader/optimizer_schema_trace.h"
#include "kv_cache_manager/optimizer/trace_loader/standard_trace_loader.h"
#include "kv_cache_manager/optimizer/trace_loader/trace_util.h"

namespace kv_cache_manager {

namespace {

// One per-instance replay lane. LiteHit state updates are serial inside a
// lane; preprocessing and row formatting are parallel across the batch.
struct InstanceLane {
    LiteHit core;
    uint64_t block_size_tokens = 0;
    uint64_t block_bytes = 0;
    bool enable_prefix_hash = false;
};

struct BatchItem {
    std::shared_ptr<OptimizerSchemaTrace> trace;
    InstanceLane *lane = nullptr;
    std::string instance_id;
    NormalizedRequest normalized;
    LiteHitFactRecord record;
    std::string row;
    std::string error;
};

// Runs fn(i) for every index with worker_count threads. worker_count == 1
// degenerates to a plain loop on the calling thread.
template <typename Fn>
void ParallelForIndex(std::size_t count, int32_t worker_count, const Fn &fn) {
    if (worker_count <= 1 || count <= 1) {
        for (std::size_t i = 0; i < count; ++i) {
            fn(i);
        }
        return;
    }
    const std::size_t threads = std::min<std::size_t>(static_cast<std::size_t>(worker_count), count);
    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (std::size_t t = 0; t < threads; ++t) {
        pool.emplace_back([&, t] {
            for (std::size_t i = t; i < count; i += threads) {
                fn(i);
            }
        });
    }
    for (auto &worker : pool) {
        worker.join();
    }
}

} // namespace

bool LiteHitOfflineRunner::Run() {
    // Empty registry_uri keeps everything in-memory: registration is validated
    // exactly like online but nothing is persisted.
    auto registry = std::make_shared<OptimizerRegistryManager>("");
    OnlineOptimizerManager manager(registry);

    for (const auto &group : config_.instance_groups()) {
        ErrorCode ec = manager.CreateInstanceGroup(group);
        if (ec != EC_OK) {
            KVCM_LOG_ERROR("LiteHitOfflineRunner: CreateInstanceGroup[%s] failed, ec=%d",
                           group.name().c_str(),
                           static_cast<int>(ec));
            return false;
        }
    }

    // Per-instance lanes. Registration through the manager validates the
    // config and yields size_full_only, the per-block byte charge recorded
    // into every fact row.
    std::unordered_map<std::string, std::unique_ptr<InstanceLane>> lanes;
    for (const auto &instance : config_.instances()) {
        if (instance.linear_step() != 0) {
            KVCM_LOG_ERROR("LiteHitOfflineRunner: instance[%s] has linear_step=%d; the facts replay is Full-only",
                           instance.instance_id().c_str(),
                           instance.linear_step());
            return false;
        }
        RegisterInstanceResult register_result;
        ErrorCode ec = manager.RegisterInstance(instance, register_result);
        if (ec != EC_OK) {
            KVCM_LOG_ERROR("LiteHitOfflineRunner: RegisterInstance[%s] failed, ec=%d",
                           instance.instance_id().c_str(),
                           static_cast<int>(ec));
            return false;
        }
        auto lane = std::make_unique<InstanceLane>();
        lane->block_size_tokens = static_cast<uint64_t>(instance.block_size());
        lane->block_bytes = static_cast<uint64_t>(register_result.size_full_only);
        lane->enable_prefix_hash = instance.enable_prefix_hash();
        lanes.emplace(instance.instance_id(), std::move(lane));
    }

    const std::string &override_id = config_.override_instance_id();
    if (!override_id.empty() && lanes.find(override_id) == lanes.end()) {
        KVCM_LOG_ERROR("LiteHitOfflineRunner: override_instance_id[%s] has no matching configured instance",
                       override_id.c_str());
        return false;
    }

    const std::string final_path = config_.output_result_path() + "/" + kLiteHitFactsFileName;
    const std::string temp_path = final_path + ".tmp";
    std::ofstream out(temp_path, std::ios::trunc);
    if (!out.is_open()) {
        KVCM_LOG_ERROR("LiteHitOfflineRunner: failed to open temp facts file [%s]", temp_path.c_str());
        return false;
    }
    out << kLiteHitFactsCsvHeader << '\n';

    const int32_t worker_count = std::max<int32_t>(1, config_.pipeline_worker_count());
    // Internal bounded window; not exposed as configuration.
    const std::size_t batch_capacity = static_cast<std::size_t>(worker_count) * 256;

    uint64_t processed = 0;
    int64_t last_timestamp_ns = std::numeric_limits<int64_t>::min();
    std::vector<BatchItem> batch;
    batch.reserve(batch_capacity);
    bool failed = false;
    std::string failure_reason;

    const auto fail_with = [&](std::string reason) {
        if (!failed) {
            failed = true;
            failure_reason = std::move(reason);
        }
    };

    // Batch = bounded reorder window: parallel preprocess, ordered per-lane
    // commits, parallel formatting, ordered write. Reading pauses while a
    // full batch drains, which is the backpressure.
    const auto process_batch = [&] {
        if (batch.empty() || failed) {
            batch.clear();
            return;
        }
        ParallelForIndex(batch.size(), worker_count, [&](std::size_t i) {
            BatchItem &item = batch[i];
            const auto *read_trace = static_cast<const GetLocationSchemaTrace *>(item.trace.get());
            const int64_t input_len = read_trace->input_len() > 0 ? read_trace->input_len() : 0;
            try {
                item.normalized = NormalizeRequest(
                    read_trace->keys(), input_len, item.lane->block_size_tokens, item.lane->enable_prefix_hash);
            } catch (const std::exception &e) { item.error = e.what(); }
        });
        for (const BatchItem &item : batch) {
            if (!item.error.empty()) {
                fail_with("invalid request trace_id[" + item.record.trace_id + "]: " + item.error);
                batch.clear();
                return;
            }
        }

        // Lane commits stay in input order; only same-lane order is
        // semantically required, and input order trivially satisfies it.
        for (BatchItem &item : batch) {
            item.record.fact = item.lane->core.ProcessRequest(item.normalized.block_keys);
        }

        ParallelForIndex(batch.size(), worker_count, [&](std::size_t i) {
            BatchItem &item = batch[i];
            item.record.input_token_len = item.normalized.input_token_len;
            item.record.block_size_tokens = item.lane->block_size_tokens;
            item.record.block_bytes = item.lane->block_bytes;
            item.row = SerializeLiteHitFactRow(item.record);
        });

        for (const BatchItem &item : batch) {
            out << item.row << '\n';
        }
        if (!out.good()) {
            fail_with("write to temp facts file failed");
        }
        processed += batch.size();
        batch.clear();
    };

    const auto on_trace = [&](const std::shared_ptr<OptimizerSchemaTrace> &trace) {
        if (failed) {
            return;
        }
        if (trace->timestamp_ns() < last_timestamp_ns) {
            fail_with("trace is not sorted by timestamp_ns at trace_id[" + trace->trace_id() + "]");
            return;
        }
        last_timestamp_ns = trace->timestamp_ns();

        // Write events carry no read access; they are recognized and ignored.
        // (RequestSchemaTrace derives from GetLocationSchemaTrace and is a
        // read access.)
        if (dynamic_cast<const GetLocationSchemaTrace *>(trace.get()) == nullptr) {
            return;
        }

        BatchItem item;
        item.trace = trace;
        item.instance_id = override_id.empty() ? trace->instance_id() : override_id;
        const auto lane_it = lanes.find(item.instance_id);
        if (lane_it == lanes.end()) {
            fail_with("unknown instance[" + item.instance_id + "] at trace_id[" + trace->trace_id() + "]");
            return;
        }
        item.lane = lane_it->second.get();
        item.record.trace_id = trace->trace_id().empty() ? DefaultTraceId(*trace) : trace->trace_id();
        item.record.instance_id = item.instance_id;
        item.record.timestamp_ns = trace->timestamp_ns();
        batch.push_back(std::move(item));
        if (batch.size() >= batch_capacity) {
            process_batch();
        }
    };

    try {
        StandardTraceLoader::StreamFromFile(config_.trace_file_path(), on_trace);
        process_batch();
    } catch (const std::exception &e) { fail_with(std::string("failed to load/replay trace: ") + e.what()); }

    if (!failed && processed == 0) {
        fail_with("trace contains no valid Request/Get events");
    }

    if (!failed) {
        out.flush();
        if (!out.good()) {
            fail_with("flush of temp facts file failed");
        }
    }
    out.close();

    if (failed) {
        std::remove(temp_path.c_str());
        KVCM_LOG_ERROR("LiteHitOfflineRunner: aborted, no facts published: %s", failure_reason.c_str());
        return false;
    }

    if (std::rename(temp_path.c_str(), final_path.c_str()) != 0) {
        std::remove(temp_path.c_str());
        KVCM_LOG_ERROR("LiteHitOfflineRunner: atomic rename to [%s] failed", final_path.c_str());
        return false;
    }

    KVCM_LOG_INFO("LiteHitOfflineRunner: done. processed=%lu facts written to %s",
                  static_cast<unsigned long>(processed),
                  final_path.c_str());
    return true;
}

} // namespace kv_cache_manager
