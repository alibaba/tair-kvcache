#include "nfs_backend.h"

#include <memory>
#include <utility>

#include "kv_cache_manager/affinity/node_metrics.h"
#include "kv_cache_manager/common/hash/hash.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/net_util.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

NfsBackend::NfsBackend(std::shared_ptr<MetricsRegistry> metrics_registry)
    : DataStorageBackend(std::move(metrics_registry)) {}

DataStorageType NfsBackend::GetType() { return DataStorageType::DATA_STORAGE_TYPE_NFS; }

bool NfsBackend::Available() { return IsOpen() && IsAvailable(); }

double NfsBackend::GetStorageUsageRatio(const std::string &trace_id) const { return 0.0; }

ErrorCode NfsBackend::DoOpen(const StorageConfig &storage_config, const std::string &trace_id) {
    if (auto cfg = std::dynamic_pointer_cast<NfsStorageSpec>(storage_config.storage_spec())) {
        spec_ = *cfg;
    } else {
        KVCM_LOG_WARN("unexpected config type, storage config: [%s]", storage_config.ToString().c_str());
        return EC_ERROR;
    }
    local_node_id_ = NetUtil::GetLocalIp();
    KVCM_LOG_INFO("open nfs backend success, config: [%s], local_node_id: [%s]",
                  spec_.ToString().c_str(),
                  local_node_id_.c_str());
    SetOpen(true);
    SetAvailable(true);
    return EC_OK;
};

ErrorCode NfsBackend::Close() {
    KVCM_LOG_INFO("close nfs backend");
    SetOpen(false);
    SetAvailable(false);
    return EC_OK;
};

std::vector<std::pair<ErrorCode, DataStorageUri>> NfsBackend::Create(const std::vector<std::string> &keys,
                                                                     size_t size_per_key,
                                                                     const std::string &trace_id,
                                                                     std::function<void()> cb) {
    std::vector<std::pair<ErrorCode, DataStorageUri>> result;
    std::vector<std::vector<std::string>> batches;
    int32_t batch_size = spec_.key_count_per_file();
    batch_size = batch_size <= 0 ? 1 : batch_size;
    size_t total_key_count = keys.size();
    for (size_t start = 0; start < total_key_count; start += batch_size) {
        size_t end = std::min(start + batch_size, total_key_count);
        batches.emplace_back(keys.begin() + start, keys.begin() + end);
    }
    for (auto &batch : batches) {
        DataStorageUri storage_uri;
        storage_uri.SetProtocol(ToString(GetType()));
        if (batch.size() > 1) {
            std::string combine_key = StringUtil::Join(batch, "|");
            std::string hash_str = StringUtil::Uint64ToHex(Hash64(combine_key.c_str(), combine_key.size(), 42));
            storage_uri.SetPath(spec_.root_path() + batch[0] + "_" + hash_str);
        } else {
            storage_uri.SetPath(spec_.root_path() + batch[0]);
        }
        storage_uri.SetParam("size", std::to_string(size_per_key));
        for (size_t j = 0; j < batch.size(); ++j) {
            if (batch_size > 1) {
                storage_uri.SetParam("blkid", std::to_string(j));
            }
            result.push_back({EC_OK, storage_uri});
        }
    }
    if (cb) {
        cb();
    }
    return result;
}

std::vector<LocationDescriptor> NfsBackend::CreateWithHints(const std::vector<std::string> &keys,
                                                            size_t size_per_key,
                                                            const WriteHints &hints,
                                                            bool strict,
                                                            const std::string &trace_id,
                                                            std::function<void()> cb) {
    const std::string preferred = hints.preferred_node_ids.empty() ? std::string() : hints.preferred_node_ids.front();

    // In strict mode the caller demands placement on a specific node.
    // NFS backend always writes to local_node_id_, so if the preferred node
    // differs (or is empty), fail fast rather than silently placing elsewhere.
    if (strict && preferred != local_node_id_) {
        KVCM_LOG_WARN("CreateWithHints strict=true but preferred_node [%s] != local_node [%s], "
                      "rejecting %zu keys, trace_id=[%s]",
                      preferred.c_str(),
                      local_node_id_.c_str(),
                      keys.size(),
                      trace_id.c_str());
        std::vector<LocationDescriptor> out;
        out.reserve(keys.size());
        for (size_t i = 0; i < keys.size(); ++i) {
            out.push_back(LocationDescriptor{EC_ERROR, DataStorageUri{}, /*node_id=*/""});
        }
        return out;
    }

    auto legacy = Create(keys, size_per_key, trace_id, std::move(cb));
    std::vector<LocationDescriptor> out;
    out.reserve(legacy.size());
    for (auto &p : legacy) {
        if (!preferred.empty()) {
            p.second.SetParam("preferred_node", preferred);
            p.second.SetParam("local_node_id", local_node_id_);
        }
        out.push_back(LocationDescriptor{p.first, std::move(p.second), local_node_id_});
    }
    return out;
}

std::vector<ErrorCode> NfsBackend::Delete(const std::vector<DataStorageUri> &storage_uris,
                                          const std::string &trace_id,
                                          std::function<void()> cb) {
    std::vector<ErrorCode> result(storage_uris.size(), EC_OK);
    // not supported yet
    return result;
}
std::vector<bool> NfsBackend::Exist(const std::vector<DataStorageUri> &storage_uris) {
    std::vector<bool> result(storage_uris.size(), true);
    // not supported yet
    return result;
}
std::vector<ErrorCode> NfsBackend::Lock(const std::vector<DataStorageUri> &storage_uris) {
    std::vector<ErrorCode> result(storage_uris.size(), EC_OK);
    // not supported yet
    return result;
}
std::vector<ErrorCode> NfsBackend::UnLock(const std::vector<DataStorageUri> &storage_uris) {
    std::vector<ErrorCode> result(storage_uris.size(), EC_OK);
    // not supported yet
    return result;
}

std::vector<NodeMetrics> NfsBackend::SnapshotPerNodeMetrics() const {
    // 仅在 backend 已打开且本机 IP 缓存成功时上报, 避免污染节点表。
    if (!IsOpen() || local_node_id_.empty()) {
        return {};
    }
    NodeMetrics m;
    m.node_id = local_node_id_;
    m.node_name = local_node_id_;
    m.storage_type = DataStorageType::DATA_STORAGE_TYPE_NFS;
    m.free_bytes = std::numeric_limits<uint64_t>::max(); // infinite storage
    m.load_ratio = 0.0;
    return {std::move(m)};
}

} // namespace kv_cache_manager
