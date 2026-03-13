#include "nfs_backend.h"

#include <memory>
#include <utility>

#include "kv_cache_manager/common/hash/hash.h"
#include "kv_cache_manager/common/logger.h"
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
    KVCM_LOG_INFO("open nfs backend success, config: [%s]", spec_.ToString().c_str());
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

std::vector<SpecCreateResult> NfsBackend::Create(const CreateBlocksRequest &request,
                                                  const std::string &trace_id,
                                                  std::function<void()> cb) {
    std::vector<SpecCreateResult> result;
    int32_t batch_size = spec_.key_count_per_file();
    batch_size = batch_size <= 0 ? 1 : batch_size;

    // Process each spec independently - never mix keys from different specs in a batch
    for (const auto &spec_block : request.spec_block_keys) {
        SpecCreateResult spec_result;
        const auto &block_keys = spec_block.block_keys;
        size_t total_key_count = block_keys.size();

        // Batch within this spec only
        for (size_t start = 0; start < total_key_count; start += batch_size) {
            size_t end = std::min(start + batch_size, total_key_count);

            // Build block key strings for this batch
            std::vector<std::string> batch_keys;
            batch_keys.reserve(end - start);
            for (size_t i = start; i < end; ++i) {
                std::string block_key = request.instance_id + "/" + spec_block.spec_name + "/" +
                                        StringUtil::Uint64ToHex(block_keys[i]);
                batch_keys.push_back(std::move(block_key));
            }

            // Generate URI for this batch
            DataStorageUri storage_uri;
            storage_uri.SetProtocol(ToString(GetType()));
            if (batch_keys.size() > 1) {
                std::string combine_key = StringUtil::Join(batch_keys, "|");
                std::string hash_str = StringUtil::Uint64ToHex(Hash64(combine_key.c_str(), combine_key.size(), 42));
                storage_uri.SetPath(spec_.root_path() + batch_keys[0] + "_" + hash_str);
            } else {
                storage_uri.SetPath(spec_.root_path() + batch_keys[0]);
            }
            storage_uri.SetParam("size", std::to_string(spec_block.spec_size));

            // Add result entry for each key in the batch
            for (size_t j = 0; j < batch_keys.size(); ++j) {
                if (batch_size > 1) {
                    storage_uri.SetParam("blkid", std::to_string(j));
                }
                spec_result.push_back({EC_OK, storage_uri});
            }
        }
        result.push_back(std::move(spec_result));
    }

    if (cb) {
        cb();
    }
    return result;
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

} // namespace kv_cache_manager
