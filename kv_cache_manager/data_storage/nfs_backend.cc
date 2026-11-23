#include "nfs_backend.h"

#include <charconv>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <system_error>
#include <utility>

#include "kv_cache_manager/common/hash/hash.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

namespace {

std::pair<ErrorCode, bool> ShouldDeleteFile(const DataStorageUri &storage_uri) {
    const std::string blkid = storage_uri.GetParam("blkid");
    if (blkid.empty()) {
        return {EC_OK, true};
    }

    std::uint64_t blkid_value = 0;
    const auto *begin = blkid.data();
    const auto *end = blkid.data() + blkid.size();
    const auto [ptr, ec] = std::from_chars(begin, end, blkid_value);
    if (ec != std::errc{} || ptr != end) {
        KVCM_LOG_WARN("Skip deleting nfs file for invalid blkid [%s], uri: [%s]",
                      blkid.c_str(),
                      storage_uri.ToUriString().c_str());
        return {EC_ERROR, false};
    }
    return {EC_OK, blkid_value == 0};
}

} // namespace

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

std::vector<ErrorCode> NfsBackend::Delete(const std::vector<DataStorageUri> &storage_uris,
                                          const std::string &trace_id,
                                          std::function<void()> cb) {
    std::vector<ErrorCode> result;
    result.reserve(storage_uris.size());
    for (const auto &storage_uri : storage_uris) {
        const auto [decision, should_delete] = ShouldDeleteFile(storage_uri);
        if (decision != EC_OK) {
            result.push_back(decision);
            continue;
        }
        if (!should_delete) {
            result.push_back(EC_OK);
            continue;
        }
        std::filesystem::path file_path = storage_uri.GetPath();
        std::error_code ec;
        const bool removed = std::filesystem::remove(file_path, ec);
        if (ec) {
            KVCM_LOG_ERROR("Failed to delete nfs file [%s]: [%s]", file_path.string().c_str(), ec.message().c_str());
            result.push_back(EC_ERROR);
            continue;
        }
        if (!removed) {
            KVCM_LOG_WARN("Try delete nfs file not exist, file: [%s]", file_path.string().c_str());
        }
        result.push_back(EC_OK);
    }
    if (cb) {
        cb();
    }
    return result;
}
std::vector<bool> NfsBackend::Exist(const std::vector<DataStorageUri> &storage_uris) {
    std::vector<bool> result;
    result.reserve(storage_uris.size());
    for (const auto &storage_uri : storage_uris) {
        std::error_code ec;
        const bool exists = std::filesystem::exists(storage_uri.GetPath(), ec);
        if (ec) {
            KVCM_LOG_ERROR("Failed to check nfs file [%s]: [%s]", storage_uri.GetPath().c_str(), ec.message().c_str());
            result.push_back(false);
            continue;
        }
        result.push_back(exists);
    }
    return result;
}
std::vector<bool> NfsBackend::MightExist(const std::vector<DataStorageUri> &storage_uris) {
    return std::vector<bool>(storage_uris.size(), true);
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
