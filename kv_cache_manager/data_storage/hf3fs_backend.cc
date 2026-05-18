#include "kv_cache_manager/data_storage/hf3fs_backend.h"

#include <fcntl.h>
#include <fstream>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <utility>
#include <vector>

#include "kv_cache_manager/common/hash/hash.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

Hf3fsDataStorageItem Hf3fsDataStorageItem::FromUri(const DataStorageUri &storage_uri) {
    Hf3fsDataStorageItem item;
    item.file_path = storage_uri.GetPath();
    return item;
}

Hf3fsBackend::Hf3fsBackend(std::shared_ptr<MetricsRegistry> metrics_registry)
    : DataStorageBackend(std::move(metrics_registry)) {}

DataStorageType Hf3fsBackend::GetType() { return DataStorageType::DATA_STORAGE_TYPE_HF3FS; }

bool Hf3fsBackend::Available() { return IsOpen() && IsAvailable(); }

double Hf3fsBackend::GetStorageUsageRatio(const std::string &trace_id) const { return 0.0; }

ErrorCode Hf3fsBackend::DoOpen(const StorageConfig &storage_config, const std::string &trace_id) {
    if (auto cfg = std::dynamic_pointer_cast<ThreeFSStorageSpec>(storage_config.storage_spec())) {
        spec_ = *cfg;
    } else {
        KVCM_LOG_WARN("unexpected config type, storage config: [%s]", storage_config.ToString().c_str());
        return EC_ERROR;
    }
    if (!DoInit()) {
        KVCM_LOG_WARN("open 3fs backend failed, spec [%s] is not valid", spec_.ToString().c_str());
        return EC_ERROR;
    }
    KVCM_LOG_INFO("open 3fs backend success, spec: [%s]", spec_.ToString().c_str());
    SetOpen(true);
    SetAvailable(true);
    return EC_OK;
};

ErrorCode Hf3fsBackend::Close() {
    KVCM_LOG_INFO("close 3fs backend");
    SetOpen(false);
    SetAvailable(false);
    return EC_OK;
};

std::vector<SpecCreateResult> Hf3fsBackend::Create(const CreateBlocksRequest &request,
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
                storage_uri.SetPath(base_path_ / (batch_keys[0] + "_" + hash_str));
            } else {
                storage_uri.SetPath(base_path_ / batch_keys[0]);
            }
            storage_uri.SetParam("size", std::to_string(spec_block.spec_size));

            // Add result entry for each key in the batch
            for (size_t j = 0; j < batch_keys.size(); ++j) {
                if (batch_size > 1) {
                    storage_uri.SetParam("blkid", std::to_string(j));
                }
                spec_result.push_back({EC_OK, storage_uri});
                if (spec_.touch_file_when_create()) {
                    Hf3fsDataStorageItem item = Hf3fsDataStorageItem::FromUri(storage_uri);
                    TouchFile(item.file_path);
                }
            }
        }
        result.push_back(std::move(spec_result));
    }

    if (cb) {
        cb();
    }
    return result;
};

std::vector<ErrorCode> Hf3fsBackend::Delete(const std::vector<DataStorageUri> &storage_uris,
                                            const std::string &trace_id,
                                            std::function<void()> cb) {
    std::vector<ErrorCode> result;
    for (size_t i = 0; i < storage_uris.size(); ++i) {
        Hf3fsDataStorageItem item = Hf3fsDataStorageItem::FromUri(storage_uris[i]);
        std::filesystem::path file_path = item.file_path;
        std::error_code ec;
        bool removed = std::filesystem::remove(file_path, ec);
        if (ec) {
            KVCM_LOG_ERROR("Failed to delete file [%s]: [%s]", file_path.string().c_str(), ec.message().c_str());
            result.push_back(EC_ERROR);
            continue;
        }
        if (!removed) {
            KVCM_LOG_WARN("Try delete file not exist, file: [%s]", file_path.string().c_str());
        }
        result.push_back(EC_OK);
    }
    if (cb) {
        cb();
    }
    return result;
};

std::vector<bool> Hf3fsBackend::Exist(const std::vector<DataStorageUri> &storage_uris) {
    std::vector<bool> result;
    for (int i = 0; i < storage_uris.size(); ++i) {
        Hf3fsDataStorageItem item = Hf3fsDataStorageItem::FromUri(storage_uris[i]);
        bool ret = std::filesystem::exists(item.file_path);
        result.push_back(ret);
    }
    return result;
}

std::vector<ErrorCode> Hf3fsBackend::Lock(const std::vector<DataStorageUri> &storage_uris) {
    std::vector<ErrorCode> result(storage_uris.size(), EC_OK);
    // not supported yet
    return result;
}

std::vector<ErrorCode> Hf3fsBackend::UnLock(const std::vector<DataStorageUri> &storage_uris) {
    std::vector<ErrorCode> result(storage_uris.size(), EC_OK);
    // not supported yet
    return result;
}

bool Hf3fsBackend::DoInit() {
    bool need_check_available = config_.check_storage_available_when_open();
    // mountpoint
    if (spec_.mountpoint().empty()) {
        KVCM_LOG_WARN("init 3fs backend failed, mountpoint is empty");
        return false;
    }
    if (need_check_available && !std::filesystem::exists(spec_.mountpoint())) {
        KVCM_LOG_WARN("init 3fs backend failed, mountpoint is not exists");
        return false;
    }
    // root dir
    if (spec_.root_dir().empty()) {
        KVCM_LOG_WARN("init 3fs backend failed, root_dir is empty");
        return false;
    }
    base_path_ = std::filesystem::path(spec_.mountpoint()) / spec_.root_dir();
    if (need_check_available && !std::filesystem::exists(base_path_)) {
        KVCM_LOG_WARN("init 3fs backend failed, root_dir is not exist : [%s]", base_path_.string().c_str());
        return false;
    }
    return true;
}

bool Hf3fsBackend::TouchFile(const std::string &file_path) {
    std::filesystem::path path(file_path);
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) {
        KVCM_LOG_WARN("Failed to create parent directories for [%s]: [%s]", file_path.c_str(), ec.message().c_str());
        return false;
    }
    std::ofstream ofs(file_path, std::ios::app);
    if (!ofs) {
        KVCM_LOG_WARN("Failed to open or create file [%s]", file_path.c_str());
        return false;
    }
    ofs.close();
    return true;
}

} // namespace kv_cache_manager
