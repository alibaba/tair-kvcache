#include "kv_cache_manager/data_storage/dummy_backend.h"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/hash/hash.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

DummyBackend::DummyBackend(std::shared_ptr<MetricsRegistry> metrics_registry)
    : DataStorageBackend(std::move(metrics_registry)) {}

DataStorageType DummyBackend::GetType() { return DataStorageType::DATA_STORAGE_TYPE_DUMMY; }

bool DummyBackend::Available() { return IsOpen() && IsAvailable(); }

double DummyBackend::GetStorageUsageRatio(const std::string &trace_id) const { return 0.0; }

ErrorCode DummyBackend::DoOpen(const StorageConfig &storage_config, const std::string &trace_id) {
    if (const auto cfg = std::dynamic_pointer_cast<DummyStorageSpec>(storage_config.storage_spec())) {
        spec_ = *cfg;
    } else {
        KVCM_LOG_WARN("unexpected config type, storage config: [%s]", storage_config.ToString().c_str());
        return ErrorCode::EC_ERROR;
    }
    if (spec_.root_path().empty()) {
        KVCM_LOG_WARN("open dummy backend failed, root_path is empty");
        return ErrorCode::EC_ERROR;
    }
    base_path_ = std::filesystem::path(spec_.root_path());
    std::error_code ec;
    std::filesystem::create_directories(base_path_, ec);
    if (ec) {
        KVCM_LOG_WARN("open dummy backend failed, cannot create root_path [%s], msg: [%s]",
                      base_path_.string().c_str(),
                      ec.message().c_str());
        return ErrorCode::EC_ERROR;
    }
    KVCM_LOG_INFO("open dummy backend success, config: [%s]", spec_.ToString().c_str());
    SetOpen(true);
    SetAvailable(true);
    return ErrorCode::EC_OK;
}

ErrorCode DummyBackend::Close() {
    KVCM_LOG_INFO("close dummy backend");
    SetOpen(false);
    SetAvailable(false);
    return ErrorCode::EC_OK;
}

std::vector<std::pair<ErrorCode, DataStorageUri>> DummyBackend::Create(const std::vector<std::string> &keys,
                                                                       const std::size_t size_per_key,
                                                                       const std::string &trace_id,
                                                                       const std::function<void()> cb) {
    std::vector<std::pair<ErrorCode, DataStorageUri>> results;
    std::vector<std::vector<std::string>> batches;

    auto batch_size = spec_.key_count_per_file();
    batch_size = batch_size <= 0 ? 1 : batch_size;
    using diff_t = std::vector<std::string>::difference_type;
    for (std::size_t start = 0; start < keys.size(); start += batch_size) {
        batches.emplace_back(std::next(keys.begin(), static_cast<diff_t>(start)),
                             std::next(keys.begin(), static_cast<diff_t>(std::min(start + batch_size, keys.size()))));
    }

    for (auto &batch : batches) {
        DataStorageUri storage_uri;
        storage_uri.SetProtocol(ToString(GetType()));

        if (batch.size() > 1) {
            std::string combine_key = StringUtil::Join(batch, "|");
            std::string hash_str = StringUtil::Uint64ToHex(Hash64(combine_key.c_str(), combine_key.size(), 42));
            storage_uri.SetPath(base_path_ / (batch[0] + "_" + hash_str));
        } else {
            storage_uri.SetPath(base_path_ / batch[0]);
        }

        storage_uri.SetParam("size", std::to_string(size_per_key));

        for (std::size_t i = 0; i != batch.size(); ++i) {
            if (batch_size > 1) {
                storage_uri.SetParam("blkid", std::to_string(i));
            }
            results.emplace_back(ErrorCode::EC_OK, storage_uri);
        }
    }

    if (cb) {
        cb();
    }

    return results;
}

std::vector<ErrorCode> DummyBackend::Delete(const std::vector<DataStorageUri> &storage_uris,
                                            const std::string &trace_id,
                                            const std::function<void()> cb) {
    std::vector<ErrorCode> results;
    for (auto &uri : storage_uris) {
        std::filesystem::path file_path = uri.GetPath();
        std::error_code ec;
        const bool removed = std::filesystem::remove(file_path, ec);
        if (ec) {
            KVCM_LOG_ERROR(
                "failed to delete file, path: [%s], msg: [%s]", file_path.string().c_str(), ec.message().c_str());
            results.push_back(ErrorCode::EC_ERROR);
            continue;
        }
        if (!removed) {
            KVCM_LOG_WARN("file not exist, path: [%s]", file_path.string().c_str());
        }
        results.push_back(ErrorCode::EC_OK);
    }

    if (cb) {
        cb();
    }

    return results;
}

std::vector<bool> DummyBackend::Exist(const std::vector<DataStorageUri> &storage_uris) {
    std::vector<bool> results;
    for (auto &uri : storage_uris) {
        std::error_code ec;
        const bool res = std::filesystem::exists(uri.GetPath(), ec);
        if (ec) {
            KVCM_LOG_ERROR("std::filesystem::exists call failed, err code: [%d], err msg: [%s], path: [%s]",
                           ec.value(),
                           ec.message().c_str(),
                           uri.GetPath().c_str());
            results.push_back(false);
            continue;
        }
        results.push_back(res);
    }
    return results;
}

std::vector<bool> DummyBackend::MightExist(const std::vector<DataStorageUri> &storage_uris) {
    return Exist(storage_uris);
}

std::vector<ErrorCode> DummyBackend::Lock(const std::vector<DataStorageUri> &storage_uris) {
    std::vector<ErrorCode> results(storage_uris.size(), ErrorCode::EC_OK);
    return results;
}

std::vector<ErrorCode> DummyBackend::UnLock(const std::vector<DataStorageUri> &storage_uris) {
    std::vector<ErrorCode> results(storage_uris.size(), ErrorCode::EC_OK);
    return results;
}

} // namespace kv_cache_manager
