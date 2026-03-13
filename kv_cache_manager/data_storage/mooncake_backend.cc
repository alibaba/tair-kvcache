#include "mooncake_backend.h"

#include <memory>
#include <random>
#include <sstream>
#include <type_traits>
#include <utility>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

MooncakeDataStorageItem MooncakeDataStorageItem::FromUri(const DataStorageUri &storage_uri) {
    MooncakeDataStorageItem item;
    item.key = storage_uri.GetParam("key");
    return item;
}

MooncakeBackend::MooncakeBackend(std::shared_ptr<MetricsRegistry> metrics_registry)
    : DataStorageBackend(std::move(metrics_registry)) {}

MooncakeBackend::~MooncakeBackend() {
    auto ec = Close();
    if (EC_OK != ec) {
        KVCM_LOG_WARN("close mooncake backend failed");
    }
}

DataStorageType MooncakeBackend::GetType() { return DataStorageType::DATA_STORAGE_TYPE_MOONCAKE; }

bool MooncakeBackend::Available() { return IsOpen() && IsAvailable(); }

double MooncakeBackend::GetStorageUsageRatio(const std::string &trace_id) const { return 0.0; }

ErrorCode MooncakeBackend::DoOpen(const StorageConfig &storage_config, const std::string &trace_id) {
    if (auto cfg = std::dynamic_pointer_cast<MooncakeStorageSpec>(storage_config.storage_spec())) {
        spec_ = *cfg;
    } else {
        KVCM_LOG_WARN("unexpected config type, storage config [%s]", storage_config.ToString().c_str());
        return EC_ERROR;
    }

    static std::random_device rd;
    static std::mt19937_64 rng(rd());
    static std::uniform_int_distribution<std::uint64_t> dis;
    const std::uint64_t rand_val = dis(rng);

    std::stringstream regenerate_local_hostname;
    regenerate_local_hostname << spec_.local_hostname() << "_"
                              << "kvcm"
                              << "_" << rand_val;

    // 完成初始化 transfer engine, 并连接 master server
    client_ = mooncake_client_create(regenerate_local_hostname.str().c_str(),
                                     spec_.metadata_connstring().c_str(),
                                     spec_.protocol().c_str(),
                                     spec_.rdma_device().c_str(),
                                     spec_.master_server_entry().c_str());
    if (client_ == nullptr) {
        KVCM_LOG_WARN("create mooncake client failed, regenerate_local_hostname: [%s], config: [%s]",
                      regenerate_local_hostname.str().c_str(),
                      spec_.ToString().c_str());
        return EC_ERROR;
    }

    SetOpen(true);
    SetAvailable(true);
    KVCM_LOG_INFO("open mooncake backend success, config: [%s]", spec_.ToString().c_str());
    return EC_OK;
};

ErrorCode MooncakeBackend::Close() {
    SetOpen(false);
    SetAvailable(false);
    KVCM_LOG_INFO("close mooncake backend");
    if (client_) {
        mooncake_client_destroy(client_);
    }
    return EC_OK;
};

std::vector<SpecCreateResult> MooncakeBackend::Create(const CreateBlocksRequest &request,
                                                      const std::string &trace_id,
                                                      std::function<void()> cb) {
    std::vector<SpecCreateResult> result;

    // Process all specs and keys - Mooncake doesn't batch, create one URI per key
    for (const auto &spec_block : request.spec_block_keys) {
        SpecCreateResult spec_result;
        for (size_t i = 0; i < spec_block.block_keys.size(); i++) {
            // Build the key string: instance_id/spec_name/hex(key)
            std::string key_str = request.instance_id + "/" + spec_block.spec_name + "/" +
                                  StringUtil::Uint64ToHex(spec_block.block_keys[i]);

            DataStorageUri storage_uri;
            storage_uri.SetProtocol(ToString(GetType()));
            storage_uri.SetPath("/");
            storage_uri.SetParam("key", key_str);
            storage_uri.SetParam("size", std::to_string(spec_block.spec_size));
            spec_result.push_back({EC_OK, storage_uri});
        }
        result.push_back(std::move(spec_result));
    }

    if (cb) {
        cb();
    }
    return result;
}
std::vector<ErrorCode> MooncakeBackend::Delete(const std::vector<DataStorageUri> &storage_uris,
                                               const std::string &trace_id,
                                               std::function<void()> cb) {
    std::vector<ErrorCode> result;
    for (int i = 0; i < storage_uris.size(); i++) {
        MooncakeDataStorageItem item = MooncakeDataStorageItem::FromUri(storage_uris[i]);
        ErrorCode_t err = mooncake_client_remove(client_, item.key.c_str());
        if (err != MOONCAKE_ERROR_OK) {
            KVCM_LOG_WARN("mooncake remove item failed, key: [%s], error: [%d]", item.key.c_str(), err);
            result.push_back(EC_ERROR);
        } else {
            result.push_back(EC_OK);
        }
    }
    if (cb) {
        cb();
    }
    return result;
}
std::vector<bool> MooncakeBackend::Exist(const std::vector<DataStorageUri> &storage_uris) {
    std::vector<bool> result;
    for (int i = 0; i < storage_uris.size(); i++) {
        MooncakeDataStorageItem item = MooncakeDataStorageItem::FromUri(storage_uris[i]);
        ErrorCode_t err = mooncake_client_query(client_, item.key.c_str());
        // TODO 失败和不存在要区分下呀
        if (err != MOONCAKE_ERROR_OK) {
            KVCM_LOG_WARN("mooncake lookup item failed, key: [%s], error: [%d]", item.key.c_str(), err);
            result.push_back(false);
        } else {
            result.push_back(true);
        }
    }
    return result;
}
std::vector<ErrorCode> MooncakeBackend::Lock(const std::vector<DataStorageUri> &storage_uris) {
    std::vector<ErrorCode> result(storage_uris.size(), EC_OK);
    // not supported yet
    return result;
}
std::vector<ErrorCode> MooncakeBackend::UnLock(const std::vector<DataStorageUri> &storage_uris) {
    std::vector<ErrorCode> result(storage_uris.size(), EC_OK);
    // not supported yet
    return result;
}

} // namespace kv_cache_manager
