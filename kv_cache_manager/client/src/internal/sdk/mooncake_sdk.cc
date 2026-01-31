#include "mooncake_sdk.h"

#include <random>
#include <sstream>

namespace kv_cache_manager {
MooncakeRemoteItem MooncakeRemoteItem::FromUri(const DataStorageUri &storage_uri) {
    MooncakeRemoteItem item;
    item.key = storage_uri.GetParam("key");
    return item;
}
MooncakeSdk::~MooncakeSdk() {
    auto ec = Close();
    if (ER_OK != ec) {
        KVCM_LOG_WARN("close mooncake sdk failed");
    }
}

ClientErrorCode MooncakeSdk::Close() {
    KVCM_LOG_INFO("close mooncake sdk");
    if (client_) {
        mooncake_client_destroy(client_);
    }
    return ER_OK;
};

ClientErrorCode MooncakeSdk::Init(const std::shared_ptr<SdkBackendConfig> &sdk_backend_config,
                                  const std::shared_ptr<StorageConfig> &storage_config) {
    sdk_backend_config_ = std::dynamic_pointer_cast<MooncakeSdkConfig>(sdk_backend_config);
    if (!sdk_backend_config_) {
        KVCM_LOG_WARN("Init mooncake sdk failed, unexpected config type [%s]",
                      sdk_backend_config_ ? ToString(sdk_backend_config_->type()).c_str() : "unknown");
        return ER_INVALID_SDKBACKEND_CONFIG;
    }
    if (sdk_backend_config_->byte_size_per_block() <= 0) {
        KVCM_LOG_WARN("Init mooncake sdk failed, invalid byte_size_per_block [%ld]",
                      sdk_backend_config_->byte_size_per_block());
        return ER_INVALID_SDKBACKEND_CONFIG;
    }
    if (sdk_backend_config_->self_location_spec_name().empty()) {
        KVCM_LOG_WARN("Init mooncake sdk failed, self_location_spec_name can not empty");
        return ER_INVALID_SDKBACKEND_CONFIG;
    }
    storage_config_ = storage_config;
    if (!storage_config_) {
        KVCM_LOG_WARN("Init mooncake sdk failed, empty storage config");
        return ER_INVALID_STORAGE_CONFIG;
    }
    auto mooncake_spec = std::dynamic_pointer_cast<MooncakeStorageSpec>(storage_config_->storage_spec());
    if (!mooncake_spec) {
        KVCM_LOG_WARN("Init mooncake sdk failed, unexpected storage config type [%s]",
                      ToString(storage_config_->type()).c_str());
        return ER_INVALID_STORAGE_CONFIG;
    }

    static std::random_device rd;
    static std::mt19937_64 rng(rd());
    static std::uniform_int_distribution<std::uint64_t> dis;
    const std::uint64_t rand_val = dis(rng);

    std::stringstream regenerate_local_hostname;
    regenerate_local_hostname << mooncake_spec->local_hostname() << "_"
                              << sdk_backend_config_->self_location_spec_name() << "_" << rand_val;

    client_ = mooncake_client_create(regenerate_local_hostname.str().c_str(),
                                     mooncake_spec->metadata_connstring().c_str(),
                                     mooncake_spec->protocol().c_str(),
                                     mooncake_spec->rdma_device().c_str(),
                                     mooncake_spec->master_server_entry().c_str());
    if (client_ == nullptr) {
        KVCM_LOG_WARN("create mooncake client failed, regenerate_local_hostname: [%s], sdk backend config: [%s], "
                      "storage config: [%s]",
                      regenerate_local_hostname.str().c_str(),
                      sdk_backend_config_->ToString().c_str(),
                      storage_config_->ToString().c_str());
        return ER_SDKINIT_ERROR;
    }

    ErrorCode_t err = mooncake_client_register_local_memory(client_,
                                                            sdk_backend_config_->local_mem_ptr(),
                                                            sdk_backend_config_->local_buffer_size(),
                                                            sdk_backend_config_->location().c_str(),
                                                            false,
                                                            false);
    if (err != MOONCAKE_ERROR_OK) {
        KVCM_LOG_WARN(
            "failed to register local mem for mooncake client, local_buffer_size: [%zu], mooncake errorcode: [%d]",
            sdk_backend_config_->local_buffer_size(),
            err);
        return ER_SDKINIT_ERROR;
    }

    ErrorCode_t err2 = mooncake_client_mount_segment(client_, sdk_backend_config_->local_buffer_size());

    if (err2 != MOONCAKE_ERROR_OK) {
        KVCM_LOG_WARN("failed to mount segment for mooncake client, global size: [%zu], mooncake errorcode: [%d]",
                      sdk_backend_config_->local_buffer_size(),
                      err);
        return ER_SDKINIT_ERROR;
    }

    return ER_OK;
}

SdkType MooncakeSdk::Type() { return SdkType::MOONCAKE; }

ClientErrorCode MooncakeSdk::Get(const std::vector<DataStorageUri> &remote_uris, const BlockBuffers &local_buffer) {
    if (remote_uris.size() != local_buffer.size()) {
        KVCM_LOG_ERROR("mooncake get failed, remote_uris size not equal to local_buffer size");
        return ER_INVALID_PARAMS;
    }
    for (int i = 0; i < remote_uris.size(); i++) {
        MooncakeRemoteItem item = MooncakeRemoteItem::FromUri(remote_uris[i]);
        std::vector<Slice_t> slices;
        auto [read_len, success] = extractSlices(item, local_buffer[i], slices);
        if (!success) {
            KVCM_LOG_WARN("mooncake get item failed, key: %s, extract slices failed", item.key.c_str());
            return ER_EXTRACT_SLICES_ERROR;
        }
        if (read_len == 0 || read_len > sdk_backend_config_->byte_size_per_block()) {
            KVCM_LOG_WARN("mooncake get but iovs are invalid, key: [%s], write_len: [%zu], byte_size_per_block: [%ld]",
                          item.key.c_str(),
                          read_len,
                          sdk_backend_config_->byte_size_per_block());
            return ER_INVALID_PARAMS;
        }
        ErrorCode_t err = mooncake_client_get(client_, item.key.c_str(), slices.data(), slices.size());
        if (err != MOONCAKE_ERROR_OK) {
            KVCM_LOG_WARN("mooncake get item failed, key: [%s], mooncake errorcode: [%d]", item.key.c_str(), err);
            return ER_SDKREAD_ERROR;
        }
    }
    return ER_OK;
}

ClientErrorCode MooncakeSdk::Put(const std::vector<DataStorageUri> &remote_uris,
                                 const BlockBuffers &local_buffers,
                                 std::shared_ptr<std::vector<DataStorageUri>> actual_remote_uris) {
    actual_remote_uris->clear();
    std::vector<Slice_t> slices;
    if (remote_uris.size() != local_buffers.size()) {
        KVCM_LOG_WARN("mooncake put failed, remote_uris size not equal to local_buffers size");
        return ER_INVALID_PARAMS;
    }
    for (int i = 0; i < remote_uris.size(); i++) {
        MooncakeRemoteItem item = MooncakeRemoteItem::FromUri(remote_uris[i]);
        auto [write_len, success] = extractSlices(item, local_buffers[i], slices);
        if (!success) {
            KVCM_LOG_WARN("mooncake put item failed, key: %s, extract slices failed", item.key.c_str());
            return ER_EXTRACT_SLICES_ERROR;
        }
        if (write_len == 0 || write_len > sdk_backend_config_->byte_size_per_block()) {
            KVCM_LOG_WARN("mooncake put but iovs are invalid, key: [%s], write_len: [%zu], byte_size_per_block: [%ld]",
                          item.key.c_str(),
                          write_len,
                          sdk_backend_config_->byte_size_per_block());
            return ER_INVALID_PARAMS;
        }

        ReplicateConfig_t cfg;
        cfg.replica_num = sdk_backend_config_->put_replica_num();
        auto err = mooncake_client_put(client_, item.key.c_str(), slices.data(), slices.size(), cfg);
        if (err != MOONCAKE_ERROR_OK) {
            KVCM_LOG_WARN("mooncake put item failed, key: [%s], mooncake errorcode: [%d]", item.key.c_str(), err);
            return ER_SDKWRITE_ERROR;
        }
    }
    return Alloc(remote_uris, *actual_remote_uris);
}

ClientErrorCode MooncakeSdk::Alloc(const std::vector<DataStorageUri> &remote_uris,
                                   std::vector<DataStorageUri> &alloc_uris) {
    alloc_uris = remote_uris;
    return ER_OK;
}

std::pair<size_t, bool> MooncakeSdk::extractSlices(const MooncakeRemoteItem &item,
                                                   const BlockBuffer &buffer,
                                                   std::vector<Slice_t> &slices) const {
    const uint64_t kMaxSliceSize = mooncake_max_slice_size();
    slices.clear();
    size_t len_byte = 0;
    for (const auto &iov : buffer.iovs) {
        // 支持 get 时 ignore 部分 iov, 空切片会被 mooncake-transfer-engine 忽略
        if (iov.ignore) {
            size_t ignore_size = iov.size;
            while (ignore_size > 0) {
                size_t chunk_size = std::min(ignore_size, kMaxSliceSize);
                slices.push_back({NULL, chunk_size});
                ignore_size -= chunk_size;
            }
            continue;
        }

        if (iov.base == nullptr) {
            KVCM_LOG_WARN("extract slices failed, iov data is null, key: [%s]", item.key.c_str());
            return {len_byte, false};
        }

        len_byte += iov.size;
        size_t remaining_size = iov.size;
        size_t offset = 0;
        char *base_ptr = static_cast<char *>(iov.base);
        while (remaining_size > 0) {
            size_t chunk_size = std::min(remaining_size, kMaxSliceSize);
            slices.push_back({base_ptr + offset, chunk_size});
            remaining_size -= chunk_size;
            offset += chunk_size;
        }
    }
    // if (len_byte != sdk_config_->byte_size_per_block()) {
    //     KVCM_LOG_WARN("extract slices failed, alloc size is not equal to byte_size_per_block, key: %s",
    //     item.key.c_str()); return {len_byte, false};
    // }
    return {len_byte, true};
}

} // namespace kv_cache_manager