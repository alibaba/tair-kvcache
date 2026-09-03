#include "mooncake_sdk.h"

#include <chrono>
#include <random>
#include <sstream>

#include "kv_cache_manager/client/src/internal/sdk/deadline_util.h"

namespace kv_cache_manager {

namespace {
// 超时归因日志：定位哪个 block/key 被拒、哪块 caller buffer 可能仍在被 DMA 写。
void LogSoftTimeout(bool is_get,
                    size_t done,
                    size_t total,
                    size_t refused_block_idx,
                    const std::string &key,
                    const void *caller_buffer,
                    size_t buffer_size,
                    int64_t elapsed_ms) {
    KVCM_LOG_WARN("mooncake %s timeout: done=%zu/%zu refused_block_idx=%zu key=%s "
                  "caller_buffer=%p caller_buffer_size=%zu elapsed_ms=%lld "
                  "soft-contract: in-flight RDMA cannot be cancelled, blocks [0,%zu) may still be written",
                  is_get ? "get" : "put",
                  done,
                  total,
                  refused_block_idx,
                  key.c_str(),
                  caller_buffer,
                  buffer_size,
                  static_cast<long long>(elapsed_ms),
                  done);
}
} // namespace

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
    if (sdk_backend_config_->self_location_spec_name().empty()) {
        KVCM_LOG_WARN("Init mooncake sdk failed, self_location_spec_name can not empty");
        return ER_INVALID_SDKBACKEND_CONFIG;
    }
    if (sdk_backend_config_->spec_byte_sizes_per_block().empty()) {
        KVCM_LOG_WARN("Init mooncake sdk failed, spec_byte_sizes_per_block is empty");
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
        KVCM_LOG_WARN("failed to register local mem for mooncake client, "
                      "local_mem_ptr: [%p], local_buffer_size: [%zu], mooncache errorcode: [%d]",
                      sdk_backend_config_->local_mem_ptr(),
                      sdk_backend_config_->local_buffer_size(),
                      err);
        return ER_SDKINIT_ERROR;
    }

    return ER_OK;
}

SdkType MooncakeSdk::Type() { return SdkType::MOONCAKE; }

ClientErrorCode MooncakeSdk::Get(const std::vector<DataStorageUri> &remote_uris,
                                 const BlockBuffers &local_buffer) {
    if (remote_uris.size() != local_buffer.size()) {
        KVCM_LOG_ERROR("mooncake get failed, remote_uris size not equal to local_buffer size");
        return ER_INVALID_PARAMS;
    }
    // 静态预算：Init 时由 wrapper 注入，从自身任务起点起算 deadline。
    const int64_t deadline_ms = SteadyClockMs() + sdk_backend_config_->timeout_config().get_timeout_ms();
    // 本 SDK 调用内的墙钟起点：超时归因日志的 elapsed_ms 用（不含线程池排队时间，
    // 那是 wrapper 层日志的职责）。
    const auto call_start = std::chrono::steady_clock::now();
    // 防御性校验上界：取所有允许的 byte_size_per_block 的最大值
    int64_t max_allowed_size = 0;
    for (const auto &[spec_name, byte_size_per_block] : sdk_backend_config_->spec_byte_sizes_per_block()) {
        max_allowed_size = std::max(max_allowed_size, byte_size_per_block);
    }
    for (int i = 0; i < remote_uris.size(); i++) {
        MooncakeRemoteItem item = MooncakeRemoteItem::FromUri(remote_uris[i]);
        std::vector<Slice_t> slices;
        auto [read_len, success] = extractSlices(item, local_buffer[i], slices);
        if (!success) {
            KVCM_LOG_WARN("mooncake get item failed, key: %s, extract slices failed", item.key.c_str());
            return ER_EXTRACT_SLICES_ERROR;
        }
        if (read_len == 0) {
            KVCM_LOG_WARN(
                "mooncake get but iovs are invalid, key: [%s], read_len: [%zu] is zero", item.key.c_str(), read_len);
            return ER_INVALID_PARAMS;
        }
        if (read_len > max_allowed_size) {
            KVCM_LOG_WARN(
                "mooncake get but iovs exceed max allowed size, key: [%s], read_len: [%zu], max_allowed: [%ld]",
                item.key.c_str(),
                read_len,
                max_allowed_size);
            return ER_INVALID_PARAMS;
        }
        // ============================================================
        // 逐 key 准入检查（核心）：slices 直接指向 caller 的 iov.base，网卡 DMA 直接
        // 写 caller 内存，而上游无法取消已下发的传输。
        // 因此在每次 mooncake_client_get 之前检查 deadline_ms：已过期立即返回超时、
        // 不发这次 I/O。效果：超时时刻最多只有 1 个 block 的 DMA 在飞（正在执行的
        // 那次），其余全部未发起 —— 暴露面从 128 个 block 降到 ≤1 个（降两个数量级）。
        // 这个检查看似"每个 key 都查一次"很啰嗦，但正是它把静默污染窗口关到最小；
        // 删掉它，超时后 128 个 block 全部可能仍在写 caller buffer。
        // ============================================================
        if (DeadlineExpired(deadline_ms)) {
            const void *caller_buffer = nullptr;
            for (const auto &iov : local_buffer[i].iovs) {
                if (iov.base != nullptr) {
                    caller_buffer = iov.base;
                    break;
                }
            }
            const auto elapsed_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - call_start)
                    .count();
            // 已下发的 blocks [0, i) 可能仍有在飞 DMA 写 caller buffer，返回后无法保证安全。
            LogSoftTimeout(/*is_get=*/true,
                           /*done=*/i,
                           remote_uris.size(),
                           i,
                           item.key,
                           caller_buffer,
                           read_len,
                           elapsed_ms);
            return ER_SDK_TIMEOUT;
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
    // 静态预算：Init 时由 wrapper 注入，从自身任务起点起算 deadline。
    const int64_t deadline_ms = SteadyClockMs() + sdk_backend_config_->timeout_config().put_timeout_ms();
    // 本 SDK 调用内的墙钟起点：超时归因日志的 elapsed_ms 用（不含线程池排队时间）。
    const auto call_start = std::chrono::steady_clock::now();
    // 防御性校验上界：取所有允许的 byte_size_per_block 的最大值
    int64_t max_allowed_size = 0;
    for (const auto &[spec_name, byte_size_per_block] : sdk_backend_config_->spec_byte_sizes_per_block()) {
        max_allowed_size = std::max(max_allowed_size, byte_size_per_block);
    }
    for (int i = 0; i < remote_uris.size(); i++) {
        MooncakeRemoteItem item = MooncakeRemoteItem::FromUri(remote_uris[i]);
        auto [write_len, success] = extractSlices(item, local_buffers[i], slices);
        if (!success) {
            KVCM_LOG_WARN("mooncake put item failed, key: %s, extract slices failed", item.key.c_str());
            return ER_EXTRACT_SLICES_ERROR;
        }
        if (write_len == 0) {
            KVCM_LOG_WARN(
                "mooncake put but iovs are invalid, key: [%s], write_len: [%zu] is zero", item.key.c_str(), write_len);
            return ER_INVALID_PARAMS;
        }
        if (write_len > max_allowed_size) {
            KVCM_LOG_WARN(
                "mooncake put but iovs exceed max allowed size, key: [%s], write_len: [%zu], max_allowed: [%ld]",
                item.key.c_str(),
                write_len,
                max_allowed_size);
            return ER_INVALID_PARAMS;
        }
        // ============================================================
        // 逐 key 准入检查（与 Get 同理，见上）：put 时网卡 DMA 读 caller 内存，
        // 超时返回后若 caller 复用/改写该内存，与在飞 DMA 构成数据竞争；上游无法
        // 取消，因此每次 mooncake_client_put 之前必须检查 deadline_ms，已过期立即返回。
        // 效果：暴露面从 128 个 block 降到 ≤1 个。
        // ============================================================
        if (DeadlineExpired(deadline_ms)) {
            const void *caller_buffer = nullptr;
            for (const auto &iov : local_buffers[i].iovs) {
                if (iov.base != nullptr) {
                    caller_buffer = iov.base;
                    break;
                }
            }
            const auto elapsed_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - call_start)
                    .count();
            LogSoftTimeout(/*is_get=*/false,
                           /*done=*/i,
                           remote_uris.size(),
                           i,
                           item.key,
                           caller_buffer,
                           write_len,
                           elapsed_ms);
            return ER_SDK_TIMEOUT;
        }
        ReplicateConfig_t cfg;
        cfg.replica_num = sdk_backend_config_->put_replica_num();
        auto err = mooncake_client_put(client_, item.key.c_str(), slices.data(), slices.size(), cfg);
        if (err != MOONCAKE_ERROR_OK) {
            KVCM_LOG_WARN("mooncake put item failed, key: [%s], mooncake errorcode: [%d]", item.key.c_str(), err);
            return ER_SDKWRITE_ERROR;
        }
    }
    // 保序契约：actual_remote_uris 与 remote_uris 同序 —— Alloc 是整体赋值
    // （alloc_uris = remote_uris），顺序天然正确；将来若 local alloc 改为逐项回填，
    // 必须按下标回填原位（下标是 block 的唯一身份，见 sdk_type.h BlockGroup::indices）。
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