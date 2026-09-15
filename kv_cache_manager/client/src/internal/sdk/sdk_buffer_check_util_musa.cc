#include <algorithm>
#include <limits>

#include "kv_cache_manager/client/src/internal/sdk/sdk_buffer_check_util.h"
#include "kv_cache_manager/common/env_util.h"
#include "kv_cache_manager/common/hash_util.h"

namespace kv_cache_manager {

std::vector<int64_t> SdkBufferCheckUtil::GetBlocksHash(const BlockBuffers &block_buffers) {
    if (block_buffers.empty() || block_buffers.front().iovs.empty()) {
        return {};
    }
    std::vector<IovDevice> iov_h;
    const size_t iov_num = block_buffers.front().iovs.size();
    iov_h.reserve(iov_num * block_buffers.size());
    for (const auto &block_buffer : block_buffers) {
        if (block_buffer.iovs.size() != iov_num) {
            return {};
        }
        for (const auto &raw_iov : block_buffer.iovs) {
            iov_h.push_back({raw_iov.base, raw_iov.size});
        }
    }
    auto crcs = GetIovsCrc(iov_h);
    if (crcs.size() != iov_h.size()) {
        return {};
    }
    std::vector<int64_t> result;
    result.reserve(block_buffers.size());
    for (size_t offset = 0; offset < crcs.size(); offset += iov_num) {
        result.push_back(HashUtil::HashIntArray(&crcs[offset], &crcs[offset + iov_num], 0));
    }
    return result;
}

std::vector<int64_t> SdkBufferCheckUtil::GetBlocksHash(
    const BlockBuffers &block_buffers, IovDevice *iovs_d, uint32_t *crcs_d, size_t max_iov_num, GpuStream_t stream) {
    if (max_iov_num == 0 || max_iov_num > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return {};
    }
    std::vector<IovDevice> iov_h(max_iov_num);
    return GetBlocksHash(block_buffers, iovs_d, crcs_d, iov_h.data(), max_iov_num, stream);
}

std::vector<int64_t> SdkBufferCheckUtil::GetBlocksHash(const BlockBuffers &block_buffers,
                                                       IovDevice *iovs_d,
                                                       uint32_t *crcs_d,
                                                       IovDevice *iovs_h_to_save,
                                                       size_t max_iov_num,
                                                       GpuStream_t stream) {
    if (block_buffers.empty() || block_buffers.front().iovs.empty() || iovs_d == nullptr || crcs_d == nullptr ||
        iovs_h_to_save == nullptr || max_iov_num == 0) {
        return {};
    }
    const size_t iov_num = block_buffers.front().iovs.size();
    size_t iovs_size = 0;
    for (const auto &block_buffer : block_buffers) {
        if (iov_num != block_buffer.iovs.size()) {
            return {};
        }
        if (block_buffer.iovs.size() > max_iov_num - iovs_size) {
            break;
        }
        for (const auto &raw_iov : block_buffer.iovs) {
            iovs_h_to_save[iovs_size].base = raw_iov.base;
            iovs_h_to_save[iovs_size].size = raw_iov.size;
            iovs_size++;
        }
    }
    auto crcs = GetIovsCrc(iovs_h_to_save, iovs_size, iovs_d, crcs_d, stream);
    if (crcs.size() != iovs_size) {
        return {};
    }
    std::vector<int64_t> result;
    result.reserve(iovs_size / iov_num);
    for (size_t offset = 0; offset < crcs.size(); offset += iov_num) {
        result.push_back(HashUtil::HashIntArray(&crcs[offset], &crcs[offset + iov_num], 0));
    }
    return result;
}

std::vector<uint32_t> SdkBufferCheckUtil::GetIovsCrc(const std::vector<IovDevice> &iovs_h) {
    if (iovs_h.empty() || iovs_h.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return {};
    }
    MusaBufferGuard iovs_guard, crcs_guard;
    if (!iovs_guard.Alloc(sizeof(IovDevice) * iovs_h.size()) ||
        !crcs_guard.Alloc(sizeof(uint32_t) * iovs_h.size())) {
        return {};
    }
    return GetIovsCrc(iovs_h,
                      static_cast<IovDevice *>(iovs_guard.Get()),
                      static_cast<uint32_t *>(crcs_guard.Get()),
                      nullptr);
}

std::vector<uint32_t> SdkBufferCheckUtil::GetIovsCrc(const std::vector<IovDevice> &iovs_h,
                                                     IovDevice *iovs_d,
                                                     uint32_t *crcs_d,
                                                     GpuStream_t stream) {
    return GetIovsCrc(iovs_h.data(), iovs_h.size(), iovs_d, crcs_d, stream);
}

SdkBufferCheckPool::SdkBufferCheckPool(size_t cell_num) { cells_.resize(cell_num); }

SdkBufferCheckPool::~SdkBufferCheckPool() {
    for (const auto &cell : cells_) {
        if (cell.gpu_stream) {
            CHECK_MUSA_ERROR(musaStreamDestroy(cell.gpu_stream), "musa stream destroy failed");
        }
        if (cell.h_iovs) {
            CHECK_MUSA_ERROR(musaFreeHost(cell.h_iovs), "musa free iovs_h_mem[%p] failed", cell.h_iovs);
        }
        if (cell.d_iovs) {
            CHECK_MUSA_ERROR(musaFree(cell.d_iovs), "musa free d_iovs[%p] failed", cell.d_iovs);
        }
        if (cell.d_crcs) {
            CHECK_MUSA_ERROR(musaFree(cell.d_crcs), "musa free d_crcs[%p] failed", cell.d_crcs);
        }
    }
}

bool SdkBufferCheckPool::Init(size_t max_check_iov_num) {
    if (cells_.empty() || max_check_iov_num == 0 ||
        max_check_iov_num > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        max_check_iov_num > std::numeric_limits<size_t>::max() / sizeof(IovDevice) ||
        max_check_iov_num > std::numeric_limits<size_t>::max() / sizeof(uint32_t)) {
        KVCM_LOG_ERROR("invalid checksum pool config, cell_num [%lu], max_check_iov_num [%lu]",
                       cells_.size(),
                       max_check_iov_num);
        return false;
    }
    size_t iovs_byte_size = max_check_iov_num * sizeof(IovDevice);
    size_t crcs_byte_size = max_check_iov_num * sizeof(uint32_t);
    CHECK_MUSA_ERROR_RETURN(musaGetDevice(&device_id_), false, "musaGetDevice failed");
    for (auto &cell : cells_) {
        CHECK_MUSA_ERROR_RETURN(
            musaMallocHost(&cell.h_iovs, iovs_byte_size), false, "musaMallocHost [%zu] bytes failed", iovs_byte_size);
        CHECK_MUSA_ERROR_RETURN(
            musaMalloc(&cell.d_iovs, iovs_byte_size), false, "musaMalloc [%zu] byte failed", iovs_byte_size);
        CHECK_MUSA_ERROR_RETURN(
            musaMalloc(&cell.d_crcs, crcs_byte_size), false, "musaMalloc [%zu] byte failed", crcs_byte_size);
        CHECK_MUSA_ERROR_RETURN(
            musaStreamCreateWithFlags(&cell.gpu_stream, musaStreamNonBlocking), false, "musa stream create failed");
        cell_queue_.push(&cell);
    }
    KVCM_LOG_INFO(
        "cell_size[%lu], iovs_byte_size[%lu], crcs_byte_size[%lu]", cells_.size(), iovs_byte_size, crcs_byte_size);
    return true;
}

SdkBufferCheckPool::CellHandle::CellHandle(SdkBufferCheckPool *pool, Cell *cell, int device_id)
    : pool_(pool), cell_(cell) {
    musaError_t err = musaGetDevice(&prev_device_id_);
    if (err != musaSuccess) {
        KVCM_LOG_WARN("musa error [%d] [%s] | musaGetDevice failed", err, musaGetErrorString(err));
    } else if (prev_device_id_ != device_id) {
        err = musaSetDevice(device_id);
        if (err != musaSuccess) {
            KVCM_LOG_WARN("musa error [%d] [%s] | musaSetDevice [%d] failed", err, musaGetErrorString(err), device_id);
        } else {
            changed_device_ = true;
        }
    }
}

SdkBufferCheckPool::CellHandle::~CellHandle() {
    if (pool_) {
        pool_->PutCell(cell_);
    }
    if (changed_device_) {
        CHECK_MUSA_ERROR(musaSetDevice(prev_device_id_), "musaSetDevice prev [%d] failed", prev_device_id_);
    }
}

SdkBufferCheckPool::CellHandle SdkBufferCheckPool::GetCell() {
    std::unique_lock lock(mutex_);
    cv_.wait(lock, [this] { return !cell_queue_.empty(); });
    Cell *cell = cell_queue_.front();
    cell_queue_.pop();
    return CellHandle(this, cell, device_id_);
}

void SdkBufferCheckPool::PutCell(Cell *cell) {
    std::unique_lock lock(mutex_);
    cell_queue_.push(cell);
    cv_.notify_one();
}

} // namespace kv_cache_manager
