#include "kv_cache_manager/client/src/internal/sdk/sdk_buffer_check_util.h"

#include <algorithm>
#include <cassert>

#include "kv_cache_manager/common/env_util.h"
#include "kv_cache_manager/common/hash_util.h"

namespace kv_cache_manager {

std::vector<int64_t> SdkBufferCheckUtil::GetBlocksHash(const BlockBuffers &block_buffers) {
    std::vector<IovDevice> iov_h;
    size_t iov_num = block_buffers.front().iovs.size();
    iov_h.reserve(iov_num * block_buffers.size());
    for (const auto &block_buffer : block_buffers) {
        for (const auto &raw_iov : block_buffer.iovs) {
            iov_h.push_back({raw_iov.base, raw_iov.size});
        }
    }
    auto crcs = GetIovsCrc(iov_h);
    std::vector<int64_t> result;
    result.reserve(block_buffers.size());
    for (size_t offset = 0; offset < crcs.size(); offset += iov_num) {
        result.push_back(HashUtil::HashIntArray(&crcs[offset], &crcs[offset + iov_num], 0));
    }
    return result;
}

std::vector<int64_t> SdkBufferCheckUtil::GetBlocksHash(
    const BlockBuffers &block_buffers, IovDevice *iovs_d, uint32_t *crcs_d, size_t max_iov_num, cudaStream_t stream) {
    std::vector<IovDevice> iov_h(max_iov_num);
    return GetBlocksHash(block_buffers, iovs_d, crcs_d, iov_h.data(), max_iov_num, stream);
}

std::vector<int64_t> SdkBufferCheckUtil::GetBlocksHash(const BlockBuffers &block_buffers,
                                                       IovDevice *iovs_d,
                                                       uint32_t *crcs_d,
                                                       IovDevice *iovs_h_to_save,
                                                       size_t max_iov_num,
                                                       cudaStream_t stream) {
    size_t iov_num = block_buffers.front().iovs.size();
    size_t iovs_size = 0;
    for (const auto &block_buffer : block_buffers) {
        assert(iov_num == block_buffer.iovs.size());
        if (iovs_size + block_buffer.iovs.size() > max_iov_num) {
            break;
        }
        for (const auto &raw_iov : block_buffer.iovs) {
            iovs_h_to_save[iovs_size].base = raw_iov.base;
            iovs_h_to_save[iovs_size].size = raw_iov.size;
            iovs_size++;
        }
    }
    auto crcs = GetIovsCrc(iovs_h_to_save, iovs_size, iovs_d, crcs_d, stream);
    std::vector<int64_t> result;
    result.reserve(iovs_size / iov_num);
    for (size_t offset = 0; offset < crcs.size(); offset += iov_num) {
        result.push_back(HashUtil::HashIntArray(&crcs[offset], &crcs[offset + iov_num], 0));
    }
    return result;
}

std::vector<uint32_t> SdkBufferCheckUtil::GetIovsCrc(const std::vector<IovDevice> &iovs_h) {
    IovDevice *iovs_d = nullptr;
    uint32_t *crcs_d = nullptr;
    CHECK_CUDA_ERROR_RETURN(cudaMalloc(&iovs_d, sizeof(IovDevice) * iovs_h.size()), {}, "cudaMalloc fail");
    CHECK_CUDA_ERROR_RETURN(cudaMalloc(&crcs_d, sizeof(uint32_t) * iovs_h.size()), {}, "cudaMalloc fail");
    auto crcs = GetIovsCrc(iovs_h, iovs_d, crcs_d, nullptr);
    CHECK_CUDA_ERROR_RETURN(cudaFree(iovs_d), {}, "cudaMalloc fail");
    CHECK_CUDA_ERROR_RETURN(cudaFree(crcs_d), {}, "cudaMalloc fail");
    return crcs;
}

std::vector<uint32_t> SdkBufferCheckUtil::GetIovsCrc(const std::vector<IovDevice> &iovs_h,
                                                     IovDevice *iovs_d,
                                                     uint32_t *crcs_d,
                                                     cudaStream_t stream) {
    return GetIovsCrc(iovs_h.data(), iovs_h.size(), iovs_d, crcs_d, stream);
}

SdkBufferCheckPool::SdkBufferCheckPool(size_t cell_num) { cells_.resize(cell_num); }

SdkBufferCheckPool::~SdkBufferCheckPool() {
    for (const auto &cell : cells_) {
        if (cell.h_iovs) {
            CHECK_CUDA_ERROR(cudaFreeHost(cell.h_iovs), "cuda free iovs_h_mem[%p] failed", cell.h_iovs);
        }
        if (cell.d_iovs) {
            CHECK_CUDA_ERROR(cudaFree(cell.d_iovs), "cuda free d_iovs[%p] failed", cell.d_iovs);
        }
        if (cell.d_iovs) {
            CHECK_CUDA_ERROR(cudaFree(cell.d_crcs), "cuda free d_crcs[%p] failed", cell.d_crcs);
        }
    }
}

bool SdkBufferCheckPool::Init(size_t max_check_iov_num) {
    size_t iovs_byte_size = max_check_iov_num * sizeof(IovDevice);
    size_t crcs_byte_size = max_check_iov_num * sizeof(uint32_t);
    for (auto &cell : cells_) {
        CHECK_CUDA_ERROR_RETURN(
            cudaMallocHost(&cell.h_iovs, iovs_byte_size), false, "cudaMallocHost [%zu] bytes failed", iovs_byte_size);
        CHECK_CUDA_ERROR_RETURN(
            cudaMalloc(&cell.d_iovs, iovs_byte_size), false, "cudaMalloc [%zu] byte failed", iovs_byte_size);
        CHECK_CUDA_ERROR_RETURN(
            cudaMalloc(&cell.d_crcs, crcs_byte_size), false, "cudaMalloc [%zu] byte failed", crcs_byte_size);
        CHECK_CUDA_ERROR_RETURN(
            cudaStreamCreateWithFlags(&cell.cuda_stream, cudaStreamNonBlocking), false, "cuda stream create failed");
        cell_queue_.push(&cell);
    }
    return true;
}

SdkBufferCheckPool::CellHandle::~CellHandle() {
    if (pool_) {
        pool_->PutCell(cell_);
    }
}

SdkBufferCheckPool::CellHandle SdkBufferCheckPool::GetCell() {
    std::unique_lock lock(mutex_);
    cv_.wait(lock, [this] { return !cell_queue_.empty(); });
    Cell *cell = cell_queue_.front();
    cell_queue_.pop();
    return CellHandle(this, cell);
}

void SdkBufferCheckPool::PutCell(Cell *cell) {
    std::unique_lock lock(mutex_);
    cell_queue_.push(cell);
    cv_.notify_one();
}

} // namespace kv_cache_manager