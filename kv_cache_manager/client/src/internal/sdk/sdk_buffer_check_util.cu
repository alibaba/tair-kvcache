#include <algorithm>
#include <cassert>

#include "kv_cache_manager/client/src/internal/sdk/sdk_buffer_check_util.h"
#include "kv_cache_manager/common/env_util.h"
#include "kv_cache_manager/common/hash_util.h"
namespace kv_cache_manager {

size_t SdkBufferCheckUtil::min_cal_byte_size_ = EnvUtil::GetEnv("KVCM_CHECK_IOV_BYTE_SIZE", 4);

namespace {

__device__ __forceinline__ uint32_t Crc32ByteDevice(uint32_t crc, uint8_t data) {
    crc ^= data;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint32_t mask = -(crc & 1u);
        crc = (crc >> 1) ^ (0xEDB88320u & mask);
    }

    return crc;
}

__global__ void GetIovsCrcDevice(const IovDevice *iovs, int iovs_size, uint32_t *out_crc, size_t cal_byte_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= iovs_size) {
        return;
    }
    const auto &iov = iovs[idx];
    const uint8_t *p = nullptr;
    uint8_t data;
    uint32_t crc = 0xFFFFFFFFu;
    // head data
    for (int i = 0; i < cal_byte_size; i++) {
        p = static_cast<const uint8_t *>(iov.base);
        data = *(p + i);
        crc = Crc32ByteDevice(crc, data);
    }
    // tail data
    for (int i = iov.size - cal_byte_size; i < iov.size; i++) {
        p = static_cast<const uint8_t *>(iov.base);
        data = *(p + i);
        crc = Crc32ByteDevice(crc, data);
    }

    out_crc[idx] = ~crc;
}

constexpr uint32_t kDefaultThreadsPerBlock = 512;

} // namespace

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

std::vector<uint32_t> SdkBufferCheckUtil::GetIovsCrc(
    const IovDevice *iovs_h_ptr, size_t iovs_size, IovDevice *iovs_d, uint32_t *crcs_d, cudaStream_t stream) {
    size_t cal_byte_size = std::min(min_cal_byte_size_, iovs_h_ptr->size / 2);
    if (cal_byte_size == 0) {
        return {};
    }
    auto iovs_byte_size = sizeof(IovDevice) * iovs_size;
    CHECK_CUDA_ERROR_RETURN(cudaMemcpyAsync(iovs_d, iovs_h_ptr, iovs_byte_size, cudaMemcpyHostToDevice, stream),
                            {},
                            "cudaMemcpy iovs_d fail");
    int block_num = (iovs_size + kDefaultThreadsPerBlock - 1) / kDefaultThreadsPerBlock;
    GetIovsCrcDevice<<<block_num, kDefaultThreadsPerBlock, 0, stream>>>(iovs_d, iovs_size, crcs_d, cal_byte_size);
    std::vector<uint32_t> crcs(iovs_size);
    auto crc_byte_size = sizeof(uint32_t) * iovs_size;
    CHECK_CUDA_ERROR_RETURN(cudaMemcpyAsync(crcs.data(), crcs_d, crc_byte_size, cudaMemcpyDeviceToHost, stream),
                            {},
                            "cudaMemcpy crcs_d fail");
    CHECK_CUDA_ERROR_RETURN(cudaStreamSynchronize(stream), {}, "cuda stream synchronize fail");
    return crcs;
}

}; // namespace kv_cache_manager