#include <limits>

#include "kv_cache_manager/client/src/internal/sdk/cuda_util.h"
#include "kv_cache_manager/client/src/internal/sdk/sdk_buffer_check_util.h"
#include "kv_cache_manager/common/env_util.h"
#include "kv_cache_manager/common/hash_util.h"
namespace kv_cache_manager {

size_t SdkBufferCheckUtil::min_cal_byte_size_ = [] {
    const int64_t configured = EnvUtil::GetEnv<int64_t>("KVCM_CHECK_IOV_BYTE_SIZE", 4);
    return configured > 0 ? static_cast<size_t>(configured) : 0;
}();

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
    // Persisted legacy checksum contract: for odd sizes this intentionally
    // skips the center byte, and the iov size is not part of the CRC input.
    // Do not "fix" this in place; introduce a versioned algorithm instead.
    const size_t current_cal_byte_size = cal_byte_size < iov.size / 2 ? cal_byte_size : iov.size / 2;
    const uint8_t *p = nullptr;
    uint8_t data;
    uint32_t crc = 0xFFFFFFFFu;
    // head data
    for (size_t i = 0; i < current_cal_byte_size; i++) {
        p = static_cast<const uint8_t *>(iov.base);
        data = *(p + i);
        crc = Crc32ByteDevice(crc, data);
    }
    // tail data
    for (size_t i = iov.size - current_cal_byte_size; i < iov.size; i++) {
        p = static_cast<const uint8_t *>(iov.base);
        data = *(p + i);
        crc = Crc32ByteDevice(crc, data);
    }

    out_crc[idx] = ~crc;
}

constexpr uint32_t kDefaultThreadsPerBlock = 512;

} // namespace

std::vector<uint32_t> SdkBufferCheckUtil::GetIovsCrc(
    const IovDevice *iovs_h_ptr, size_t iovs_size, IovDevice *iovs_d, uint32_t *crcs_d, GpuStream_t stream) {
    if (iovs_h_ptr == nullptr || iovs_d == nullptr || crcs_d == nullptr || iovs_size == 0 || min_cal_byte_size_ == 0 ||
        iovs_size > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return {};
    }
    for (size_t i = 0; i < iovs_size; ++i) {
        if (iovs_h_ptr[i].base == nullptr || iovs_h_ptr[i].size < 2) {
            return {};
        }
    }
    auto iovs_byte_size = sizeof(IovDevice) * iovs_size;
    CHECK_CUDA_ERROR_RETURN(cudaMemcpyAsync(iovs_d, iovs_h_ptr, iovs_byte_size, cudaMemcpyHostToDevice, stream),
                            {},
                            "cudaMemcpy iovs_d fail");
    int block_num = (iovs_size + kDefaultThreadsPerBlock - 1) / kDefaultThreadsPerBlock;
    GetIovsCrcDevice<<<block_num, kDefaultThreadsPerBlock, 0, stream>>>(iovs_d, iovs_size, crcs_d, min_cal_byte_size_);
    std::vector<uint32_t> crcs(iovs_size);
    auto crc_byte_size = sizeof(uint32_t) * iovs_size;
    CHECK_CUDA_ERROR_RETURN(cudaMemcpyAsync(crcs.data(), crcs_d, crc_byte_size, cudaMemcpyDeviceToHost, stream),
                            {},
                            "cudaMemcpy crcs_d fail");
    CHECK_CUDA_ERROR_RETURN(cudaStreamSynchronize(stream), {}, "cuda stream synchronize fail");
    return crcs;
}

}; // namespace kv_cache_manager
