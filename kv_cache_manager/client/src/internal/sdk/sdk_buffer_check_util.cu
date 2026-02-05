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