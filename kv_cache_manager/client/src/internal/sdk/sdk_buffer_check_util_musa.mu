#include "kv_cache_manager/client/src/internal/sdk/sdk_buffer_check_util.h"
#include "kv_cache_manager/client/src/internal/sdk/musa_util.h"
#include "kv_cache_manager/common/env_util.h"

using musaStream_t = MUstream_st*;

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
    for (int i = 0; i < cal_byte_size; i++) {
        p = static_cast<const uint8_t *>(iov.base);
        data = *(p + i);
        crc = Crc32ByteDevice(crc, data);
    }
    for (int i = iov.size - cal_byte_size; i < iov.size; i++) {
        p = static_cast<const uint8_t *>(iov.base);
        data = *(p + i);
        crc = Crc32ByteDevice(crc, data);
    }
    out_crc[idx] = ~crc;
}

constexpr uint32_t kDefaultThreadsPerBlock = 512;

struct ScopedMusaStream {
    musaStream_t h = nullptr;
    ~ScopedMusaStream() {
        if (h != nullptr) {
            musaStreamDestroy(h);
        }
    }
    ScopedMusaStream(const ScopedMusaStream &) = delete;
    ScopedMusaStream &operator=(const ScopedMusaStream &) = delete;
};

}  // namespace

std::vector<uint32_t> SdkBufferCheckUtil::GetIovsCrc(
    const IovDevice *iovs_h_ptr, size_t iovs_size, IovDevice *iovs_d, uint32_t *crcs_d, GpuStream_t stream) {
    size_t cal_byte_size = std::min(min_cal_byte_size_, iovs_h_ptr->size / 2);
    if (cal_byte_size == 0) {
        return {};
    }
    ScopedMusaStream owned_tmp;
    musaStream_t musa_actual_stream;
    if (stream == nullptr) {
        musaStream_t tmp{};
        CHECK_MUSA_ERROR_RETURN(musaStreamCreateWithFlags(&tmp, 0), {}, "musaStreamCreate fail");
        owned_tmp.h = tmp;
        musa_actual_stream = tmp;
    } else {
        musa_actual_stream = static_cast<musaStream_t>(stream);
    }
    auto iovs_byte_size = sizeof(IovDevice) * iovs_size;
    CHECK_MUSA_ERROR_RETURN(
        musaMemcpyAsync(iovs_d, iovs_h_ptr, iovs_byte_size, musaMemcpyHostToDevice, musa_actual_stream), {},
        "musaMemcpy iovs_d fail");
    int block_num = (iovs_size + kDefaultThreadsPerBlock - 1) / kDefaultThreadsPerBlock;
    GetIovsCrcDevice<<<block_num, kDefaultThreadsPerBlock, 0, musa_actual_stream>>>(iovs_d, iovs_size, crcs_d,
                                                                               cal_byte_size);
    std::vector<uint32_t> crcs(iovs_size);
    auto crc_byte_size = sizeof(uint32_t) * iovs_size;
    CHECK_MUSA_ERROR_RETURN(
        musaMemcpyAsync(crcs.data(), crcs_d, crc_byte_size, musaMemcpyDeviceToHost, musa_actual_stream), {},
        "musaMemcpy crcs_d fail");
    CHECK_MUSA_ERROR_RETURN(musaStreamSynchronize(musa_actual_stream), {}, "musa stream synchronize fail");
    return crcs;
}

}  // namespace kv_cache_manager
