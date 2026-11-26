#include "kv_cache_manager/client/src/internal/sdk/sdk_buffer_check_util.h"

#include <memory>
#include <utility>

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
    ScopedMusaStream() = default;
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
    return GetIovsCrc(iovs_h_ptr, iovs_size, iovs_d, crcs_d, stream, nullptr);
}

std::vector<uint32_t> SdkBufferCheckUtil::GetIovsCrc(
    const IovDevice *iovs_h_ptr, size_t iovs_size, IovDevice *iovs_d, uint32_t *crcs_d, GpuStream_t stream,
    bool *caller_read_pending) {
    if (caller_read_pending) *caller_read_pending = false;
    size_t cal_byte_size = CrcSampleBytes(iovs_h_ptr->size);
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
    // Legacy overloads accept a caller-owned table; keep its upload source alive.
    std::unique_ptr<std::vector<IovDevice>> owned_iovs;
    if (!caller_read_pending) {
        owned_iovs = std::make_unique<std::vector<IovDevice>>(iovs_h_ptr, iovs_h_ptr + iovs_size);
        iovs_h_ptr = owned_iovs->data();
    }
    auto crcs = std::make_unique<std::vector<uint32_t>>(iovs_size);
    const auto iovs_byte_size = sizeof(IovDevice) * iovs_size;
    const auto upload_error = musaMemcpyAsync(
        iovs_d, iovs_h_ptr, iovs_byte_size, musaMemcpyHostToDevice, musa_actual_stream);
    auto copy_error = upload_error;
    if (upload_error == musaSuccess) {
        if (caller_read_pending) *caller_read_pending = true;
        int block_num = (iovs_size + kDefaultThreadsPerBlock - 1) / kDefaultThreadsPerBlock;
        GetIovsCrcDevice<<<block_num, kDefaultThreadsPerBlock, 0, musa_actual_stream>>>(
            iovs_d, iovs_size, crcs_d, cal_byte_size);
        copy_error = musaMemcpyAsync(
            crcs->data(), crcs_d, sizeof(uint32_t) * iovs_size, musaMemcpyDeviceToHost, musa_actual_stream);
    }
    // Copy errors can report earlier asynchronous failures. Fence before releasing either host buffer.
    const auto sync_error = musaStreamSynchronize(musa_actual_stream);
    if (sync_error != musaSuccess) {
        if (upload_error == musaSuccess) (void)crcs.release();
        (void)owned_iovs.release();
        CHECK_MUSA_ERROR_RETURN(sync_error, {}, "SDK buffer check fence failed; retaining pending host buffers");
    }
    if (caller_read_pending) *caller_read_pending = false;
    CHECK_MUSA_ERROR_RETURN(copy_error, {}, "asynchronous SDK buffer check copy failed");
    return std::move(*crcs);
}

}  // namespace kv_cache_manager
