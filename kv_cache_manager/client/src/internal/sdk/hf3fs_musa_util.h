#pragma once

#include "kv_cache_manager/common/logger.h"

#ifdef USING_MUSA
#include <musa_runtime.h>
#define CHECK_MUSA_ERROR_HF3FS(musa_call)                                                                              \
    do {                                                                                                               \
        musaError_t err = (musa_call);                                                                                 \
        if (err != musaSuccess) {                                                                                      \
            KVCM_LOG_ERROR("%s failed, err: %s ", #musa_call, musaGetErrorString(err));                                \
            throw std::runtime_error("musa runtime error: " + std::string(musaGetErrorString(err)));                   \
        }                                                                                                              \
    } while (0)
#endif

namespace kv_cache_manager {

#ifdef USING_MUSA

class Hf3fsMusaUtil final {
public:
    Hf3fsMusaUtil() = default;
    ~Hf3fsMusaUtil() { Release(); }

public:
    bool Init() {
        auto err = musaStreamCreateWithFlags(&musa_stream_, musaStreamNonBlocking);
        if (err != musaSuccess) {
            KVCM_LOG_WARN("init failed, create musa stream failed, err: %s", musaGetErrorString(err));
            return false;
        }
        return true;
    }

    void Release() {
        if (musa_stream_) {
            CHECK_MUSA_ERROR_HF3FS(musaStreamDestroy(musa_stream_));
            musa_stream_ = nullptr;
        }
    }

    void CopyAsyncHostToDevice(void *dst, const void *src, size_t size) {
        CHECK_MUSA_ERROR_HF3FS(musaMemcpyAsync(dst, src, size, musaMemcpyHostToDevice, musa_stream_));
    }

    void CopyAsyncDeviceToHost(void *dst, const void *src, size_t size) {
        CHECK_MUSA_ERROR_HF3FS(musaMemcpyAsync(dst, src, size, musaMemcpyDeviceToHost, musa_stream_));
    }

    void Sync() { CHECK_MUSA_ERROR_HF3FS(musaStreamSynchronize(musa_stream_)); }

    bool RegisterHost(void *ptr, size_t size) const {
        if (ptr == nullptr || size == 0) {
            return false;
        }
        auto err = musaHostRegister(ptr, size, musaHostRegisterDefault);
        if (err != musaSuccess) {
            KVCM_LOG_WARN("musa host register failed, err: %s, ptr: %p, size: %zu", musaGetErrorString(err), ptr, size);
            return false;
        }
        return true;
    }

    void UnregisterHost(void *ptr) const {
        if (ptr != nullptr) {
            CHECK_MUSA_ERROR_HF3FS(musaHostUnregister(ptr));
        }
    }

private:
    musaStream_t musa_stream_{nullptr};
};

#endif

} // namespace kv_cache_manager
