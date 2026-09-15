#pragma once

#include <musa_runtime.h>

#include "kv_cache_manager/common/logger.h"

#define CHECK_MUSA_ERROR(musa_call, format, args...)                                                                   \
    do {                                                                                                               \
        musaError_t err = (musa_call);                                                                                 \
        if (err != musaSuccess) {                                                                                      \
            KVCM_LOG_WARN("musa error [%d] [%s] | " format, err, musaGetErrorString(err), ##args);                     \
        }                                                                                                              \
    } while (0)

#define CHECK_MUSA_ERROR_RETURN(musa_call, return_value, format, args...)                                              \
    do {                                                                                                               \
        musaError_t err = (musa_call);                                                                                 \
        if (err != musaSuccess) {                                                                                      \
            KVCM_LOG_WARN("musa error [%d] [%s] | " format, err, musaGetErrorString(err), ##args);                     \
            return return_value;                                                                                       \
        }                                                                                                              \
    } while (0)

namespace kv_cache_manager {

class MusaBufferGuard {
public:
    MusaBufferGuard() = default;
    MusaBufferGuard(const MusaBufferGuard &) = delete;
    MusaBufferGuard &operator=(const MusaBufferGuard &) = delete;
    ~MusaBufferGuard() {
        if (ptr_ != nullptr) {
            auto err = musaFree(ptr_);
            if (err != musaSuccess) {
                KVCM_LOG_ERROR("musaFree [%p] failed in destructor: %s", ptr_, musaGetErrorString(err));
            }
        }
    }

    bool Alloc(size_t size) {
        auto err = musaMalloc(&ptr_, size);
        if (err != musaSuccess) {
            ptr_ = nullptr;
            KVCM_LOG_ERROR("musaMalloc [%lu] bytes failed: %s", size, musaGetErrorString(err));
            return false;
        }
        return true;
    }

    void *Get() const { return ptr_; }

private:
    void *ptr_ = nullptr;
};

} // namespace kv_cache_manager
