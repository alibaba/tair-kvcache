#pragma once

#include <gmock/gmock.h>

#include "kv_cache_manager/client/src/internal/sdk/hf3fs_usrbio_api.h"

namespace kv_cache_manager {

class MockHf3fsUsrbioApi : public Hf3fsUsrbioApi {
public:
    MOCK_METHOD(int, Hf3fsRegFd, (int fd, uint64_t flags), (const, override));
    MOCK_METHOD(void, Hf3fsDeregFd, (int fd), (const, override));

    MOCK_METHOD(int,
                Hf3fsIovCreate,
                (::hf3fs_iov * iov, const char *mount, size_t size, size_t block_size, int numa),
                (const, override));
    MOCK_METHOD(void, Hf3fsIovDestroy, (::hf3fs_iov * iov), (const, override));

    MOCK_METHOD(int,
                Hf3fsIorCreate,
                (::hf3fs_ior * ior,
                 const char *mount,
                 int entries,
                 bool for_read,
                 int io_depth,
                 int timeout,
                 int numa,
                 uint64_t flags),
                (const, override));
    MOCK_METHOD(void, Hf3fsIorDestroy, (::hf3fs_ior * ior), (const, override));

    MOCK_METHOD(int,
                Hf3fsPrepIo,
                (const ::hf3fs_ior *ior,
                 const ::hf3fs_iov *iov,
                 bool read,
                 void *ptr,
                 int fd,
                 size_t off,
                 uint64_t len,
                 const void *userdata),
                (const, override));
    MOCK_METHOD(int, Hf3fsSubmitIos, (const ::hf3fs_ior *ior), (const, override));
    MOCK_METHOD(int,
                Hf3fsWaitForIos,
                (const ::hf3fs_ior *ior, ::hf3fs_cqe *cqes, int cqec, int min_results, const struct timespec *abs),
                (const, override));
};

} // namespace kv_cache_manager
