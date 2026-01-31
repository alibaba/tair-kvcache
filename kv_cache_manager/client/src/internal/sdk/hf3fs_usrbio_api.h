#pragma once

#include "hf3fs_usrbio.h"

namespace kv_cache_manager {

class Hf3fsUsrbioApi {
public:
    Hf3fsUsrbioApi() {}
    virtual ~Hf3fsUsrbioApi() {}

public:
    // virtual for mock test
    virtual int Hf3fsRegFd(int fd, uint64_t flags) const { return ::hf3fs_reg_fd(fd, flags); }
    virtual void Hf3fsDeregFd(int fd) const { ::hf3fs_dereg_fd(fd); }
    virtual int Hf3fsIovCreate(
        struct hf3fs_iov *iov, const char *hf3fs_mount_point, size_t size, size_t block_size, int numa) const {
        return ::hf3fs_iovcreate(iov, hf3fs_mount_point, size, block_size, numa);
    }
    virtual void Hf3fsIovDestroy(struct hf3fs_iov *iov) const { ::hf3fs_iovdestroy(iov); }
    virtual int Hf3fsIorCreate(struct hf3fs_ior *ior,
                               const char *hf3fs_mount_point,
                               int entries,
                               bool for_read,
                               int io_depth,
                               int timeout,
                               int numa,
                               uint64_t flags) const {
        return ::hf3fs_iorcreate4(ior, hf3fs_mount_point, entries, for_read, io_depth, timeout, numa, flags);
    }
    virtual void Hf3fsIorDestroy(struct hf3fs_ior *ior) const { ::hf3fs_iordestroy(ior); }
    virtual int Hf3fsPrepIo(const struct hf3fs_ior *ior,
                            const struct hf3fs_iov *iov,
                            bool read,
                            void *ptr,
                            int fd,
                            size_t off,
                            uint64_t len,
                            const void *userdata) const {
        return ::hf3fs_prep_io(ior, iov, read, ptr, fd, off, len, userdata);
    }
    virtual int Hf3fsSubmitIos(const struct hf3fs_ior *ior) const { return ::hf3fs_submit_ios(ior); }
    virtual int Hf3fsWaitForIos(const struct hf3fs_ior *ior,
                                struct hf3fs_cqe *cqes,
                                int cqec,
                                int min_results,
                                const struct timespec *abs_timeout) const {
        return ::hf3fs_wait_for_ios(ior, cqes, cqec, min_results, abs_timeout);
    }
};

} // namespace kv_cache_manager