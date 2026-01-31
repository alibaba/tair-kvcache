#include "kv_cache_manager/client/src/internal/sdk/hf3fs_usrbio_client.h"

#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <numeric>
#include <optional>
#include <sys/stat.h>
#include <unistd.h>

#include "kv_cache_manager/client/src/internal/sdk/hf3fs_cuda_util.h"
#include "kv_cache_manager/client/src/internal/sdk/hf3fs_mempool.h"
#include "kv_cache_manager/client/src/internal/sdk/hf3fs_usrbio_api.h"
#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

Hf3fsUsrbioClient::Hf3fsUsrbioClient(const Hf3fsFileConfig &config,
                                     const Hf3fsIovHandle &read_iov_handle,
                                     const Hf3fsIovHandle &write_iov_handle,
                                     const std::shared_ptr<Hf3fsUsrbioApi> &usrbio_api)
    : config_(config)
    , filepath_(config.filepath)
    , read_iov_handle_(read_iov_handle)
    , write_iov_handle_(write_iov_handle)
    , usrbio_api_(usrbio_api) {
    if (usrbio_api_ == nullptr) {
        usrbio_api_ = std::make_shared<Hf3fsUsrbioApi>();
    }
}

Hf3fsUsrbioClient::~Hf3fsUsrbioClient() {
    Close();
    usrbio_api_.reset();
}

bool Hf3fsUsrbioClient::Read(const std::vector<Iov> &iovs) {
    int64_t read_len = 0;
    int64_t total_len = 0;
    for (const auto &iov : iovs) {
        total_len += iov.size;
        read_len += iov.ignore ? 0 : iov.size;
        if (!iov.ignore && iov.size == 0) {
            KVCM_LOG_WARN("read failed, iov size is 0, file: %s, iov count: %zu", filepath_.c_str(), iovs.size());
            return false;
        }
    }
    if (read_len <= 0) {
        KVCM_LOG_WARN("read but read len is 0, file: %s, iovs size: %zu", filepath_.c_str(), iovs.size());
        return true;
    }
    if (const auto file_len_opt = FileLength(); !file_len_opt.has_value()) {
        KVCM_LOG_WARN("read failed, get file length failed, file: %s", filepath_.c_str());
        return false;
    } else if (total_len > file_len_opt.value()) {
        KVCM_LOG_WARN("read failed, iovs len exceed file len, file: %s, file len: %ld, total len: %ld, read len: %ld",
                      filepath_.c_str(),
                      file_len_opt.value(),
                      total_len,
                      read_len);
        return false;
    }

    if (!Open()) {
        return false;
    }

    return DoRead(iovs);
}

bool Hf3fsUsrbioClient::DoRead(const std::vector<Iov> &iovs) {
    const auto segments = BuildContiguousSegments(iovs);
    if (segments.empty()) {
        KVCM_LOG_WARN("do read failed, segments are empty, file: %s, iovs size: %zu", filepath_.c_str(), iovs.size());
        return false;
    }

    const int64_t read_len = std::accumulate(
        segments.begin(), segments.end(), 0, [](int64_t len, const Segment &seg) { return len + seg.len; });
    auto handle =
        InitIovIor(true, read_iov_handle_, read_len, kDefaultReadSizePerIo, static_cast<int32_t>(segments.size()));
    if (handle == nullptr) {
        KVCM_LOG_WARN("read failed, init iov/ior failed, file: %s, len: %ld", filepath_.c_str(), read_len);
        return false;
    }

    if (!ReadFrom3FS(handle, segments)) {
        ReleaseIovIor(handle);
        return false;
    }

    CopyIovs(iovs, handle->iov_handle, true);
    ReleaseIovIor(handle);
    return true;
}

bool Hf3fsUsrbioClient::ReadFrom3FS(const std::shared_ptr<Hf3fsHandle> &handle,
                                    const std::vector<Segment> &segments) const {
    auto &ior = handle->ior_handle.ior;
    auto &iov = handle->iov_handle.iov;
    auto iov_base = handle->iov_handle.iov_base.get();
    const auto ior_entries = handle->ior_handle.ior_entries;
    const auto iov_block_size = handle->iov_handle.iov_block_size;
    int64_t iov_offset = 0;
    int32_t submit_io_count = 0;
    bool read_success = true;

    for (int i = 0; i < segments.size(); ++i) {
        const auto &seg = segments[i];
        int64_t remaining_len = seg.len;
        int64_t file_offset = seg.offset;

        while (remaining_len > 0) {
            uint64_t step_len = 0;
            if (iov_block_size > 0) {
                step_len = std::min(remaining_len, CalcLeftSizeInBlock(iov_block_size, iov_offset));
            } else {
                step_len = remaining_len > kDefaultReadSizePerIo ? kDefaultReadSizePerIo : remaining_len;
            }

            auto ret =
                usrbio_api_->Hf3fsPrepIo(ior, iov, true, iov_base + iov_offset, fd_, file_offset, step_len, nullptr);
            if (ret < 0) {
                KVCM_LOG_WARN(
                    "read from 3fs failed, 3fs prep io failed, errno: %s, file: %s, fd: %d, submit_io_count: %d, "
                    "ior_entries: %d, step_len: %lu, iov_block_size: %lu, remaining_len: %ld",
                    strerror(-ret),
                    filepath_.c_str(),
                    fd_,
                    submit_io_count,
                    ior_entries,
                    step_len,
                    iov_block_size,
                    remaining_len);
                read_success = false;
                break;
            }
            ++submit_io_count;
            iov_offset += step_len;
            file_offset += step_len;
            remaining_len -= step_len;

            const bool last_io = (i + 1 == static_cast<int>(segments.size())) && (remaining_len <= 0);
            if (submit_io_count < ior_entries && !last_io) {
                continue;
            }

            ret = usrbio_api_->Hf3fsSubmitIos(ior);
            if (ret != 0) {
                KVCM_LOG_WARN("read from 3fs failed, 3fs submit ios failed, errno: %s, file: %s, submit_io_count: "
                              "%d, ior_entries: %d",
                              strerror(-ret),
                              filepath_.c_str(),
                              submit_io_count,
                              ior_entries);
                read_success = false;
                break;
            }

            // submit_io_count 达到最大或者没得读
            if (!WaitIos(handle->ior_handle, submit_io_count)) {
                read_success = false;
                break;
            }
            submit_io_count = 0; // reset
        }

        if (!read_success) {
            break;
        }
    }

    return read_success;
}

bool Hf3fsUsrbioClient::Write(const std::vector<Iov> &iovs) {
    const int64_t write_len = std::accumulate(
        iovs.begin(), iovs.end(), 0, [](int64_t len, const Iov &iov) { return len + (iov.ignore ? 0 : iov.size); });
    if (write_len <= 0) {
        KVCM_LOG_WARN("write len is 0, file: %s, iovs size: %zu", filepath_.c_str(), iovs.size());
        return true;
    }

    if (!Open(true)) {
        return false;
    }

    if (!DoWrite(iovs)) {
        Close();
        return false;
    }

    Fsync();
    Close();
    return true;
}

bool Hf3fsUsrbioClient::DoWrite(const std::vector<Iov> &iovs) {
    const auto segments = BuildContiguousSegments(iovs);
    if (segments.empty()) {
        KVCM_LOG_WARN("do write failed, segments are empty, file: %s, iovs size: %zu", filepath_.c_str(), iovs.size());
        return false;
    }

    const int64_t write_len = std::accumulate(
        segments.begin(), segments.end(), 0, [](int64_t len, const Segment &seg) { return len + seg.len; });
    auto handle =
        InitIovIor(false, write_iov_handle_, write_len, kDefaultWriteSizePerIo, static_cast<int32_t>(segments.size()));
    if (handle == nullptr) {
        KVCM_LOG_WARN("do write failed, init iov/ior failed, file: %s, len: %ld", filepath_.c_str(), write_len);
        return false;
    }

    CopyIovs(iovs, handle->iov_handle, false);

    bool success = WriteTo3FS(handle, segments);
    ReleaseIovIor(handle);
    return success;
}

bool Hf3fsUsrbioClient::WriteTo3FS(const std::shared_ptr<Hf3fsHandle> &handle,
                                   const std::vector<Hf3fsUsrbioClient::Segment> &segments) {
    auto &ior = handle->ior_handle.ior;
    auto &iov = handle->iov_handle.iov;
    auto iov_base = handle->iov_handle.iov_base.get();
    const auto iov_block_size = handle->iov_handle.iov_block_size;
    const auto ior_entries = handle->ior_handle.ior_entries;
    int64_t iov_offset = 0;
    int submit_io_count = 0;
    bool write_success = true;

    for (int i = 0; i < segments.size(); ++i) {
        const auto &seg = segments[i];
        int64_t remaining_len = seg.len;
        int64_t file_offset = seg.offset;

        while (remaining_len > 0) {
            uint64_t step_len = 0;
            if (iov_block_size > 0) {
                step_len = std::min<int64_t>(remaining_len, CalcLeftSizeInBlock(iov_block_size, iov_offset));
            } else {
                step_len = remaining_len > kDefaultWriteSizePerIo ? kDefaultWriteSizePerIo : remaining_len;
            }

            auto ret =
                usrbio_api_->Hf3fsPrepIo(ior, iov, false, iov_base + iov_offset, fd_, file_offset, step_len, nullptr);
            if (ret < 0) {
                KVCM_LOG_WARN("write to 3fs failed, 3fs prep io failed, errno: %s, file: %s, submit_io_count: %d, "
                              "ior_entries: %d, step_len: %lu, iov_block_size: %lu, remaining_len: %ld",
                              strerror(-ret),
                              filepath_.c_str(),
                              submit_io_count,
                              ior_entries,
                              step_len,
                              iov_block_size,
                              remaining_len);
                write_success = false;
                break;
            }
            ++submit_io_count;
            iov_offset += step_len;
            file_offset += step_len;
            remaining_len -= step_len;

            const bool last_io = (i + 1 == segments.size()) && (remaining_len <= 0);
            if (submit_io_count < ior_entries && !last_io) {
                continue;
            }

            ret = usrbio_api_->Hf3fsSubmitIos(ior);
            if (ret != 0) {
                KVCM_LOG_WARN("write to 3fs failed, 3fs submit ios failed, errno: %s, file: %s, submit_io_count: %d, "
                              "ior_entries: %d",
                              strerror(-ret),
                              filepath_.c_str(),
                              submit_io_count,
                              ior_entries);
                write_success = false;
                break;
            }

            if (!WaitIos(handle->ior_handle, submit_io_count)) {
                write_success = false;
                break;
            }

            submit_io_count = 0; // reset
        }

        if (!write_success) {
            break;
        }
    }

    return write_success;
}

bool Hf3fsUsrbioClient::WaitIos(const Hf3fsIorHandle &ior_handle, int32_t submit_io_count) const {
    if (ior_handle.ior == nullptr) {
        return false;
    }

    hf3fs_cqe cqes[submit_io_count];
    auto ior = ior_handle.ior;
    const auto ior_entries = ior_handle.ior_entries;

    int completed_io_count = usrbio_api_->Hf3fsWaitForIos(ior, cqes, submit_io_count, submit_io_count, nullptr);
    if (completed_io_count < 0) {
        KVCM_LOG_WARN("wait io failed, 3fs wait for ios failed, errno: %s, file: %s, submit ios: %d, ior entries: %d",
                      strerror(-completed_io_count),
                      filepath_.c_str(),
                      submit_io_count,
                      ior_entries);
        return false;
    }

    for (int i = 0; i < completed_io_count; ++i) {
        if (cqes[i].result < 0) {
            KVCM_LOG_WARN(
                "wait io failed, cqe result errno: %s, file: %s, submit ios: %d, completed ios: %d, ior entries: %d",
                strerror(-cqes[i].result),
                filepath_.c_str(),
                submit_io_count,
                completed_io_count,
                ior_entries);
            return false;
        }
    }

    if (completed_io_count != submit_io_count) {
        KVCM_LOG_WARN("wait io failed, file: %s, submit ios: %d, completed ios: %d, ior entries: %d",
                      filepath_.c_str(),
                      submit_io_count,
                      completed_io_count,
                      ior_entries);
        return false;
    }

    // all ios done
    return true;
}

std::vector<Hf3fsUsrbioClient::Segment> Hf3fsUsrbioClient::BuildContiguousSegments(const std::vector<Iov> &iovs) const {
    std::vector<Hf3fsUsrbioClient::Segment> segments;

    int64_t file_offset = 0;
    int64_t seg_offset = 0;
    int64_t seg_len = 0;
    std::optional<MemoryType> seg_type;

    for (const auto &iov : iovs) {
        if (iov.ignore) {
            if (iov.size == 0) {
                continue;
            }
            if (seg_len > 0) {
                segments.push_back({seg_offset, seg_len});
                seg_offset = 0;
                seg_len = 0;
                seg_type.reset();
            }
            file_offset += iov.size;
            continue;
        }
        if (seg_len == 0) {
            seg_offset = file_offset;
            seg_len = iov.size;
            seg_type = iov.type;
        } else {
            if (seg_type.has_value() && iov.type != seg_type.value()) {
                segments.push_back({seg_offset, seg_len});
                seg_offset = file_offset;
                seg_len = iov.size;
                seg_type = iov.type;
            } else {
                seg_len += iov.size;
            }
        }
        file_offset += iov.size;
    }
    if (seg_len > 0) {
        segments.push_back({seg_offset, seg_len});
    }
    return segments;
}

std::shared_ptr<Hf3fsHandle> Hf3fsUsrbioClient::InitIovIor(
    bool for_read, Hf3fsIovHandle &iov_handle, int64_t len, int32_t size_per_io, int32_t base_ior_entries) {
    if (iov_handle.iov_mempool == nullptr) {
        KVCM_LOG_WARN("init iov/ior failed, iov mempool is nullptr, read: %d, file: %s", for_read, filepath_.c_str());
        return nullptr;
    }

    auto iov_buffer = iov_handle.iov_mempool->Alloc(len);
    if (iov_buffer == nullptr) {
        KVCM_LOG_WARN("init iov/ior failed, mempool alloc failed, read: %d, len: %zu, mempool free size: %zu, file: %s",
                      for_read,
                      len,
                      iov_handle.iov_mempool->FreeSize(),
                      filepath_.c_str());
        return nullptr;
    }

    Hf3fsIorHandle ior_handle;
    const int32_t iov_block_size = iov_handle.iov_block_size != 0 ? iov_handle.iov_block_size : size_per_io;
    ior_handle.ior_entries = len / iov_block_size + base_ior_entries;

    struct hf3fs_ior *ior{nullptr};
    if (!CreateIor(ior, for_read, ior_handle.ior_entries, ior_handle.ior_io_depth, ior_handle.ior_timeout_ms)) {
        KVCM_LOG_WARN(
            "init iov/ior failed, create ior failed, file: %s, ior entries: %d, ior io depth: %d, ior timeout ms: %d",
            filepath_.c_str(),
            ior_handle.ior_entries,
            ior_handle.ior_io_depth,
            ior_handle.ior_timeout_ms);
        iov_handle.iov_mempool->Free(iov_buffer);
        return nullptr;
    }

    auto handle = std::make_shared<Hf3fsHandle>();
    handle->iov_handle = iov_handle;
    handle->iov_handle.iov_size = len;
    handle->iov_handle.iov_base = std::shared_ptr<uint8_t>(static_cast<uint8_t *>(iov_buffer), [](void *) {});
    handle->ior_handle = ior_handle;
    handle->ior_handle.ior = ior;
    return handle;
}

void Hf3fsUsrbioClient::ReleaseIovIor(const std::shared_ptr<Hf3fsHandle> &handle) const {
    if (handle == nullptr) {
        return;
    }

    auto &iov_handle = handle->iov_handle;
    if (iov_handle.iov_base && iov_handle.iov_mempool) {
        iov_handle.iov_mempool->Free(iov_handle.iov_base.get());
    }
    // iov_handle.iov      = nullptr;
    iov_handle.iov_base = nullptr;

    DestroyIor(handle->ior_handle.ior);
    handle->ior_handle.ior = nullptr;
}

bool Hf3fsUsrbioClient::CreateIor(
    struct hf3fs_ior *&ior, bool for_read, int ior_entries, int ior_io_depth, int ior_timeout_ms) const {
    if (config_.mountpoint.empty()) {
        KVCM_LOG_WARN("create ior failed, mountpoint is empty");
        return false;
    }

    ior = new struct hf3fs_ior();
    auto ret = usrbio_api_->Hf3fsIorCreate(
        ior, config_.mountpoint.c_str(), ior_entries, for_read, ior_io_depth, ior_timeout_ms, -1, 0);
    if (ret != 0) {
        KVCM_LOG_WARN(
            "3fs ior create failed, read: %d, errno: %s, mountpoint: %s, entries: %d, io depth: %d, timeout: %d",
            for_read,
            strerror(-ret),
            config_.mountpoint.c_str(),
            ior_entries,
            ior_io_depth,
            ior_timeout_ms);
        delete ior;
        ior = nullptr;
        return false;
    }
    return true;
}

void Hf3fsUsrbioClient::DestroyIor(struct hf3fs_ior *ior) const {
    if (ior != nullptr) {
        usrbio_api_->Hf3fsIorDestroy(ior);
        delete ior;
    }
}

void Hf3fsUsrbioClient::CopyIovs(const std::vector<Iov> &iovs, const Hf3fsIovHandle &iov_handle, bool load) const {
    int64_t iov_offset = 0;
    auto iov_base = iov_handle.iov_base.get();
    auto cuda_util = iov_handle.cuda_util;

    for (const auto &iov : iovs) {
        if (iov.ignore) {
            continue;
        }
        if (load) {
            if (iov.type == MemoryType::GPU) {
                cuda_util->CopyAsyncHostToDevice(iov.base, iov_base + iov_offset, iov.size);
            } else {
                ::memcpy(iov.base, iov_base + iov_offset, iov.size);
            }
        } else {
            if (iov.type == MemoryType::GPU) {
                cuda_util->CopyAsyncDeviceToHost(iov_base + iov_offset, iov.base, iov.size);
            } else {
                ::memcpy(iov_base + iov_offset, iov.base, iov.size);
            }
        }
        iov_offset += iov.size;
    }
    if (cuda_util) {
        cuda_util->Sync();
    }
}

int64_t Hf3fsUsrbioClient::CalcLeftSizeInBlock(int64_t iov_block_size, int64_t iov_offset) const {
    // 计算当前 iov block 块剩余可用的大小, 避免跨 block 读写
    const int64_t block_start = (iov_offset / iov_block_size) * iov_block_size;
    const int64_t block_end = block_start + iov_block_size;
    const int64_t left_size_in_block = block_end - iov_offset;
    return left_size_in_block;
}

bool Hf3fsUsrbioClient::Open(bool write) {
    if (write) { // assume write only trigger once
        Close();
    }

    if (fd_ != -1) {
        KVCM_LOG_DEBUG("file already opened, file: %s, write: %d", filepath_.c_str(), write);
        return true;
    }

    int flags = O_RDWR;
    if (write) {
        flags |= O_CREAT;
    }

    int fd = ::open(filepath_.c_str(), flags, 0666);
    if (fd == -1) {
        KVCM_LOG_WARN(
            "open file failed, file: %s, write: %d, fd: %d, errno: %s", filepath_.c_str(), write, fd, strerror(errno));
        return false;
    }

    auto ret = usrbio_api_->Hf3fsRegFd(fd, 0);
    if (ret > 0) {
        KVCM_LOG_WARN(
            "3fs reg fd failed, file: %s, write: %d, fd: %d, errno: %s", filepath_.c_str(), write, fd, strerror(ret));
        Close(false);
        // if (write) {
        //     Del();
        // }
        return false;
    }
    fd_ = fd;
    return true;
}

bool Hf3fsUsrbioClient::Close(bool dereg_fd) {
    if (fd_ != -1) {
        if (dereg_fd) {
            usrbio_api_->Hf3fsDeregFd(fd_);
        }
        ::close(fd_);
        fd_ = -1;
    }
    return true;
}

void Hf3fsUsrbioClient::Del() const {
    KVCM_LOG_DEBUG("remove 3fs file: %s", filepath_.c_str());
    ::remove(filepath_.c_str());
}

bool Hf3fsUsrbioClient::Fsync() const {
    if (fd_ == -1) {
        return false;
    }
    if (::fsync(fd_) != 0) {
        KVCM_LOG_WARN("fsync failed, errno: %s, file: %s", strerror(errno), filepath_.c_str());
        return false;
    }
    return true;
}

std::optional<int64_t> Hf3fsUsrbioClient::FileLength() const {
    struct stat file_stat;
    if (auto ret = ::stat(filepath_.c_str(), &file_stat); ret != 0) {
        KVCM_LOG_WARN("get file length failed, stat failed, file: %s, ret: %d, errno: %s",
                      filepath_.c_str(),
                      ret,
                      strerror(errno));
        return std::nullopt;
    }
    return static_cast<int64_t>(file_stat.st_size); // byte
}

} // namespace kv_cache_manager
