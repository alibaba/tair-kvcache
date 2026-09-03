#pragma once

#include <functional>
#include <memory>
#include <optional>

#include "kv_cache_manager/client/include/common.h"
#include "kv_cache_manager/client/src/internal/sdk/hf3fs_gpu_util_alias.h"
#include "kv_cache_manager/client/src/internal/sdk/hf3fs_usrbio_api.h"

namespace kv_cache_manager {

class Hf3fsMempool;

struct Hf3fsIovHandle {
    ::hf3fs_iov *iov{nullptr};
    std::shared_ptr<uint8_t> iov_base; // iov addr
    size_t iov_size{0};                // iov 的共享内存大小
    size_t iov_block_size{0};          // 每个共享内存块的大小, 0 表示单个大型共享内存块
    std::shared_ptr<Hf3fsMempool> iov_mempool;
    std::shared_ptr<Hf3fsGpuUtil> gpu_util;
};

struct Hf3fsIorHandle {
    ::hf3fs_ior *ior{nullptr};
    int ior_entries{1024}; // 可以提交的最大读/写请求数, 表示 io 请求个数的上限
    int ior_io_depth{0};   // ior 中的 io 深度, 表示每次提交的 io 请求数
    int ior_timeout_ms{0}; // io 批处理的最大等待时间, 仅在 io_depth 为负数时生效
};

struct Hf3fsHandle {
    Hf3fsIorHandle ior_handle;
    Hf3fsIovHandle iov_handle;
};

struct Hf3fsFileConfig {
    std::string filepath;
    std::string mountpoint;
};

class Hf3fsUsrbioClient : public std::enable_shared_from_this<Hf3fsUsrbioClient> {
public:
    Hf3fsUsrbioClient(const Hf3fsFileConfig &config,
                      const Hf3fsIovHandle &read_iov_handle,
                      const Hf3fsIovHandle &write_iov_handle,
                      const std::shared_ptr<Hf3fsUsrbioApi> &usrbio_api = nullptr);
    ~Hf3fsUsrbioClient();

public:
    bool Read(const std::vector<Iov> &iovs, int64_t deadline_ms);
    bool Write(const std::vector<Iov> &iovs, int64_t deadline_ms);

private:
    struct Segment {
        int64_t offset{0};
        int64_t len{0};
    };

    // read related
    bool DoRead(const std::vector<Iov> &iovs, int64_t deadline_ms);
    bool ReadFrom3FS(const std::shared_ptr<Hf3fsHandle> &handle,
                     const std::vector<Segment> &segments,
                     int64_t deadline_ms) const;

    // write related
    bool DoWrite(const std::vector<Iov> &iovs, int64_t deadline_ms);
    bool
    WriteTo3FS(const std::shared_ptr<Hf3fsHandle> &handle, const std::vector<Segment> &segments, int64_t deadline_ms);
    // for_read 仅用于超时/失败日志归因（read/write）
    bool
    WaitIos(const Hf3fsIorHandle &ior_handle, int32_t submit_io_count, int64_t deadline_ms, bool for_read = true) const;

    // iov/ior related
    std::vector<Segment> BuildContiguousSegments(const std::vector<Iov> &iovs) const;
    std::shared_ptr<Hf3fsHandle>
    InitIovIor(bool for_read, Hf3fsIovHandle &iov_handle, int64_t len, int32_t size_per_io, int32_t base_ior_entries);
    void ReleaseIovIor(const std::shared_ptr<Hf3fsHandle> &handle) const;
    bool CreateIor(::hf3fs_ior *&ior, bool for_read, int ior_entries, int ior_io_depth, int ior_timeout_ms) const;
    void DestroyIor(::hf3fs_ior *ior) const;
    void CopyIovs(const std::vector<Iov> &iovs, const Hf3fsIovHandle &iov_handle, bool load) const;
    int64_t CalcLeftSizeInBlock(int64_t iov_block_size, int64_t iov_offset) const;

    // file related
    bool Open(bool write = false);
    // dereg_fd=false：跳过 Hf3fsDeregFd（仍 close fd；kernel 对 in-flight I/O 持有
    // struct file 引用，close 本身安全）。已提交 I/O 的失败路径必须走 false。
    bool Close(bool dereg_fd = true);
    bool Fsync() const;
    void Del() const;
    std::optional<int64_t> FileLength() const;

private:
    const Hf3fsFileConfig config_;
    const std::string filepath_;
    Hf3fsIovHandle read_iov_handle_;
    Hf3fsIovHandle write_iov_handle_;

    int fd_{-1};
    // 提交事实标志：本 client 已向 3FS 提交过 I/O 请求后置位。此后任何失败路径
    // 都不得释放 iov/ior、也不得 DeregFd —— 3FS 无取消语义，请求可能仍在飞，
    // 引用着这些资源（泄漏优于 UAF，见 docs/design/client_sdk_io_contract.md）。
    // 判定依据是"是否提交过"而非时钟：wait 用 CLOCK_REALTIME、deadline 用
    // CLOCK_MONOTONIC，墙钟跳变会让时钟判定与真实 in-flight 状态脱节。
    mutable bool ios_submitted_{false};
    std::shared_ptr<Hf3fsUsrbioApi> usrbio_api_;

    const int32_t kDefaultReadSizePerIo{1ULL << 20};  // 1MB
    const int32_t kDefaultWriteSizePerIo{1ULL << 20}; // 1MB
};

} // namespace kv_cache_manager
