#pragma once

#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <string>
#include <sys/file.h>
#include <unistd.h>
#include <utility>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/config/distributed_lock_backend.h"

namespace kv_cache_manager {

class DistributedLockFileBackend : public DistributedLockBackend {
public:
    DistributedLockFileBackend() = default;
    ~DistributedLockFileBackend() override;

public:
    ErrorCode Init(const StandardUri &standard_uri) noexcept override;
    ErrorCode TryLock(const std::string &lock_key, const std::string &lock_value, int64_t ttl_ms) override;
    ErrorCode RenewLock(const std::string &lock_key, const std::string &lock_value, int64_t ttl_ms) override;
    ErrorCode Unlock(const std::string &lock_key, const std::string &lock_value) override;
    ErrorCode
    GetLockHolder(const std::string &lock_key, std::string &out_current_value, int64_t &out_expire_time_ms) override;

private:
    // RAII 类，用于管理文件锁和文件描述符
    class FileLockGuard {
    public:
        // 构造函数，接受文件路径
        explicit FileLockGuard(const std::string &file_path, bool create_if_not_exist = true)
            : file_path_(file_path), create_if_not_exist_(create_if_not_exist) {}

        // 移动构造函数
        FileLockGuard(FileLockGuard &&other) noexcept
            : file_path_(std::move(other.file_path_))
            , create_if_not_exist_(other.create_if_not_exist_)
            , fd_(other.fd_)
            , locked_(other.locked_) {
            other.fd_ = -1;
            other.locked_ = false;
        }

        // 移动赋值运算符
        FileLockGuard &operator=(FileLockGuard &&other) noexcept {
            if (this != &other) {
                Release(); // 释放当前资源
                file_path_ = std::move(other.file_path_);
                create_if_not_exist_ = other.create_if_not_exist_;
                fd_ = other.fd_;
                locked_ = other.locked_;
                other.fd_ = -1;
                other.locked_ = false;
            }
            return *this;
        }

        // 禁止拷贝
        FileLockGuard(const FileLockGuard &) = delete;
        FileLockGuard &operator=(const FileLockGuard &) = delete;

        // 析构函数，自动释放锁并关闭文件描述符
        ~FileLockGuard() { Release(); }

        // 获取文件描述符
        int fd() const { return fd_; }

        // 检查是否持有锁
        bool is_locked() const { return locked_; }

        // 获取文件路径
        const std::string &file_path() const { return file_path_; }

        // 尝试获取锁（非阻塞）
        ErrorCode TryLock(bool &locked) {
            if (fd_ < 0) {
                // 打开文件
                int flags = O_RDWR;
                if (create_if_not_exist_) {
                    flags |= O_CREAT;
                }
                fd_ = open(file_path_.c_str(), flags, 0644);
                if (fd_ < 0) {
                    if (errno == ENOENT) {
                        return EC_NOENT;
                    } else {
                        KVCM_LOG_ERROR("failed to open lock file: %s, error: %s, errno:%d, flags: %d",
                                       file_path_.c_str(),
                                       strerror(errno),
                                       errno,
                                       flags);
                        return EC_IO_ERROR;
                    }
                }
            }

            // 非阻塞获取文件锁
            int result = flock(fd_, LOCK_EX | LOCK_NB);
            if (result == 0) {
                locked_ = true;
                locked = true;
                return EC_OK;
            } else if (errno == EWOULDBLOCK) {
                locked = false;
                return EC_OK;
            } else {
                KVCM_LOG_ERROR("flock failed with error: %s", strerror(errno));
                locked = false;
                return EC_IO_ERROR;
            }
        }

        // 阻塞获取锁
        ErrorCode Lock() {
            if (fd_ < 0) {
                // 打开文件
                int flags = O_RDWR;
                if (create_if_not_exist_) {
                    flags |= O_CREAT;
                }
                fd_ = open(file_path_.c_str(), flags, 0644);
                if (fd_ < 0) {
                    if (errno == ENOENT) {
                        return EC_NOENT;
                    } else {
                        KVCM_LOG_ERROR("failed to open lock file: %s, error: %s, errno:%d, flags: %d",
                                       file_path_.c_str(),
                                       strerror(errno),
                                       errno,
                                       flags);
                        return EC_IO_ERROR;
                    }
                }
            }

            // 阻塞获取文件锁
            int result = flock(fd_, LOCK_EX);
            if (result == 0) {
                locked_ = true;
                return EC_OK;
            } else {
                KVCM_LOG_ERROR("flock failed with error: %s", strerror(errno));
                return EC_IO_ERROR;
            }
        }

        // 手动释放锁和文件描述符
        void Release() {
            if (fd_ >= 0) {
                if (locked_) {
                    flock(fd_, LOCK_UN);
                }
                close(fd_);
                fd_ = -1;
                locked_ = false;
            }
        }

        // 分离文件描述符（调用者负责管理）
        int Detach() {
            int fd = fd_;
            locked_ = false;
            fd_ = -1;
            return fd;
        }

    private:
        std::string file_path_;
        bool create_if_not_exist_;
        int fd_{-1};
        bool locked_{false};
    };

private:
    // 获取锁文件的完整路径
    std::string GetLockFilePath(const std::string &lock_key) const;

    // 读取锁文件内容
    ErrorCode ReadLockFileContent(int fd, std::string &content);

    // 写入锁文件内容
    ErrorCode WriteLockFileContent(int fd, const std::string &content);

    // 检查锁是否过期
    bool IsLockExpired(const std::string &lock_content, int64_t &expire_time_ms);

    // 序列化锁内容（value:expire_time_ms格式）
    std::string SerializeLockContent(const std::string &lock_value, int64_t expire_time_ms) const;

    // 解析锁内容，返回是否成功解析出value和expire_time_ms
    bool
    ParseLockContent(const std::string &lock_content, std::string &out_lock_value, int64_t &out_expire_time_ms) const;

private:
    std::string lock_dir_path_;
};

} // namespace kv_cache_manager