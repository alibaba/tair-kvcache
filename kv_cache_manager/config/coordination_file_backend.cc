#include "kv_cache_manager/config/coordination_file_backend.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <string>
#include <unistd.h>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/standard_uri.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

namespace kv_cache_manager {

CoordinationFileBackend::~CoordinationFileBackend() {}

ErrorCode CoordinationFileBackend::Init(const StandardUri &standard_uri) noexcept {
    lock_dir_path_ = standard_uri.GetPath();

    if (lock_dir_path_.empty()) {
        KVCM_LOG_ERROR("lock directory path is empty");
        return EC_BADARGS;
    }

    // 确保锁目录存在
    std::error_code ec;
    if (!std::filesystem::exists(lock_dir_path_, ec)) {
        if (!std::filesystem::create_directories(lock_dir_path_, ec)) {
            KVCM_LOG_ERROR(
                "failed to create lock directory: %s, error: %s", lock_dir_path_.c_str(), ec.message().c_str());
            return EC_IO_ERROR;
        }
    }

    KVCM_LOG_INFO("CoordinationFileBackend initialized with directory: %s", lock_dir_path_.c_str());
    return EC_OK;
}

std::string CoordinationFileBackend::SanitizeKey(const std::string &key) {
    std::string safe_key = key;
    std::replace(safe_key.begin(), safe_key.end(), '/', '_');
    std::replace(safe_key.begin(), safe_key.end(), '\\', '_');
    return safe_key;
}

std::string CoordinationFileBackend::GetLockFilePath(const std::string &lock_key) const {
    return lock_dir_path_ + "/" + SanitizeKey(lock_key) + ".lock";
}

ErrorCode CoordinationFileBackend::ReadFileContent(int fd, std::string &content) {
    // 移动到文件开头
    if (lseek(fd, 0, SEEK_SET) < 0) {
        KVCM_LOG_ERROR("lseek failed: %s", strerror(errno));
        return EC_IO_ERROR;
    }

    content.clear();
    char buffer[4096];
    ssize_t bytes_read;
    while ((bytes_read = read(fd, buffer, sizeof(buffer))) > 0) {
        content.append(buffer, bytes_read);
    }
    if (bytes_read < 0) {
        KVCM_LOG_ERROR("read failed: %s", strerror(errno));
        return EC_IO_ERROR;
    }

    return EC_OK;
}

ErrorCode CoordinationFileBackend::WriteFileContent(int fd, const std::string &content) {
    // 清空文件
    if (ftruncate(fd, 0) < 0) {
        KVCM_LOG_ERROR("ftruncate failed: %s", strerror(errno));
        return EC_IO_ERROR;
    }

    // 移动到文件开头
    if (lseek(fd, 0, SEEK_SET) < 0) {
        KVCM_LOG_ERROR("lseek failed: %s", strerror(errno));
        return EC_IO_ERROR;
    }

    ssize_t bytes_written = write(fd, content.c_str(), content.size());
    if (bytes_written < 0 || static_cast<size_t>(bytes_written) != content.size()) {
        KVCM_LOG_ERROR("write failed: %s", strerror(errno));
        return EC_IO_ERROR;
    }

    return EC_OK;
}

bool CoordinationFileBackend::IsLockExpired(const std::string &lock_content, int64_t &expire_time_ms) {
    std::string unused_value;
    if (!ParseLockContent(lock_content, unused_value, expire_time_ms)) {
        return true; // 解析失败，视为过期
    }
    auto expire_time = std::chrono::system_clock::time_point(std::chrono::milliseconds(expire_time_ms));
    auto now = std::chrono::system_clock::now();
    return now >= expire_time;
}

std::string CoordinationFileBackend::SerializeLockContent(const std::string &lock_value,
                                                             int64_t expire_time_ms) const {
    rapidjson::StringBuffer sb;
    rapidjson::Writer<rapidjson::StringBuffer> writer(sb);

    writer.StartObject();
    writer.Key("value");
    writer.String(lock_value.c_str(), lock_value.size(), false);
    writer.Key("expire_time_ms");
    writer.Int64(expire_time_ms);
    writer.EndObject();

    return sb.GetString();
}

bool CoordinationFileBackend::ParseLockContent(const std::string &lock_content,
                                                  std::string &out_lock_value,
                                                  int64_t &out_expire_time_ms) const {
    rapidjson::Document doc;
    if (doc.Parse(lock_content.c_str()).HasParseError()) {
        KVCM_LOG_ERROR("failed to parse lock content as JSON: %s", lock_content.c_str());
        return false;
    }

    if (!doc.IsObject()) {
        KVCM_LOG_ERROR("lock content is not a JSON object: %s", lock_content.c_str());
        return false;
    }

    // 解析 value 字段
    auto value_iter = doc.FindMember("value");
    if (value_iter == doc.MemberEnd() || !value_iter->value.IsString()) {
        KVCM_LOG_ERROR("lock content missing or invalid 'value' field: %s", lock_content.c_str());
        return false;
    }
    out_lock_value.assign(value_iter->value.GetString(), value_iter->value.GetStringLength());

    // 解析 expire_time_ms 字段
    auto expire_iter = doc.FindMember("expire_time_ms");
    if (expire_iter == doc.MemberEnd() || !expire_iter->value.IsInt64()) {
        KVCM_LOG_ERROR("lock content missing or invalid 'expire_time_ms' field: %s", lock_content.c_str());
        return false;
    }
    out_expire_time_ms = expire_iter->value.GetInt64();

    return true;
}

ErrorCode
CoordinationFileBackend::TryLock(const std::string &lock_key, const std::string &lock_value, int64_t ttl_ms) {
    if (lock_key.empty() || lock_value.empty() || ttl_ms <= 0) {
        KVCM_LOG_ERROR("Invalid arguments for TryLock: key=%s, ttl_ms=%ld", lock_key.c_str(), ttl_ms);
        return EC_BADARGS;
    }

    std::string lock_file_path = GetLockFilePath(lock_key);

    FileLockGuard guard(lock_file_path, true);

    ErrorCode ec = guard.Lock();
    if (ec != EC_OK) {
        return ec;
    }

    // 检查锁是否过期（如果已有内容）
    std::string current_content;
    ec = ReadFileContent(guard.fd(), current_content);
    if (ec == EC_OK && !current_content.empty()) {
        int64_t expire_time_ms;
        if (!IsLockExpired(current_content, expire_time_ms)) {
            std::string current_value;
            if (!ParseLockContent(current_content, current_value, expire_time_ms)) {
                return EC_ERROR;
            }
            if (current_value == lock_value) {
                return EC_OK;
            } else {
                return EC_EXIST;
            }
        }
    }

    // 写入新的锁内容
    auto expire_time = std::chrono::system_clock::now() + std::chrono::milliseconds(ttl_ms);
    int64_t expire_time_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(expire_time.time_since_epoch()).count();
    std::string new_content = SerializeLockContent(lock_value, expire_time_ms);

    ec = WriteFileContent(guard.fd(), new_content);
    if (ec != EC_OK) {
        return ec;
    }
    return EC_OK;
}

ErrorCode
CoordinationFileBackend::RenewLock(const std::string &lock_key, const std::string &lock_value, int64_t ttl_ms) {
    std::string lock_file_path = GetLockFilePath(lock_key);

    FileLockGuard guard(lock_file_path, false);

    ErrorCode ec = guard.Lock();
    if (ec != EC_OK) {
        return ec;
    }

    // 我们持有了锁，检查内容是否匹配
    std::string current_content;
    ec = ReadFileContent(guard.fd(), current_content);
    if (ec != EC_OK) {
        return ec;
    }

    if (current_content.empty()) {
        return EC_NOENT;
    }

    // 解析内容
    std::string current_value;
    int64_t old_expire_time_ms;
    if (!ParseLockContent(current_content, current_value, old_expire_time_ms)) {
        return EC_ERROR;
    }
    if (current_value != lock_value) {
        return EC_MISMATCH; // 锁值不匹配
    }

    // 检查是否过期
    if (IsLockExpired(current_content, old_expire_time_ms)) {
        return EC_NOENT; // 锁已过期
    }

    // 更新过期时间
    auto expire_time = std::chrono::system_clock::now() + std::chrono::milliseconds(ttl_ms);
    int64_t expire_time_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(expire_time.time_since_epoch()).count();
    std::string new_content = SerializeLockContent(lock_value, expire_time_ms);

    ec = WriteFileContent(guard.fd(), new_content);

    return ec;
}

ErrorCode CoordinationFileBackend::Unlock(const std::string &lock_key, const std::string &lock_value) {
    std::string lock_file_path = GetLockFilePath(lock_key);

    FileLockGuard guard(lock_file_path, false);

    ErrorCode ec = guard.Lock();
    if (ec != EC_OK) {
        return ec;
    }

    // 我们持有了锁，检查内容是否匹配
    std::string current_content;
    ec = ReadFileContent(guard.fd(), current_content);
    if (ec != EC_OK) {
        return ec;
    }

    if (current_content.empty()) {
        return EC_NOENT;
    }

    // 解析内容
    std::string current_value;
    int64_t unused_expire_time_ms;
    if (!ParseLockContent(current_content, current_value, unused_expire_time_ms)) {
        return EC_ERROR;
    }
    if (current_value != lock_value) {
        return EC_MISMATCH;
    }

    // 清空文件内容（表示锁已释放）
    ec = WriteFileContent(guard.fd(), "");
    return ec;
}

ErrorCode CoordinationFileBackend::GetLockHolder(const std::string &lock_key,
                                                    std::string &out_current_value,
                                                    int64_t &out_expire_time_ms) {
    std::string lock_file_path = GetLockFilePath(lock_key);

    FileLockGuard guard(lock_file_path, false);

    ErrorCode ec = guard.Lock();
    if (ec != EC_OK) {
        return ec;
    }

    // 读取文件内容
    std::string current_content;
    ec = ReadFileContent(guard.fd(), current_content);
    if (ec != EC_OK) {
        return ec;
    }

    if (current_content.empty()) {
        return EC_NOENT;
    }

    // 检查是否过期
    int64_t expire_time_ms;
    if (IsLockExpired(current_content, expire_time_ms)) {
        return EC_NOENT; // 锁已过期
    }

    // 解析内容
    std::string value;
    if (!ParseLockContent(current_content, value, expire_time_ms)) {
        return EC_ERROR;
    }
    out_current_value = value;
    out_expire_time_ms = expire_time_ms;

    return EC_OK;
}

std::string CoordinationFileBackend::GetKVFilePath(const std::string &key) const {
    return lock_dir_path_ + "/" + SanitizeKey(key) + ".kv";
}

ErrorCode CoordinationFileBackend::WriteFileAtomic(const std::string &file_path, const std::string &content) {
    std::string tmp_path = file_path + ".tmp.XXXXXX";
    int fd = mkstemp(&tmp_path[0]);
    if (fd < 0) {
        KVCM_LOG_ERROR("mkstemp failed for %s: %s", file_path.c_str(), strerror(errno));
        return EC_IO_ERROR;
    }

    ssize_t bytes_written = write(fd, content.c_str(), content.size());
    if (bytes_written < 0 || static_cast<size_t>(bytes_written) != content.size()) {
        KVCM_LOG_ERROR("write failed: %s", strerror(errno));
        close(fd);
        unlink(tmp_path.c_str());
        return EC_IO_ERROR;
    }

    close(fd);

    if (std::rename(tmp_path.c_str(), file_path.c_str()) != 0) {
        KVCM_LOG_ERROR("rename failed from %s to %s: %s", tmp_path.c_str(), file_path.c_str(), strerror(errno));
        unlink(tmp_path.c_str());
        return EC_IO_ERROR;
    }

    return EC_OK;
}

ErrorCode CoordinationFileBackend::SetValue(const std::string &key, const std::string &value) {
    if (key.empty()) {
        KVCM_LOG_ERROR("Invalid arguments for SetValue: key is empty");
        return EC_BADARGS;
    }

    std::string kv_file_path = GetKVFilePath(key);
    return WriteFileAtomic(kv_file_path, value);
}

ErrorCode CoordinationFileBackend::GetValue(const std::string &key, std::string &out_value) {
    if (key.empty()) {
        KVCM_LOG_ERROR("Invalid arguments for GetValue: key is empty");
        return EC_BADARGS;
    }

    std::string kv_file_path = GetKVFilePath(key);

    int fd = open(kv_file_path.c_str(), O_RDONLY);
    if (fd < 0) {
        if (errno == ENOENT) {
            return EC_NOENT;
        }
        KVCM_LOG_ERROR("failed to open file: %s, error: %s", kv_file_path.c_str(), strerror(errno));
        return EC_IO_ERROR;
    }

    ErrorCode ec = ReadFileContent(fd, out_value);
    close(fd);
    return ec;
}

} // namespace kv_cache_manager