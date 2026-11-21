#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace kv_cache_manager {

inline constexpr std::size_t kKvMetaMaxObjectUserData2Bytes = 256;

struct KvMetaObjectUserData {
    std::optional<std::int32_t> userdata1;
    std::optional<std::string> userdata2;

    [[nodiscard]] bool empty() const noexcept { return !userdata1 && !userdata2; }
    [[nodiscard]] bool bounded() const noexcept {
        return !userdata2 || userdata2->size() <= kKvMetaMaxObjectUserData2Bytes;
    }
};

inline constexpr std::size_t kKvMetaMaxBatchItems = 64;
inline constexpr std::size_t kKvMetaMaxKeyBytes = 512;
inline constexpr std::size_t kKvMetaMaxInstanceIdBytes = 512;
inline constexpr std::size_t kKvMetaMaxInstanceGroupBytes = 512;
inline constexpr std::size_t kKvMetaMaxWriteSessionIdBytes = 512;
inline constexpr std::size_t kKvMetaMaxUserDataBytes = 64 * 1024;
inline constexpr std::size_t kKvMetaMaxActiveWriteSessions = 65536;
inline constexpr std::size_t kKvMetaMaxLocationUriBytes = 64 * 1024;
inline constexpr std::uint64_t kKvMetaMaxValueBytes = 1ULL * 1024 * 1024 * 1024;
inline constexpr std::uint64_t kKvMetaMaxBatchBytes = 4ULL * 1024 * 1024 * 1024;
inline constexpr std::int64_t kKvMetaMaxWriteTimeoutSeconds = 1800;
inline constexpr std::size_t kKvMetaObjectNonceBytes = 32;
inline constexpr char kKvMetaValueSpecName[] = "value";

} // namespace kv_cache_manager
