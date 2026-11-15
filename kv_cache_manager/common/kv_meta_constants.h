#pragma once

#include <cstddef>
#include <cstdint>

namespace kv_cache_manager {

inline constexpr std::size_t kKvMetaMaxBatchItems = 64;
inline constexpr std::size_t kKvMetaMaxKeyBytes = 512;
inline constexpr std::size_t kKvMetaMaxInstanceIdBytes = 512;
inline constexpr std::size_t kKvMetaMaxInstanceGroupBytes = 512;
inline constexpr std::size_t kKvMetaMaxWriteSessionIdBytes = 512;
inline constexpr std::size_t kKvMetaMaxUserDataBytes = 64 * 1024;
inline constexpr std::size_t kKvMetaMaxActiveWriteSessions = 4096;
inline constexpr std::size_t kKvMetaMaxLocationUriBytes = 64 * 1024;
inline constexpr std::uint64_t kKvMetaMaxValueBytes = 1ULL * 1024 * 1024 * 1024;
inline constexpr std::uint64_t kKvMetaMaxBatchBytes = 4ULL * 1024 * 1024 * 1024;
inline constexpr std::int64_t kKvMetaMaxWriteTimeoutSeconds = 1800;
inline constexpr std::size_t kKvMetaObjectNonceBytes = 32;
inline constexpr char kKvMetaValueSpecName[] = "value";

} // namespace kv_cache_manager
