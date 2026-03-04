#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace kv_cache_manager {

// ============================================================================
// 统计记录数据定义
//
// 所有 Tracker 使用的 Record 结构体统一定义于此。
// ============================================================================

struct ReadRecord {
    int64_t timestamp_us;
    size_t external_read_blocks;
    size_t external_hit_blocks;
    size_t internal_read_blocks;
    size_t internal_hit_blocks;
    size_t current_cache_blocks;
    std::vector<size_t> blocks_per_instance;
};

struct WriteRecord {
    int64_t timestamp_us;
    size_t write_blocks;
};

struct BlockLifecycleRecord {
    int64_t block_key;
    int64_t birth_time_us;
    int64_t death_time_us; // -1 表示仍存活
    int64_t lifespan_us;
    size_t access_count;
    int64_t last_access_time_us;
    bool is_alive; // true表示trace结束时仍存活, false表示被驱逐
};

} // namespace kv_cache_manager
