#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace kv_cache_manager {

struct BlockEntry;

// ============================================================================
// 统计记录数据定义
//
// 所有 Tracker 使用的 Record 结构体统一定义于此。
// ============================================================================

struct ReadRecord {
    int64_t timestamp_ns;
    size_t remote_read_blocks;
    size_t remote_hit_blocks;
    size_t local_read_blocks;
    size_t local_hit_blocks;
    size_t current_cache_blocks;
    std::vector<size_t> per_tier_hit_blocks; // per-tier hit block num, indexed by tier priority
    std::vector<std::string> tier_names;     // tier names for CSV column headers
    std::vector<size_t> per_tier_blocks;     // per-tier block num for current instance
    std::vector<size_t> blocks_per_instance;
    std::string trace_id;
    const std::vector<int64_t> *keys_ptr = nullptr; // 借用，仅 OnReadComplete 期间有效
};

struct WriteRecord {
    int64_t timestamp_ns;
    size_t write_blocks;          // 请求写入的 block 总数（含已存在的）
    size_t newly_inserted_blocks; // 实际新插入的 block 数（不含已存在的）

    std::string trace_id; // 当前 trace 标识
};

struct BlockLifecycleRecord {
    int64_t block_key;
    int64_t birth_time_ns;
    int64_t death_time_ns; // -1 表示仍存活
    int64_t lifespan_ns;
    size_t access_count;
    int64_t last_access_time_ns;
    bool is_alive;                   // true表示trace结束时仍存活, false表示被驱逐
    BlockEntry *block_ptr = nullptr; // 存活期间持有指针，Finalize 时读取最终统计值
};

} // namespace kv_cache_manager
