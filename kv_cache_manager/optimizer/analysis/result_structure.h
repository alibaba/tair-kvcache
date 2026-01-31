#pragma once
#include <memory>
#include <vector>
namespace kv_cache_manager {

struct ResultCounters {
    uint64_t total_blocks = 0;
    uint64_t total_write_blocks = 0;
    uint64_t total_read_blocks = 0;
    uint64_t total_hit_blocks = 0;

    uint64_t total_read_requests = 0;
    uint64_t total_requests = 0;
};

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
struct Result {
    ResultCounters counters;
    std::vector<ReadRecord> read_results;
    std::vector<WriteRecord> write_results;
};
} // namespace kv_cache_manager