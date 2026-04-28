#include "kv_cache_manager/optimizer/index/global_registry.h"

#include <filesystem>
#include <fstream>

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

void GlobalRegistry::ExportDuplicateStats(const std::string &output_dir) {
    if (duplicate_snapshots_.empty()) {
        KVCM_LOG_WARN("No duplicate snapshots recorded, skipping export");
        return;
    }

    std::filesystem::create_directories(output_dir);

    std::string filename = output_dir + "/global_duplicate_blocks.csv";
    std::ofstream file(filename);
    if (!file.is_open()) {
        KVCM_LOG_ERROR("Failed to open file for writing duplicate stats: %s", filename.c_str());
        return;
    }

    file << "TimestampUs,TotalUniqueKeys,TotalBlockCopies,DuplicateBlockCopies\n";

    for (const auto &s : duplicate_snapshots_) {
        file << s.timestamp_us << "," << s.total_unique_keys << ","
             << s.total_block_copies << "," << s.duplicate_block_copies << "\n";
    }

    file.close();
    KVCM_LOG_INFO("Duplicate block stats exported to: %s (total snapshots: %zu)",
                  filename.c_str(),
                  duplicate_snapshots_.size());
}

} // namespace kv_cache_manager
