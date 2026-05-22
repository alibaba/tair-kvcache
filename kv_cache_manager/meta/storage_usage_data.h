#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/data_storage/storage_config.h"

namespace kv_cache_manager {

// Per-instance storage usage data, aggregated by DataStorageType.
// Thread-safe via atomic counters per storage type slot.
class StorageUsageData : public Jsonizable {
public:
    StorageUsageData() = default;
    ~StorageUsageData() override = default;

    [[nodiscard]] std::uint64_t GetStorageUsage() const noexcept;
    [[nodiscard]] std::uint64_t GetStorageUsageByType(const DataStorageType &type) const noexcept;

    void Reset() noexcept;
    void SetStorageUsageByType(const DataStorageType &type, std::uint64_t value) noexcept;

    std::uint64_t AddStorageUsageByType(const DataStorageType &type, std::uint64_t value) noexcept;
    std::uint64_t SubStorageUsageByType(const DataStorageType &type, std::uint64_t value) noexcept;

    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;
    bool FromRapidValue(const rapidjson::Value &rapid_value) override;

    [[nodiscard]] std::string Serialize() const noexcept;
    ErrorCode Deserialize(const std::string &str) noexcept;

private:
    using array_t_ = std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(DataStorageType::COUNT)>;
    using size_t_ = array_t_::size_type;

    // storage usage data array aggregated by storage type
    // slot 0: DATA_STORAGE_TYPE_UNKNOWN **UNUSED**
    // slot 1: DATA_STORAGE_TYPE_HF3FS usage data
    // slot 2: DATA_STORAGE_TYPE_MOONCAKE usage data
    // slot 3: DATA_STORAGE_TYPE_TAIR_MEMPOOL usage data
    // slot 4: DATA_STORAGE_TYPE_NFS usage data
    // slot 5: DATA_STORAGE_TYPE_VCNS_HF3FS **UNUSED** (merged into HF3FS)
    // slot 6: DATA_STORAGE_TYPE_DUMMY usage data (testing only)
    array_t_ storage_usage_by_type_;
};

} // namespace kv_cache_manager
