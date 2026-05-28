#include "kv_cache_manager/meta/storage_usage_data.h"

#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <string>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/data_storage/storage_config.h"

namespace kv_cache_manager {

std::uint64_t StorageUsageData::GetStorageUsage() const noexcept {
    std::uint64_t storage_usage = 0;
    for (size_t_ i = 0; i != storage_usage_by_type_.size(); ++i) {
        if (i == static_cast<size_t_>(DataStorageType::DATA_STORAGE_TYPE_UNKNOWN) ||
            i == static_cast<size_t_>(DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS)) {
            continue;
        }
        storage_usage += storage_usage_by_type_.at(i).load();
    }
    return storage_usage;
}

std::uint64_t StorageUsageData::GetStorageUsageByType(const DataStorageType &type) const noexcept {
    const size_t_ idx = ToIndex(ToBaseType(type));
    if (idx >= storage_usage_by_type_.size()) {
        KVCM_LOG_WARN("data storage type to index out of range, array size: [%zu], type as index: [%zu]",
                      storage_usage_by_type_.size(),
                      idx);
        return 0;
    }
    return storage_usage_by_type_.at(idx).load();
}

void StorageUsageData::Reset() noexcept {
    // array.fill(0) won't work here due to the deleted operator= of the
    // std::atomic type, explicitly assign 0 to all elements in the
    // array instead
    for (auto &v : storage_usage_by_type_) {
        v.store(0);
    }
}

void StorageUsageData::SetStorageUsageByType(const DataStorageType &type, const std::uint64_t value) noexcept {
    const size_t_ idx = ToIndex(ToBaseType(type));
    if (idx >= storage_usage_by_type_.size()) {
        KVCM_LOG_WARN("data storage type to index out of range, array size: [%zu], type as index: [%zu]",
                      storage_usage_by_type_.size(),
                      idx);
        return;
    }
    storage_usage_by_type_.at(idx).store(value);
}

std::uint64_t StorageUsageData::AddStorageUsageByType(const DataStorageType &type, const std::uint64_t value) noexcept {
    const size_t_ idx = ToIndex(ToBaseType(type));
    if (idx >= storage_usage_by_type_.size()) {
        KVCM_LOG_WARN("data storage type to index out of range, array size: [%zu], type as index: [%zu]",
                      storage_usage_by_type_.size(),
                      idx);
        return 0;
    }
    return storage_usage_by_type_.at(idx).fetch_add(value);
}

std::uint64_t StorageUsageData::SubStorageUsageByType(const DataStorageType &type, const std::uint64_t value) noexcept {
    const size_t_ idx = ToIndex(ToBaseType(type));
    if (idx >= storage_usage_by_type_.size()) {
        KVCM_LOG_WARN("data storage type to index out of range, array size: [%zu], type as index: [%zu]",
                      storage_usage_by_type_.size(),
                      idx);
        return 0;
    }

    auto &ref = storage_usage_by_type_.at(idx);
    std::uint64_t expected = ref.load(), desired = 0;
    bool underflow = false;
    do {
        if (expected < value) {
            underflow = true;
            desired = 0;
        } else {
            desired = expected - value;
        }
    } while (!ref.compare_exchange_weak(expected, desired));
    if (underflow) {
        KVCM_LOG_DEBUG("storage usage underflow for type [%zu]: "
                       "current [%" PRIu64 "] < subtract [%" PRIu64 "], clamped to 0",
                       idx,
                       expected,
                       value);
    }
    return desired;
}

void StorageUsageData::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    for (size_t_ i = 0; i != storage_usage_by_type_.size(); ++i) {
        const auto type = static_cast<DataStorageType>(i);
        const std::string key = ToString(type);
        Put(writer, key, storage_usage_by_type_.at(i).load());
    }
}

bool StorageUsageData::FromRapidValue(const rapidjson::Value &rapid_value) {
    if (!rapid_value.IsObject()) {
        return false;
    }

    // parse into a temporary buffer first to avoid partial updates
    std::array<std::uint64_t, static_cast<std::size_t>(DataStorageType::COUNT)> buf{};
    for (auto it = rapid_value.MemberBegin(); it != rapid_value.MemberEnd(); ++it) {
        const std::string key(it->name.GetString(), it->name.GetStringLength());
        const DataStorageType type = ToDataStorageType(key);
        if (ToString(type) != key) {
            // round-trip mismatch: key is not a recognized type
            // prevent ToDataStorageType silently maps every unknown
            // string to DATA_STORAGE_TYPE_UNKNOWN
            KVCM_LOG_ERROR("deserialize storage usage data failed: unrecognized storage type [%s]", key.c_str());
            return false;
        }
        const size_t_ idx = ToIndex(type);
        if (it->value.IsUint64()) {
            buf.at(idx) = it->value.GetUint64();
        } else {
            KVCM_LOG_ERROR("deserialize storage usage data failed: non-integer value for key [%s]", key.c_str());
            return false;
        }
    }

    // all values parsed successfully, apply to the actual array
    for (size_t_ i = 0; i != storage_usage_by_type_.size(); ++i) {
        storage_usage_by_type_.at(i).store(buf.at(i));
    }
    return true;
}

std::string StorageUsageData::Serialize() const noexcept {
    const std::string str = ToJsonString();
    KVCM_LOG_DEBUG("serializing storage usage data into: [%s]", str.c_str());
    return str;
}

ErrorCode StorageUsageData::Deserialize(const std::string &str) noexcept {
    KVCM_LOG_DEBUG("deserializing storage usage data: [%s]", str.c_str());
    std::string str_copy{str};
    StringUtil::Trim(str_copy);
    if (str_copy.empty()) {
        KVCM_LOG_ERROR("deserialize storage usage data failed: input string is empty");
        return ErrorCode::EC_ERROR;
    }
    if (!FromJsonString(str_copy)) {
        KVCM_LOG_ERROR("deserialize storage usage data failed: invalid JSON [%s]", str_copy.c_str());
        return ErrorCode::EC_ERROR;
    }
    return ErrorCode::EC_OK;
}

} // namespace kv_cache_manager
