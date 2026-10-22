#include "kv_cache_manager/meta/cache_location.h"

#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/meta/common.h"
namespace kv_cache_manager {

LocationSpec::LocationSpec() = default;
LocationSpec::~LocationSpec() = default;

bool IsIndexInMaskRange(const BlockMask &mask, size_t index) {
    if (std::holds_alternative<BlockMaskVector>(mask)) {
        const auto &mask_vector = std::get<BlockMaskVector>(mask);
        if (index < mask_vector.size()) {
            return mask_vector[index];
        }
        return false;
    } else if (std::holds_alternative<BlockMaskOffset>(mask)) {
        const auto &mask_offset = std::get<BlockMaskOffset>(mask);
        return index < mask_offset;
    }
    return false;
}
bool IsBlockMaskValid(const BlockMask &mask, size_t size) {
    if (std::holds_alternative<BlockMaskVector>(mask)) {
        const auto &mask_vector = std::get<BlockMaskVector>(mask);
        return mask_vector.size() == size;
    } else if (std::holds_alternative<BlockMaskOffset>(mask)) {
        const auto &mask_offset = std::get<BlockMaskOffset>(mask);
        return mask_offset <= size;
    }
    return false;
}

CacheLocation::CacheLocation() = default;

bool CacheLocation::FromJsonStringInterned(const std::string &json, LocationIdInterner &interner) {
    rapidjson::Document doc;
    return Parse(doc, json) && FromRapidValueImpl(doc, &interner);
}

InternedLocationId LocationIdInterner::Intern(const std::string &id, bool *retained) {
    auto &shard = shards_[std::hash<std::string>{}(id) & (kShardCount - 1)];
    std::lock_guard lock(shard.mutex);
    auto it = shard.ids.find(id);
    if (it != shard.ids.end()) {
        if (auto live = it->second.lock()) {
            if (retained) {
                *retained = true;
            }
            return live;
        }
    }
    auto shared = std::make_shared<const std::string>(id);
    const bool keep = it != shard.ids.end() || shard.ids.size() < kMaxIdsPerShard;
    if (it != shard.ids.end()) {
        it->second = shared;
    } else if (keep) {
        shard.ids.emplace(*shared, shared);
    }
    if (retained) {
        *retained = keep;
    }
    return shared;
}

CacheLocation::CacheLocation(DataStorageType type, size_t spec_size, const std::vector<LocationSpec> &location_specs)
    : location_specs_(location_specs), type_(type) {
    set_spec_size(spec_size);
}

CacheLocation::CacheLocation(const std::string &id,
                             CacheLocationStatus status,
                             DataStorageType type,
                             size_t spec_size,
                             const std::vector<LocationSpec> &location_specs)
    : id_(std::make_shared<const std::string>(id)), location_specs_(location_specs), status_(status), type_(type) {
    set_spec_size(spec_size);
}

CacheLocation::~CacheLocation() = default;

} // namespace kv_cache_manager
