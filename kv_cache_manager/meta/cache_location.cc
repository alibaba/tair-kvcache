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

CacheLocation::CacheLocation(DataStorageType type, size_t spec_size, const std::vector<LocationSpec> &location_specs)
    : type_(type), spec_size_(spec_size), location_specs_(location_specs) {}

CacheLocation::CacheLocation(const std::string &id,
                             CacheLocationStatus status,
                             DataStorageType type,
                             size_t spec_size,
                             const std::vector<LocationSpec> &location_specs)
    : id_(id), status_(status), type_(type), spec_size_(spec_size), location_specs_(location_specs) {}

CacheLocation::~CacheLocation() = default;

BlockCacheLocationsMeta::BlockCacheLocationsMeta() = default;

BlockCacheLocationsMeta::~BlockCacheLocationsMeta() = default;

void BlockCacheLocationsMeta::AddNewLocation(const CacheLocation &location, std::string &out_location_id) {
    do {
        out_location_id = StringUtil::GenerateRandomString(8);
    } while (location_map_.count(out_location_id) > 0);

    location_map_.insert({out_location_id, location});
    location_map_[out_location_id].set_id(out_location_id);
}

ErrorCode BlockCacheLocationsMeta::UpdateLocationStatus(const std::string &location_id, CacheLocationStatus status) {
    auto it = location_map_.find(location_id);
    if (it == location_map_.end()) {
        return ErrorCode::EC_NOENT;
    }

    it->second.set_status(status);
    return ErrorCode::EC_OK;
}

ErrorCode BlockCacheLocationsMeta::DeleteLocation(const std::string &location_id) {
    size_t delete_count = location_map_.erase(location_id);

    return delete_count > 0 ? ErrorCode::EC_OK : ErrorCode::EC_NOENT;
}
ErrorCode BlockCacheLocationsMeta::GetLocationStatus(const std::string &location_id, CacheLocationStatus &out_status) {
    auto it = location_map_.find(location_id);
    if (it == location_map_.end()) {
        return ErrorCode::EC_NOENT;
    }
    out_status = it->second.status();
    return ErrorCode::EC_OK;
}
size_t BlockCacheLocationsMeta::GetLocationCount() const { return location_map_.size(); }

void BlockCacheLocationsMeta::ToFieldMap(std::map<std::string, std::string> &out_field_map) const noexcept {
    for (const auto &kv : location_map_) {
        // Per-location entries land at L#{location_id}; the value is the full
        // CacheLocation JSON so callers can deserialize one row at a time.
        out_field_map[PROPERTY_LOCATION_PREFIX + kv.first] = kv.second.ToJsonString();
    }
}

bool BlockCacheLocationsMeta::FromFieldMap(const std::map<std::string, std::string> &field_map) {
    for (const auto &kv : field_map) {
        if (kv.first.rfind(PROPERTY_LOCATION_PREFIX, 0) != 0) {
            // Skip BP#/P#/legacy fields here; they belong to other consumers.
            continue;
        }
        const std::string location_id = kv.first.substr(PROPERTY_LOCATION_PREFIX.size());
        if (location_id.empty()) {
            // Defensive: an empty location_id means a malformed key like "L#".
            return false;
        }
        CacheLocation loc;
        if (!loc.FromJsonString(kv.second)) {
            return false;
        }
        if (loc.id().empty()) {
            // Materialize the id from the field name for callers that may not
            // have populated CacheLocation::id_ at write time.
            loc.set_id(location_id);
        }
        location_map_[location_id] = std::move(loc);
    }
    return true;
}

} // namespace kv_cache_manager