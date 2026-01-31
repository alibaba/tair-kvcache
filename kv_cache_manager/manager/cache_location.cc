#include "cache_location.h"

#include "kv_cache_manager/common/string_util.h"
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

} // namespace kv_cache_manager