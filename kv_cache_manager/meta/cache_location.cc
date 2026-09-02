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

CacheLocation::CacheLocation(const CacheLocation &other)
    : status_(other.status_)
    , type_(other.type_)
    , spec_size_(other.spec_size_)
    , create_time_(other.create_time_)
    , location_specs_(other.location_specs_)
    , validated_total_size_(other.validated_total_size_)
    , migration_copy_guard_(other.migration_copy_guard_) {
    if (const auto *owned = std::get_if<std::string>(&other.id_)) {
        id_.emplace<std::string>(*owned);
    } else {
        id_.emplace<InternedLocationId>(std::get<InternedLocationId>(other.id_));
    }
}

CacheLocation::CacheLocation(DataStorageType type, size_t spec_size, const std::vector<LocationSpec> &location_specs)
    : type_(type), spec_size_(spec_size), location_specs_(location_specs) {}

CacheLocation::CacheLocation(const std::string &id,
                             CacheLocationStatus status,
                             DataStorageType type,
                             size_t spec_size,
                             const std::vector<LocationSpec> &location_specs)
    : id_(id), status_(status), type_(type), spec_size_(spec_size), location_specs_(location_specs) {}

CacheLocation::~CacheLocation() = default;

} // namespace kv_cache_manager
