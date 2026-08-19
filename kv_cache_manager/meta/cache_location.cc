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
    , validated_total_size_(other.validated_total_size_) {
    if (const auto *owned = std::get_if<std::string>(&other.id_)) {
        id_.emplace<std::string>(*owned);
    } else {
        id_.emplace<InternedLocationId>(std::get<InternedLocationId>(other.id_));
    }
}

CacheLocation &CacheLocation::operator=(const CacheLocation &other) {
    if (this == &other) {
        return *this;
    }
    const bool remains_uncacheable =
        estimated_mem_usage_state_.load(std::memory_order_relaxed) == kUncacheableEstimatedMemUsage;
    estimated_mem_usage_state_.store(remains_uncacheable ? kUncacheableEstimatedMemUsage : kUnknownEstimatedMemUsage,
                                     std::memory_order_relaxed);
    validated_total_size_ = kUnknownValidatedTotalSize;
    id_ = other.id_;
    status_ = other.status_;
    type_ = other.type_;
    spec_size_ = other.spec_size_;
    create_time_ = other.create_time_;
    location_specs_ = other.location_specs_;
    validated_total_size_ = other.validated_total_size_;
    return *this;
}

CacheLocation::CacheLocation(CacheLocation &&other) noexcept
    : id_(std::move(other.id_))
    , status_(other.status_)
    , type_(other.type_)
    , spec_size_(other.spec_size_)
    , create_time_(other.create_time_)
    , location_specs_(std::move(other.location_specs_))
    , validated_total_size_(other.validated_total_size_)
    , estimated_mem_usage_state_(other.estimated_mem_usage_state_.load(std::memory_order_relaxed) ==
                                         kUncacheableEstimatedMemUsage
                                     ? kUncacheableEstimatedMemUsage
                                     : kUnknownEstimatedMemUsage) {
    other.InvalidateEstimatedMemUsage();
    other.validated_total_size_ = kUnknownValidatedTotalSize;
}

CacheLocation &CacheLocation::operator=(CacheLocation &&other) noexcept {
    if (this == &other) {
        return *this;
    }
    const bool remains_uncacheable =
        estimated_mem_usage_state_.load(std::memory_order_relaxed) == kUncacheableEstimatedMemUsage ||
        other.estimated_mem_usage_state_.load(std::memory_order_relaxed) == kUncacheableEstimatedMemUsage;
    id_ = std::move(other.id_);
    status_ = other.status_;
    type_ = other.type_;
    spec_size_ = other.spec_size_;
    create_time_ = other.create_time_;
    location_specs_ = std::move(other.location_specs_);
    validated_total_size_ = other.validated_total_size_;
    estimated_mem_usage_state_.store(remains_uncacheable ? kUncacheableEstimatedMemUsage : kUnknownEstimatedMemUsage,
                                     std::memory_order_relaxed);
    other.InvalidateEstimatedMemUsage();
    other.validated_total_size_ = kUnknownValidatedTotalSize;
    return *this;
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
