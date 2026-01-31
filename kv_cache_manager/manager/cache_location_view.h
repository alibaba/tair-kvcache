#pragma once

#include <string_view>
#include <vector>

#include "kv_cache_manager/data_storage/common_define.h"
#include "kv_cache_manager/manager/cache_location.h"

namespace kv_cache_manager {

// these classes are same to meta_service.proto
class LocationSpecView {
public:
    LocationSpecView(const LocationSpec &raw_location_spec) : raw_location_spec_(raw_location_spec) {}

    inline const std::string &name() const { return raw_location_spec_.name(); }
    inline const std::string &uri() const { return raw_location_spec_.uri(); }

private:
    const LocationSpec &raw_location_spec_;
};

class CacheLocationView {
public:
    class LocationSpecTransformer {
    public:
        LocationSpecTransformer(DataStorageType type) : type_(type) {}
        LocationSpecView operator()(const LocationSpec &spec) const;

    private:
        DataStorageType type_;
    };
    using LocationSpecViewVec = std::vector<LocationSpecView>;
    explicit CacheLocationView(const CacheLocation &cache_location);
    inline DataStorageType type() const { return cache_location_.type(); }
    inline int32_t spec_size() const { return cache_location_.spec_size(); }
    inline const LocationSpecViewVec &location_specs() const { return location_specs_view_; }

private:
    const CacheLocation &cache_location_;
    LocationSpecViewVec location_specs_view_;
};

class CacheLocationTransformer {
public:
    CacheLocationView operator()(const CacheLocation &cache_location) const;
};
using CacheLocationViewVec = std::vector<CacheLocationView>;

class CacheLocationViewVecWrapper {
public:
    CacheLocationViewVecWrapper();
    CacheLocationViewVecWrapper(CacheLocationViewVecWrapper &&other);
    explicit CacheLocationViewVecWrapper(CacheLocationVector &&raw_cache_locations);
    inline const CacheLocationViewVec &cache_locations_view() const { return cache_locations_view_; }

private:
    CacheLocationViewVec cache_locations_view_;
    CacheLocationVector raw_cache_locations_;
};

class CacheMetaVecWrapper {
public:
    CacheMetaVecWrapper();
    CacheMetaVecWrapper(CacheMetaVecWrapper &&other);
    CacheMetaVecWrapper(std::vector<std::string> &&metas, CacheLocationVector &&raw_cache_locations);
    inline const CacheLocationViewVec &cache_locations_view() const { return locations_.cache_locations_view(); }
    inline const std::vector<std::string> &metas() const { return metas_; }

private:
    std::vector<std::string> metas_;
    CacheLocationViewVecWrapper locations_;
};

class StartWriteCacheInfo {
public:
    StartWriteCacheInfo() = default;
    StartWriteCacheInfo(StartWriteCacheInfo &&other)
        : write_session_id_(std::move(other.write_session_id_))
        , block_mask_(std::move(other.block_mask_))
        , locations_(std::move(other.locations_)) {}
    StartWriteCacheInfo(std::string &&write_session_id, BlockMask &&block_mask, CacheLocationViewVecWrapper &&locations)
        : write_session_id_(std::move(write_session_id))
        , block_mask_(std::move(block_mask))
        , locations_(std::move(locations)) {}
    // Getter methods for read access
    const std::string &write_session_id() const { return write_session_id_; }
    const BlockMask &block_mask() const { return block_mask_; }
    const CacheLocationViewVecWrapper &locations() const { return locations_; }
    CacheLocationViewVecWrapper &locations_mut() { return locations_; }

private:
    std::string write_session_id_;
    BlockMask block_mask_;
    CacheLocationViewVecWrapper locations_;
};

} // namespace kv_cache_manager