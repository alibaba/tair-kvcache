#include "kv_cache_manager/manager/cache_location_view.h"

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

CacheLocationView::CacheLocationView(const CacheLocation &cache_location) : cache_location_(cache_location) {
    LocationSpecTransformer transformer(cache_location.type());
    const auto &location_specs = cache_location_.location_specs();
#if __cplusplus > 201703L
    location_specs_view_ = location_specs | std::views::transform(transformer);
#else
    size_t size = location_specs.size();
    location_specs_view_.reserve(size);
    for (const auto &location_spec : location_specs) {
        location_specs_view_.push_back(transformer(location_spec));
    }
#endif
}

LocationSpecView CacheLocationView::LocationSpecTransformer::operator()(const LocationSpec &spec) const {
    // 暂时什么都没干
    return {spec};
}

CacheLocationView CacheLocationTransformer::operator()(const CacheLocation &cache_location) const {
    return CacheLocationView(cache_location);
}

CacheLocationViewVecWrapper::CacheLocationViewVecWrapper() = default;

CacheLocationViewVecWrapper::CacheLocationViewVecWrapper(CacheLocationViewVecWrapper &&other)
    : cache_locations_view_(std::move(other.cache_locations_view_))
    , raw_cache_locations_(std::move(other.raw_cache_locations_)) {}

CacheLocationViewVecWrapper::CacheLocationViewVecWrapper(CacheLocationVector &&raw_cache_locations)
    : raw_cache_locations_(std::move(raw_cache_locations)) {

    size_t size = raw_cache_locations_.size();
    if (size) {
        CacheLocationTransformer transformer;
        cache_locations_view_.reserve(size);
        for (const auto &cache_location : raw_cache_locations_) {
            cache_locations_view_.push_back(transformer(cache_location));
        }
    }
}

CacheMetaVecWrapper::CacheMetaVecWrapper() = default;

CacheMetaVecWrapper::CacheMetaVecWrapper(CacheMetaVecWrapper &&other)
    : metas_(std::move(other.metas_)), locations_(std::move(other.locations_)) {}

CacheMetaVecWrapper::CacheMetaVecWrapper(std::vector<std::string> &&metas, CacheLocationVector &&raw_cache_locations)
    : metas_(std::move(metas)), locations_(std::move(raw_cache_locations)) {}

} // namespace kv_cache_manager