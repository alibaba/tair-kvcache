#pragma once

#include <cassert>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/meta/cache_location.h"

namespace kv_cache_manager {

// =============================================================================
// Common type aliases used across the meta module.
//
// These aliases used to be scattered across MetaIndexer / MetaStorageBackend /
// MetaStorageBackendManager etc. as nested `using` declarations, which forced
// every consumer to spell `MetaIndexer::KeyVector`, `MetaStorageBackend::FieldMap`
// and so on. Centralising them in this single header gives the meta module a
// stable, dependency-free vocabulary that other components can rely on.
//
// Do **not** put aliases that are tightly coupled to a specific class
// (e.g. `BlockCacheLocationsMeta::location_map_t`) here - keep them next to
// the owning class. This file is intended for primitives that span the whole
// module.
// =============================================================================

// ---------- Storage primitives ----------
using KeyType = int64_t;
using KeyVector = std::vector<KeyType>;
// Alias preserved for backwards compatibility with existing call sites that
// still reference the old `KeyTypeVec` spelling.
using KeyTypeVec = KeyVector;

using FieldMap = std::map<std::string, std::string>;
using FieldMapVec = std::vector<FieldMap>;

// ---------- Property primitives ----------
using PropertyMap = FieldMap;
using PropertyMapVector = std::vector<PropertyMap>;

// ---------- Location primitives ----------
using LocationId = std::string;
using LocationIdVector = std::vector<LocationId>;
// Borrowed ids used by the pure-local one-location RMW fast path. The owner
// (normally a MetaSearcher task vector) must outlive the synchronous call.
using LocationIdRefVector = std::vector<const LocationId *>;
using LocationIdsPerKey = std::vector<LocationIdVector>;
using LocationsPerKey = std::vector<CacheLocationVector>;
// Non-owning immutable values used only by the synchronous pure-local RMW
// path. The retained cache handles and metadata shard locks keep the backing
// MetaMemCacheItem (and its location map) alive until the matching write or
// explicit handle release. These views must never escape that interval.
using CacheLocationViewVector = std::vector<const CacheLocation *>;

// ---------- Read-Modify-Write helpers ----------
// Action codes shared by both block-level and location-level RMW modifiers.
enum ModifierAction {
    MA_OK = 1,
    MA_FAIL = 2,
    MA_SKIP = 3,
    MA_DELETE = 4
};
using ModifierResult = std::pair<ModifierAction, ErrorCode>;
using LocationModifierResult = std::pair<ModifierAction, std::vector<ErrorCode>>;

// Lightweight block-level modifier: only sees the existing location id list
// (no location values are deserialized). The modifier can produce new
// CacheLocations to be written via the output CacheLocationMap.
using BlockIdsOnlyModifierFunc = std::function<ModifierResult(const LocationIdVector & /*existing_location_ids*/,
                                                              ErrorCode /*get_ec*/,
                                                              size_t /*key_index*/,
                                                              PropertyMap & /*upsert_property_map*/,
                                                              CacheLocationMap & /*out_new_locations*/)>;

// Note: an earlier `BlockModifierFunc` variant (sees the entire CacheLocationMap
// of one block_key) was removed because nothing in the codebase ever used
// it. If a future caller really needs the deserialized map, it should fetch
// it via MetaIndexer::GetLocations and then drive ReadModifyWriteLocation.

// Location-level modifier: sees one (block_key, location_id, CacheLocation) at
// a time. Backend only reads/writes the specified location field.
using LocationModifierFunc = std::function<LocationModifierResult(const std::vector<ErrorCode> & /*get_ecs*/,
                                                                  const LocationIdVector & /*loc_ids*/,
                                                                  size_t /*key_index*/,
                                                                  CacheLocationVector & /*locs*/,
                                                                  PropertyMap & /*upsert_property_map*/)>;

// Allocation-light targeted-upsert modifier for the common one-location-per-
// key ReportEvent shape. `existing_location` is a borrowed immutable view
// protected by the synchronous RMW's retained cache handle and shard lock;
// `out_location` receives the independently-owned replacement when MA_OK is
// returned. MA_DELETE is not supported by this specialized path.
using SingleLocationModifierFunc = std::function<ModifierResult(ErrorCode /*get_ec*/,
                                                                const LocationId & /*location_id*/,
                                                                size_t /*key_index*/,
                                                                const CacheLocation * /*existing_location*/,
                                                                CacheLocationConstPtr & /*out_location*/)>;

// ---------- Batch primitives ----------
// Single-batch view consumed by MetaStorageBackendManager. Fields with suffix
// `[j]` are parallel arrays indexed by position within the batch.
struct BatchMetaData {
    std::vector<int32_t> batch_shard_indexs; // shard mutex indices in this batch
    std::vector<int32_t> batch_indexs;       // raw index in the original KeyVector
    KeyVector batch_keys;                    // keys in this batch
    CacheLocationMapVector batch_locations;  // optional per-key CacheLocations
    LocationIdsPerKey batch_location_ids;    // optional per-key location ids
    PropertyMapVector batch_properties;      // optional per-key properties

    // Ensure batch_locations and batch_properties are sized to match batch_keys.
    void EnsureLocationsAndPropertiesResized() {
        if (batch_locations.empty()) {
            batch_locations.resize(batch_keys.size());
        }
        if (batch_properties.empty()) {
            batch_properties.resize(batch_keys.size());
        }
    }
};

} // namespace kv_cache_manager
