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
using LocationMap = std::unordered_map<LocationId, CacheLocation>;
using LocationMapVector = std::vector<LocationMap>;
using LocationIdsPerKey = std::vector<LocationIdVector>;
using LocationsPerKey = std::vector<CacheLocationVector>;

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
// CacheLocations to be written via the output LocationMap.
using BlockIdsOnlyModifierFunc = std::function<ModifierResult(const LocationIdVector & /*existing_ids*/,
                                                              ErrorCode /*get_ec*/,
                                                              size_t /*key_index*/,
                                                              PropertyMap & /*upsert_property_map*/,
                                                              LocationMap & /*out_new_locations*/)>;

// Note: an earlier `BlockModifierFunc` variant (sees the entire LocationMap
// of one block_key) was removed because nothing in the codebase ever used
// it. If a future caller really needs the deserialized map, it should fetch
// it via MetaIndexer::GetLocations and then drive ReadModifyWriteLocation.

// Location-level modifier: sees one (block_key, location_id, CacheLocation) at
// a time. Backend only reads/writes the specified location field.
using LocationModifierFunc = std::function<LocationModifierResult(CacheLocationVector & /*loc*/,
                                                                std::vector<ErrorCode> /*get_ec*/,
                                                                size_t /*key_index*/,
                                                                const LocationIdVector & /*loc_id*/,
                                                                PropertyMap & /*upsert_property_map*/)>;

// ---------- Batch primitives ----------
// Single-batch view consumed by MetaStorageBackendManager. Fields with suffix
// `[j]` are parallel arrays indexed by position within the batch.
struct BatchMetaData {
    std::vector<int32_t> batch_shard_indexs; // shard mutex indices in this batch
    std::vector<int32_t> batch_indexs;       // raw index in the original KeyVector
    KeyVector batch_keys;                    // keys in this batch
    LocationMapVector batch_locations;       // optional per-key CacheLocations
    LocationIdsPerKey batch_location_ids;    // optional per-key location ids
    PropertyMapVector batch_properties;      // optional per-key properties
};

} // namespace kv_cache_manager
