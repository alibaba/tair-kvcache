#pragma once

#include "kv_cache_manager/meta/meta_storage_backend.h"

namespace kv_cache_manager {

// Intermediate base class for cache-type backends (e.g. MetaLocalBackend).
// Extends MetaStorageBackend with conditional write operations that accept
// previous_error_codes, allowing callers to skip keys that already failed
// in a prior write stage (e.g. persistent write) without copying key/value vectors.
class MetaLocalBaseBackend : public MetaStorageBackend {
public:
    ~MetaLocalBaseBackend() override = default;

    using MetaStorageBackend::Delete;
    using MetaStorageBackend::Put;
    using MetaStorageBackend::UpdateFields;
    using MetaStorageBackend::Upsert;

    virtual std::vector<ErrorCode> PutIfAbsent(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept = 0;

    // Conditional write: only processes keys where previous_error_codes[i] == EC_OK.
    // For skipped keys, the returned error code is copied from previous_error_codes.
    virtual std::vector<ErrorCode> Put(const KeyTypeVec &keys,
                                       const FieldMapVec &field_maps,
                                       const std::vector<ErrorCode> &previous_error_codes) noexcept = 0;
    virtual std::vector<ErrorCode> PutIfAbsent(const KeyTypeVec &keys,
                                               const FieldMapVec &field_maps,
                                               const std::vector<ErrorCode> &previous_error_codes) noexcept = 0;
    virtual std::vector<ErrorCode> UpdateFields(const KeyTypeVec &keys,
                                                const FieldMapVec &field_maps,
                                                const std::vector<ErrorCode> &previous_error_codes) noexcept = 0;
    virtual std::vector<ErrorCode> Upsert(const KeyTypeVec &keys,
                                          const FieldMapVec &field_maps,
                                          const std::vector<ErrorCode> &previous_error_codes) noexcept = 0;
    virtual std::vector<ErrorCode> Delete(const KeyTypeVec &keys,
                                          const std::vector<ErrorCode> &previous_error_codes) noexcept = 0;
};

} // namespace kv_cache_manager
