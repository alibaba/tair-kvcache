#pragma once

#include <libnuraft/buffer.hxx>
#include <libnuraft/ptr.hxx>

#include <cstdint>
#include <string>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/meta/types.h"

namespace kv_cache_manager {
namespace raft_meta {

// Op-codes for the meta-storage Raft log envelope. Stable wire enum: append
// only, never reuse a value.
enum class OpType : uint8_t {
    kPut = 1,
    kUpsert = 2,
    kDelete = 3,
    kDeleteLocations = 4,
    kPutMetaData = 5,
    kPutIfAbsent = 6,
    kRegistrySave = 7,
    kRegistryDelete = 8,
};

// One write operation in the raft log. Single op per log entry — batch APIs
// at the MetaStorageBackend boundary expand into one entry per key when the
// backend is the raft variant. This keeps replay simple and lets us treat
// each entry's commit() as an atomic state-machine step.
//
// instance_id scopes the op to one logical Instance: the state machine routes
// each commit to a per-instance MetaLocalBackend, so multiple Instances can
// share a single raft group without leaking keys across Instance boundaries
// (Instance isolation is a hard constraint — see CLAUDE.md).
struct LogOp {
    OpType type = OpType::kPut;
    std::string instance_id;

    // For per-key ops (Put / Upsert / Delete / DeleteLocations / PutIfAbsent).
    KeyType key = 0;
    CacheLocationMap locations;       // Put / Upsert / PutIfAbsent payload
    PropertyMap properties;           // Put / Upsert / PutIfAbsent payload
    LocationIdVector location_ids;    // DeleteLocations payload

    // For PutMetaData.
    FieldMap meta_fields;

    // For RegistrySave / RegistryDelete.
    std::string registry_key;
    std::map<std::string, std::string> registry_fields;
};

// Encode a LogOp into a NuRaft buffer ready to be wrapped in a log_entry.
// The format is little-endian, length-prefixed where lengths are dynamic, so
// it round-trips cleanly through buffer_serializer's get_/put_ family.
nuraft::ptr<nuraft::buffer> Encode(const LogOp &op);

// Decode a NuRaft buffer (cursor at any position; we reset to 0) back into a
// LogOp. Returns EC_OK on success, EC_CORRUPTION on malformed/truncated input
// or unknown op type.
ErrorCode Decode(nuraft::buffer &buf, LogOp &out);

} // namespace raft_meta
} // namespace kv_cache_manager
