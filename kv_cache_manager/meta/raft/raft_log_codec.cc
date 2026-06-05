#include "kv_cache_manager/meta/raft/raft_log_codec.h"

#include <libnuraft/buffer.hxx>
#include <libnuraft/buffer_serializer.hxx>

#include <exception>
#include <memory>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/meta/cache_location.h"

namespace kv_cache_manager {
namespace raft_meta {

namespace {

constexpr uint8_t kCodecVersion = 1;

// Estimate buffer size up-front so we avoid the resize-and-copy path inside
// buffer. Slightly over-counts, which is fine — buffer::alloc takes a fixed
// size and we just don't fully use it.
size_t EstimatePropertyMapSize(const PropertyMap &props) {
    size_t total = sizeof(uint32_t); // entry count
    for (const auto &[k, v] : props) {
        total += sizeof(uint32_t) + k.size();
        total += sizeof(uint32_t) + v.size();
    }
    return total;
}

size_t EstimateLocationMapSize(const CacheLocationMap &locs) {
    size_t total = sizeof(uint32_t);
    for (const auto &[id, loc] : locs) {
        total += sizeof(uint32_t) + id.size();
        size_t json_len = loc ? loc->ToJsonString().size() : 0;
        total += sizeof(uint32_t) + json_len;
    }
    return total;
}

size_t EstimateLogOpSize(const LogOp &op) {
    size_t total = 1 /*ver*/ + 1 /*op*/ + sizeof(uint32_t) + op.instance_id.size();
    switch (op.type) {
    case OpType::kPut:
    case OpType::kUpsert:
    case OpType::kPutIfAbsent:
        total += sizeof(int64_t);
        total += EstimateLocationMapSize(op.locations);
        total += EstimatePropertyMapSize(op.properties);
        break;
    case OpType::kDelete:
        total += sizeof(int64_t);
        break;
    case OpType::kDeleteLocations:
        total += sizeof(int64_t);
        total += sizeof(uint32_t);
        for (const auto &id : op.location_ids) {
            total += sizeof(uint32_t) + id.size();
        }
        break;
    case OpType::kPutMetaData:
        total += EstimatePropertyMapSize(op.meta_fields);
        break;
    }
    return total;
}

void WriteFieldMap(nuraft::buffer_serializer &bs, const FieldMap &fields) {
    bs.put_u32(static_cast<uint32_t>(fields.size()));
    for (const auto &[k, v] : fields) {
        bs.put_str(k);
        bs.put_str(v);
    }
}

void WriteLocationMap(nuraft::buffer_serializer &bs, const CacheLocationMap &locs) {
    bs.put_u32(static_cast<uint32_t>(locs.size()));
    for (const auto &[id, loc] : locs) {
        bs.put_str(id);
        bs.put_str(loc ? loc->ToJsonString() : std::string());
    }
}

bool ReadFieldMap(nuraft::buffer_serializer &bs, FieldMap &out) {
    uint32_t n = bs.get_u32();
    for (uint32_t i = 0; i < n; ++i) {
        std::string k = bs.get_str();
        std::string v = bs.get_str();
        out.emplace(std::move(k), std::move(v));
    }
    return true;
}

bool ReadLocationMap(nuraft::buffer_serializer &bs, CacheLocationMap &out) {
    uint32_t n = bs.get_u32();
    for (uint32_t i = 0; i < n; ++i) {
        std::string id = bs.get_str();
        std::string json = bs.get_str();
        if (json.empty()) {
            out.emplace(std::move(id), CacheLocationConstPtr{});
            continue;
        }
        auto loc = std::make_shared<CacheLocation>();
        if (!loc->FromJsonString(json)) {
            KVCM_LOG_ERROR("raft_log_codec: failed to parse CacheLocation json[%s]", json.c_str());
            return false;
        }
        out.emplace(std::move(id), std::const_pointer_cast<const CacheLocation>(loc));
    }
    return true;
}

} // namespace

nuraft::ptr<nuraft::buffer> Encode(const LogOp &op) {
    size_t cap = EstimateLogOpSize(op);
    nuraft::ptr<nuraft::buffer> buf = nuraft::buffer::alloc(cap);
    nuraft::buffer_serializer bs(buf);
    bs.put_u8(kCodecVersion);
    bs.put_u8(static_cast<uint8_t>(op.type));
    bs.put_str(op.instance_id);
    switch (op.type) {
    case OpType::kPut:
    case OpType::kUpsert:
    case OpType::kPutIfAbsent:
        bs.put_i64(op.key);
        WriteLocationMap(bs, op.locations);
        WriteFieldMap(bs, op.properties);
        break;
    case OpType::kDelete:
        bs.put_i64(op.key);
        break;
    case OpType::kDeleteLocations:
        bs.put_i64(op.key);
        bs.put_u32(static_cast<uint32_t>(op.location_ids.size()));
        for (const auto &id : op.location_ids) {
            bs.put_str(id);
        }
        break;
    case OpType::kPutMetaData:
        WriteFieldMap(bs, op.meta_fields);
        break;
    }
    return buf;
}

ErrorCode Decode(nuraft::buffer &buf, LogOp &out) {
    try {
        nuraft::buffer_serializer bs(buf);
        uint8_t ver = bs.get_u8();
        if (ver != kCodecVersion) {
            KVCM_LOG_ERROR("raft_log_codec: unknown version[%u]", ver);
            return EC_CORRUPTION;
        }
        uint8_t op_raw = bs.get_u8();
        out.type = static_cast<OpType>(op_raw);
        out.instance_id = bs.get_str();
        switch (out.type) {
        case OpType::kPut:
        case OpType::kUpsert:
        case OpType::kPutIfAbsent:
            out.key = bs.get_i64();
            if (!ReadLocationMap(bs, out.locations)) {
                return EC_CORRUPTION;
            }
            ReadFieldMap(bs, out.properties);
            break;
        case OpType::kDelete:
            out.key = bs.get_i64();
            break;
        case OpType::kDeleteLocations: {
            out.key = bs.get_i64();
            uint32_t n = bs.get_u32();
            out.location_ids.reserve(n);
            for (uint32_t i = 0; i < n; ++i) {
                out.location_ids.push_back(bs.get_str());
            }
            break;
        }
        case OpType::kPutMetaData:
            ReadFieldMap(bs, out.meta_fields);
            break;
        default:
            KVCM_LOG_ERROR("raft_log_codec: unknown op type[%u]", op_raw);
            return EC_CORRUPTION;
        }
    } catch (const std::exception &e) {
        KVCM_LOG_ERROR("raft_log_codec: decode threw[%s]", e.what());
        return EC_CORRUPTION;
    }
    return EC_OK;
}

} // namespace raft_meta
} // namespace kv_cache_manager
