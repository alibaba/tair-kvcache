#include "kv_cache_manager/service/util/report_event_json_parser.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>

#if defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#include "kv_cache_manager/service/util/proto_message_json_util.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

namespace kv_cache_manager {
namespace {

using JsonValue = rapidjson::Value;

constexpr size_t kLargeRequestThreshold = 32 * 1024;
constexpr size_t kDefaultJsonPoolChunkSize = 64 * 1024;
constexpr size_t kMaxJsonPoolChunkSize = 4 * 1024 * 1024;
constexpr size_t kDefaultJsonStackSize = 256;
constexpr size_t kMinLargeJsonStackSize = 64 * 1024;
constexpr size_t kMaxJsonStackSize = 1024 * 1024;
constexpr size_t kMaxReusableMutableJsonCapacity = 4 * 1024 * 1024;

struct ThreadLocalMutableJsonBuffer {
    std::string value;
    bool in_use = false;
};

class MutableJsonBufferLease {
public:
    explicit MutableJsonBufferLease(size_t size) {
        if (size <= kMaxReusableMutableJsonCapacity && !thread_local_buffer_.in_use) {
            thread_local_buffer_.in_use = true;
            value_ = &thread_local_buffer_.value;
            reusable_ = true;
            if (value_->capacity() < size) {
                // Avoid std::string's geometric growth retaining more than the
                // explicit per-worker cap after alternating payload sizes.
                std::string replacement;
                replacement.reserve(size);
                value_->swap(replacement);
            }
        } else {
            value_ = &fallback_;
            fallback_.reserve(size);
        }
    }

    MutableJsonBufferLease(const MutableJsonBufferLease &) = delete;
    MutableJsonBufferLease &operator=(const MutableJsonBufferLease &) = delete;

    ~MutableJsonBufferLease() {
        if (reusable_) {
            value_->clear();
            thread_local_buffer_.in_use = false;
        }
    }

    std::string &value() { return *value_; }

private:
    static thread_local ThreadLocalMutableJsonBuffer thread_local_buffer_;
    std::string fallback_;
    std::string *value_ = nullptr;
    bool reusable_ = false;
};

thread_local ThreadLocalMutableJsonBuffer MutableJsonBufferLease::thread_local_buffer_;

struct AsciiScanResult {
    bool is_ascii;
    bool has_nul;
};

AsciiScanResult FinishAsciiScan(std::string_view input, bool has_nul) {
    constexpr uint64_t kHighBits = 0x8080808080808080ULL;
    constexpr uint64_t kLowBits = 0x0101010101010101ULL;
    while (input.size() >= sizeof(uint64_t)) {
        uint64_t word;
        std::memcpy(&word, input.data(), sizeof(word));
        if ((word & kHighBits) != 0) {
            return {false, has_nul};
        }
        if (((word - kLowBits) & ~word & kHighBits) != 0) {
            has_nul = true;
        }
        input.remove_prefix(sizeof(word));
    }
    for (const unsigned char byte : input) {
        if ((byte & 0x80U) != 0) {
            return {false, has_nul};
        }
        if (byte == 0) {
            has_nul = true;
        }
    }
    return {true, has_nul};
}

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2"))) AsciiScanResult ScanAsciiAvx2(std::string_view input) {
    bool has_nul = false;
    const __m256i zero = _mm256_setzero_si256();
    while (input.size() >= 32) {
        const __m256i bytes =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(static_cast<const void *>(input.data())));
        if (_mm256_movemask_epi8(bytes) != 0) {
            return {false, has_nul};
        }
        if (_mm256_movemask_epi8(_mm256_cmpeq_epi8(bytes, zero)) != 0) {
            has_nul = true;
        }
        input.remove_prefix(32);
    }
    return FinishAsciiScan(input, has_nul);
}
#endif

AsciiScanResult ScanAscii(std::string_view input) {
    bool has_nul = false;
#if defined(__aarch64__)
    const uint8x16_t high_bit = vdupq_n_u8(0x80U);
    while (input.size() >= 16) {
        const uint8x16_t bytes = vld1q_u8(reinterpret_cast<const uint8_t *>(input.data()));
        if (vmaxvq_u8(vandq_u8(bytes, high_bit)) != 0) {
            return {false, has_nul};
        }
        if (vminvq_u8(bytes) == 0) {
            has_nul = true;
        }
        input.remove_prefix(16);
    }
#elif (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    if (__builtin_cpu_supports("avx2")) {
        return ScanAsciiAvx2(input);
    }
#if defined(__SSE2__)
    const __m128i zero = _mm_setzero_si128();
    while (input.size() >= 16) {
        const __m128i bytes =
            _mm_loadu_si128(reinterpret_cast<const __m128i *>(static_cast<const void *>(input.data())));
        if (_mm_movemask_epi8(bytes) != 0) {
            return {false, has_nul};
        }
        if (_mm_movemask_epi8(_mm_cmpeq_epi8(bytes, zero)) != 0) {
            has_nul = true;
        }
        input.remove_prefix(16);
    }
#endif
#endif
    return FinishAsciiScan(input, has_nul);
}

template <size_t ExpectedSize>
inline bool NameIs(const JsonValue &name, const char (&expected)[ExpectedSize]) {
    static_assert(ExpectedSize > 1);
    return name.GetStringLength() == ExpectedSize - 1 && std::memcmp(name.GetString(), expected, ExpectedSize - 1) == 0;
}

template <size_t SnakeSize, size_t CamelSize>
inline bool NameIs(const JsonValue &name, const char (&snake_case)[SnakeSize], const char (&camel_case)[CamelSize]) {
    static_assert(SnakeSize > 1 && CamelSize > 1);
    const size_t actual_size = name.GetStringLength();
    const char *actual = name.GetString();
    return (actual_size == SnakeSize - 1 && std::memcmp(actual, snake_case, SnakeSize - 1) == 0) ||
           (actual_size == CamelSize - 1 && std::memcmp(actual, camel_case, CamelSize - 1) == 0);
}

template <typename Setter>
bool SetString(const JsonValue &value, Setter &&setter) {
    if (!value.IsString()) {
        return false;
    }
    setter(value.GetString(), value.GetStringLength());
    return true;
}

bool ParseStorageType(const JsonValue &value, proto::meta::StorageType &out) {
    if (value.IsInt()) {
        switch (value.GetInt()) {
        case proto::meta::ST_UNSPECIFIED:
        case proto::meta::ST_3FS:
        case proto::meta::ST_MOONCAKE:
        case proto::meta::ST_TAIRMEMPOOL:
        case proto::meta::ST_NFS:
        case proto::meta::ST_VCNS_3FS:
        case proto::meta::ST_DUMMY:
        case proto::meta::ST_EVENT_REPORT_L1P5:
        case proto::meta::ST_EVENT_REPORT_L2:
            out = static_cast<proto::meta::StorageType>(value.GetInt());
            return true;
        default:
            return false;
        }
    }
    if (!value.IsString()) {
        return false;
    }
    const std::string_view name(value.GetString(), value.GetStringLength());
    if (name == "ST_UNSPECIFIED") {
        out = proto::meta::ST_UNSPECIFIED;
    } else if (name == "ST_3FS") {
        out = proto::meta::ST_3FS;
    } else if (name == "ST_MOONCAKE") {
        out = proto::meta::ST_MOONCAKE;
    } else if (name == "ST_TAIRMEMPOOL") {
        out = proto::meta::ST_TAIRMEMPOOL;
    } else if (name == "ST_NFS") {
        out = proto::meta::ST_NFS;
    } else if (name == "ST_VCNS_3FS") {
        out = proto::meta::ST_VCNS_3FS;
    } else if (name == "ST_DUMMY") {
        out = proto::meta::ST_DUMMY;
    } else if (name == "ST_EVENT_REPORT_L1P5") {
        out = proto::meta::ST_EVENT_REPORT_L1P5;
    } else if (name == "ST_EVENT_REPORT_L2") {
        out = proto::meta::ST_EVENT_REPORT_L2;
    } else {
        return false;
    }
    return true;
}

bool ParseEventType(const JsonValue &value, proto::meta::ReportEventType &out) {
    if (value.IsInt()) {
        switch (value.GetInt()) {
        case proto::meta::EVENT_UNSPECIFIED:
        case proto::meta::EVENT_NODE_REGISTER:
        case proto::meta::EVENT_BLOCK_ADD:
        case proto::meta::EVENT_BLOCK_DELETE:
        case proto::meta::EVENT_HOST_DOWN:
        case proto::meta::EVENT_HEARTBEAT:
        case proto::meta::EVENT_BLOCK_SNAPSHOT:
            out = static_cast<proto::meta::ReportEventType>(value.GetInt());
            return true;
        default:
            return false;
        }
    }
    if (!value.IsString()) {
        return false;
    }
    const std::string_view name(value.GetString(), value.GetStringLength());
    if (name == "EVENT_UNSPECIFIED") {
        out = proto::meta::EVENT_UNSPECIFIED;
    } else if (name == "EVENT_NODE_REGISTER") {
        out = proto::meta::EVENT_NODE_REGISTER;
    } else if (name == "EVENT_BLOCK_ADD") {
        out = proto::meta::EVENT_BLOCK_ADD;
    } else if (name == "EVENT_BLOCK_DELETE") {
        out = proto::meta::EVENT_BLOCK_DELETE;
    } else if (name == "EVENT_HOST_DOWN") {
        out = proto::meta::EVENT_HOST_DOWN;
    } else if (name == "EVENT_HEARTBEAT") {
        out = proto::meta::EVENT_HEARTBEAT;
    } else if (name == "EVENT_BLOCK_SNAPSHOT") {
        out = proto::meta::EVENT_BLOCK_SNAPSHOT;
    } else {
        return false;
    }
    return true;
}

bool ParseLocationSpec(const JsonValue &value, proto::meta::LocationSpec *out) {
    if (!value.IsObject() || !out) {
        return false;
    }
    bool seen_name = false;
    bool seen_uri = false;
    for (const auto &member : value.GetObject()) {
        if (NameIs(member.name, "name")) {
            if (seen_name ||
                !SetString(member.value, [out](const char *data, size_t size) { out->set_name(data, size); })) {
                return false;
            }
            seen_name = true;
        } else if (NameIs(member.name, "uri")) {
            if (seen_uri ||
                !SetString(member.value, [out](const char *data, size_t size) { out->set_uri(data, size); })) {
                return false;
            }
            seen_uri = true;
        }
    }
    return true;
}

template <typename RepeatedMessage, typename AddFunc>
bool ParseSpecs(const JsonValue &value, RepeatedMessage *out, AddFunc &&add) {
    if (!value.IsArray()) {
        return false;
    }
    out->Reserve(static_cast<int>(value.Size()));
    for (const auto &entry : value.GetArray()) {
        if (!ParseLocationSpec(entry, add())) {
            return false;
        }
    }
    return true;
}

bool ParseNodeRegister(const JsonValue &value, proto::meta::NodeRegisterEventParams *out) {
    if (!value.IsObject() || !out) {
        return false;
    }
    bool seen_mediums = false;
    for (const auto &member : value.GetObject()) {
        if (!NameIs(member.name, "mediums")) {
            continue;
        }
        if (seen_mediums || !member.value.IsArray()) {
            return false;
        }
        seen_mediums = true;
        out->mutable_mediums()->Reserve(static_cast<int>(member.value.Size()));
        for (const auto &medium : member.value.GetArray()) {
            if (!medium.IsString()) {
                return false;
            }
            out->add_mediums(medium.GetString(), medium.GetStringLength());
        }
    }
    return true;
}

bool ParseBlockAdd(const JsonValue &value, proto::meta::BlockAddEventParams *out) {
    if (!value.IsObject() || !out) {
        return false;
    }
    bool seen_block_key = false;
    bool seen_uri = false;
    bool seen_medium = false;
    bool seen_specs = false;
    for (const auto &member : value.GetObject()) {
        if (NameIs(member.name, "block_key", "blockKey")) {
            if (seen_block_key ||
                !SetString(member.value, [out](const char *data, size_t size) { out->set_block_key(data, size); })) {
                return false;
            }
            seen_block_key = true;
        } else if (NameIs(member.name, "uri")) {
            if (seen_uri ||
                !SetString(member.value, [out](const char *data, size_t size) { out->set_uri(data, size); })) {
                return false;
            }
            seen_uri = true;
        } else if (NameIs(member.name, "medium")) {
            if (seen_medium ||
                !SetString(member.value, [out](const char *data, size_t size) { out->set_medium(data, size); })) {
                return false;
            }
            seen_medium = true;
        } else if (NameIs(member.name, "specs")) {
            if (seen_specs || !ParseSpecs(member.value, out->mutable_specs(), [out] { return out->add_specs(); })) {
                return false;
            }
            seen_specs = true;
        }
    }
    return true;
}

bool ParseBlockDelete(const JsonValue &value, proto::meta::BlockDeleteEventParams *out) {
    if (!value.IsObject() || !out) {
        return false;
    }
    bool seen_block_key = false;
    bool seen_medium = false;
    bool seen_spec_names = false;
    for (const auto &member : value.GetObject()) {
        if (NameIs(member.name, "block_key", "blockKey")) {
            if (seen_block_key ||
                !SetString(member.value, [out](const char *data, size_t size) { out->set_block_key(data, size); })) {
                return false;
            }
            seen_block_key = true;
        } else if (NameIs(member.name, "medium")) {
            if (seen_medium ||
                !SetString(member.value, [out](const char *data, size_t size) { out->set_medium(data, size); })) {
                return false;
            }
            seen_medium = true;
        } else if (NameIs(member.name, "spec_names", "specNames")) {
            if (seen_spec_names || !member.value.IsArray()) {
                return false;
            }
            seen_spec_names = true;
            out->mutable_spec_names()->Reserve(static_cast<int>(member.value.Size()));
            for (const auto &name : member.value.GetArray()) {
                if (!name.IsString()) {
                    return false;
                }
                out->add_spec_names(name.GetString(), name.GetStringLength());
            }
        }
    }
    return true;
}

bool ParseSnapshotItem(const JsonValue &value, proto::meta::BlockSnapshotItem *out) {
    if (!value.IsObject() || !out) {
        return false;
    }
    bool seen_block_key = false;
    bool seen_medium = false;
    bool seen_specs = false;
    for (const auto &member : value.GetObject()) {
        if (NameIs(member.name, "block_key", "blockKey")) {
            if (seen_block_key ||
                !SetString(member.value, [out](const char *data, size_t size) { out->set_block_key(data, size); })) {
                return false;
            }
            seen_block_key = true;
        } else if (NameIs(member.name, "medium")) {
            if (seen_medium ||
                !SetString(member.value, [out](const char *data, size_t size) { out->set_medium(data, size); })) {
                return false;
            }
            seen_medium = true;
        } else if (NameIs(member.name, "specs")) {
            if (seen_specs || !ParseSpecs(member.value, out->mutable_specs(), [out] { return out->add_specs(); })) {
                return false;
            }
            seen_specs = true;
        }
    }
    return true;
}

bool ParseBlockSnapshot(const JsonValue &value, proto::meta::BlockSnapshotEventParams *out) {
    if (!value.IsObject() || !out) {
        return false;
    }
    bool seen_medium = false;
    bool seen_blocks = false;
    for (const auto &member : value.GetObject()) {
        if (NameIs(member.name, "medium")) {
            if (seen_medium ||
                !SetString(member.value, [out](const char *data, size_t size) { out->set_medium(data, size); })) {
                return false;
            }
            seen_medium = true;
        } else if (NameIs(member.name, "blocks")) {
            if (seen_blocks || !member.value.IsArray()) {
                return false;
            }
            seen_blocks = true;
            out->mutable_blocks()->Reserve(static_cast<int>(member.value.Size()));
            for (const auto &block : member.value.GetArray()) {
                if (!ParseSnapshotItem(block, out->add_blocks())) {
                    return false;
                }
            }
        }
    }
    return true;
}

bool ParseHeartbeat(const JsonValue &value, proto::meta::HeartbeatEventParams *out) {
    if (!value.IsObject() || !out) {
        return false;
    }
    bool seen_system_status = false;
    for (const auto &member : value.GetObject()) {
        if (!NameIs(member.name, "system_status", "systemStatus")) {
            continue;
        }
        if (seen_system_status || !member.value.IsObject()) {
            return false;
        }
        seen_system_status = true;
        auto *status = out->mutable_system_status();
        for (const auto &entry : member.value.GetObject()) {
            if (!entry.value.IsString()) {
                return false;
            }
            std::string key(entry.name.GetString(), entry.name.GetStringLength());
            // JsonStringToMessage rejects duplicate protobuf map keys. Do not
            // silently apply last-value-wins here merely because RapidJSON's
            // DOM preserves duplicate object members: request acceptance must
            // not depend on whether the specialized parser was selected.
            if (status->find(key) != status->end()) {
                return false;
            }
            (*status)[std::move(key)] = std::string(entry.value.GetString(), entry.value.GetStringLength());
        }
    }
    return true;
}

bool ParseEvent(const JsonValue &value, proto::meta::EventItem *out) {
    if (!value.IsObject() || !out) {
        return false;
    }
    bool seen_event_type = false;
    bool seen_params = false;
    for (const auto &member : value.GetObject()) {
        if (NameIs(member.name, "event_type", "eventType")) {
            proto::meta::ReportEventType event_type;
            if (seen_event_type || !ParseEventType(member.value, event_type)) {
                return false;
            }
            out->set_event_type(event_type);
            seen_event_type = true;
        } else if (NameIs(member.name, "node_register", "nodeRegister")) {
            if (seen_params || !ParseNodeRegister(member.value, out->mutable_node_register())) {
                return false;
            }
            seen_params = true;
        } else if (NameIs(member.name, "block_add", "blockAdd")) {
            if (seen_params || !ParseBlockAdd(member.value, out->mutable_block_add())) {
                return false;
            }
            seen_params = true;
        } else if (NameIs(member.name, "block_delete", "blockDelete")) {
            if (seen_params || !ParseBlockDelete(member.value, out->mutable_block_delete())) {
                return false;
            }
            seen_params = true;
        } else if (NameIs(member.name, "host_down", "hostDown")) {
            if (seen_params || !member.value.IsObject()) {
                return false;
            }
            out->mutable_host_down();
            seen_params = true;
        } else if (NameIs(member.name, "heartbeat")) {
            if (seen_params || !ParseHeartbeat(member.value, out->mutable_heartbeat())) {
                return false;
            }
            seen_params = true;
        } else if (NameIs(member.name, "block_snapshot", "blockSnapshot")) {
            if (seen_params || !ParseBlockSnapshot(member.value, out->mutable_block_snapshot())) {
                return false;
            }
            seen_params = true;
        }
    }
    return true;
}

bool ParseRequest(const JsonValue &root, proto::meta::ReportEventRequest *out) {
    if (!root.IsObject() || !out) {
        return false;
    }
    bool seen_trace_id = false;
    bool seen_instance_id = false;
    bool seen_host_ip_port = false;
    bool seen_events = false;
    bool seen_storage_type = false;
    for (const auto &member : root.GetObject()) {
        if (NameIs(member.name, "trace_id", "traceId")) {
            if (seen_trace_id ||
                !SetString(member.value, [out](const char *data, size_t size) { out->set_trace_id(data, size); })) {
                return false;
            }
            seen_trace_id = true;
        } else if (NameIs(member.name, "instance_id", "instanceId")) {
            if (seen_instance_id ||
                !SetString(member.value, [out](const char *data, size_t size) { out->set_instance_id(data, size); })) {
                return false;
            }
            seen_instance_id = true;
        } else if (NameIs(member.name, "host_ip_port", "hostIpPort")) {
            if (seen_host_ip_port ||
                !SetString(member.value, [out](const char *data, size_t size) { out->set_host_ip_port(data, size); })) {
                return false;
            }
            seen_host_ip_port = true;
        } else if (NameIs(member.name, "events")) {
            if (seen_events || !member.value.IsArray()) {
                return false;
            }
            seen_events = true;
            out->mutable_events()->Reserve(static_cast<int>(member.value.Size()));
            for (const auto &event : member.value.GetArray()) {
                if (!ParseEvent(event, out->add_events())) {
                    return false;
                }
            }
        } else if (NameIs(member.name, "storage_type", "storageType")) {
            proto::meta::StorageType storage_type;
            if (seen_storage_type || !ParseStorageType(member.value, storage_type)) {
                return false;
            }
            out->set_storage_type(storage_type);
            seen_storage_type = true;
        }
    }
    return true;
}

} // namespace

bool ReportEventJsonParser::TryFromJson(std::string_view json, proto::meta::ReportEventRequest *message) {
    if (!message || json.empty()) {
        return false;
    }
    message->Clear();
    const bool is_large = json.size() >= kLargeRequestThreshold;
    const size_t pool_chunk_size = is_large ? std::min(json.size(), kMaxJsonPoolChunkSize) : kDefaultJsonPoolChunkSize;
    const size_t stack_size =
        is_large ? std::clamp(json.size() / 8, kMinLargeJsonStackSize, kMaxJsonStackSize) : kDefaultJsonStackSize;
    rapidjson::MemoryPoolAllocator<> allocator(pool_chunk_size);
    rapidjson::Document document(&allocator, stack_size);
    const AsciiScanResult ascii = ScanAscii(json);
    if (is_large && ascii.is_ascii && !ascii.has_nul) {
        // RapidJSON's read-only parser copies every decoded key/value into the
        // DOM pool before ParseRequest copies it into protobuf. A single
        // contiguous mutable copy lets in-situ parsing reference/unescape the
        // strings in that buffer instead, removing thousands of small DOM
        // string copies. The buffer outlives ParseRequest below. Raw NUL bytes
        // stay on the length-aware path so an early C-string terminator can
        // never make a malformed body look valid.
        MutableJsonBufferLease mutable_json(json.size());
        mutable_json.value().assign(json.data(), json.size());
        document.ParseInsitu<rapidjson::kParseDefaultFlags>(mutable_json.value().data());
        if (document.HasParseError()) {
            return false;
        }
        return ParseRequest(document, message);
    } else if (ascii.is_ascii) {
        // ASCII is a strict subset of UTF-8. Skipping RapidJSON's per-codepoint
        // validator avoids a full branch-heavy decode on the overwhelmingly
        // common URI/key payload without accepting any invalid byte sequence.
        document.Parse<rapidjson::kParseDefaultFlags>(json.data(), json.size());
    } else {
        document.Parse<rapidjson::kParseValidateEncodingFlag>(json.data(), json.size());
    }
    return !document.HasParseError() && ParseRequest(document, message);
}

bool ReportEventJsonParser::FromJson(std::string_view json, proto::meta::ReportEventRequest *message) {
    if (!message) {
        return false;
    }
    if (TryFromJson(json, message)) {
        return true;
    }
    message->Clear();
    return ProtoMessageJsonUtil::FromJson(json, message);
}

bool ReportEventJsonParser::FromMutableNullTerminatedJson(char *json,
                                                          size_t size,
                                                          proto::meta::ReportEventRequest *message) {
    if (!message || !json || size == 0) {
        return false;
    }

    const std::string_view view(json, size);
    if (size < kLargeRequestThreshold) {
        // The immutable parser already performs its own ASCII/UTF-8 scan.
        // Avoid scanning small heartbeat/register bodies twice merely because
        // the HTTP handler also supports the large mutable-body fast path.
        return FromJson(view, message);
    }
    const AsciiScanResult ascii = ScanAscii(view);
    if (!ascii.is_ascii || ascii.has_nul) {
        // Non-ASCII and raw-NUL inputs retain the existing length-aware
        // validation path. They are rare enough that preserving one shared
        // compatibility path is preferable to plumbing scan state through
        // both decoders.
        return FromJson(view, message);
    }

    // cinatra 0.5.5 stores the request body in std::string and exposes a view,
    // so the byte immediately after the view is its terminator. Keep the check
    // here as a defensive guard against a future transport implementation that
    // no longer satisfies the explicit API contract above.
    if (json[size] != '\0') {
        return FromJson(view, message);
    }

    message->Clear();
    const size_t pool_chunk_size = std::min(size, kMaxJsonPoolChunkSize);
    const size_t stack_size = std::clamp(size / 8, kMinLargeJsonStackSize, kMaxJsonStackSize);
    rapidjson::MemoryPoolAllocator<> allocator(pool_chunk_size);
    rapidjson::Document document(&allocator, stack_size);
    document.ParseInsitu<rapidjson::kParseDefaultFlags>(json);
    if (document.HasParseError()) {
        // Both parsers require valid JSON. The mutable source may already have
        // been changed by RapidJSON, so it cannot be passed to the protobuf
        // fallback; a syntax error is not a compatibility fallback case.
        return false;
    }
    if (ParseRequest(document, message)) {
        return true;
    }

    // The fast converter intentionally delegates rare protobuf-JSON spellings
    // such as null fields and unknown enum names. In-situ parsing has changed
    // the source buffer, but the DOM still represents the complete JSON value;
    // serialize that value only on this rare path before invoking the generic
    // protobuf parser. Unknown fields, duplicate members and escaped strings
    // remain represented in the DOM and therefore preserve fallback behavior.
    rapidjson::StringBuffer normalized_json;
    rapidjson::Writer<rapidjson::StringBuffer> writer(normalized_json);
    if (!document.Accept(writer)) {
        return false;
    }
    message->Clear();
    return ProtoMessageJsonUtil::FromJson(std::string_view(normalized_json.GetString(), normalized_json.GetSize()),
                                          message);
}

} // namespace kv_cache_manager
