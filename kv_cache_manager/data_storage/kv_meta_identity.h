#pragma once

#include <array>
#include <charconv>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>

#include "kv_cache_manager/common/hash/hash.h"

namespace kv_cache_manager {

// These values are part of the persisted KVMeta ownership schema and are
// shared by the server manager and the standalone data-plane client. Keep the
// derivation in this low-level header so the two trust boundaries cannot drift
// to different instance identities.
inline constexpr std::string_view kKvMetaInternalInstancePrefix = "__kv_meta_v1__";
inline constexpr std::uint64_t kKvMetaInstancePathHashSeed = 0x6e91'ca34'0bd7'52f8ULL;
inline constexpr std::uint64_t kKvMetaObjectKeyHashSeed = 0x8bc5'1f2d'671a'94e3ULL;
// A single 64-bit hash remains the metadata shard key for compatibility, but
// it is not strong enough to distinguish physical objects at cache scale.
// New physical names carry four fixed-seed hashes over the internal
// instance id and complete logical key.  The resulting 256-bit fingerprint is
// a non-cryptographic collision defence (not an authentication primitive) and
// is still short enough for PACE's 128-byte allocation-token limit. Durable
// ownership continues to be proven by the reversibly encoded complete key in
// metadata.
inline constexpr std::array<std::uint64_t, 4> kKvMetaObjectFingerprintSeeds{
    0x2418'7f6a'91cd'e305ULL,
    0xd773'5ae1'46b8'0c2fULL,
    0x59e4'02bc'af17'68d3ULL,
    0xa61c'd935'72ef'4b08ULL,
};
inline constexpr std::size_t kKvMetaObjectFingerprintHexChars = 64;
inline constexpr std::size_t kKvMetaCanonicalUint64MaxHexChars = 16;
inline constexpr std::size_t kKvMetaCapabilityTokenEntropyBytes = 16;
inline constexpr std::size_t kKvMetaCapabilityTokenHexChars = kKvMetaCapabilityTokenEntropyBytes * 2;
// Retain the original public constant name for source compatibility. Its value
// is the encoded character count (not the entropy-byte count).
inline constexpr std::size_t kKvMetaObjectNonceBytes = kKvMetaCapabilityTokenHexChars;
inline constexpr std::size_t kKvMetaObjectNamespacePrefixBytes = sizeof("kvmeta/") - 1;
// Worst case: kvmeta/<16-char instance hash>/<64-char fingerprint>/<32-char
// generation>. TairMempool's exact allocation contract accepts at most 128
// token bytes; make a future format expansion fail at compile time instead of
// turning every PACE allocation into a production-only invalid-argument error.
inline constexpr std::size_t kKvMetaMaxPhysicalObjectKeyBytes =
    kKvMetaObjectNamespacePrefixBytes + kKvMetaCanonicalUint64MaxHexChars + 1 + kKvMetaObjectFingerprintHexChars + 1 +
    kKvMetaObjectNonceBytes;
inline constexpr std::size_t kKvMetaExactAllocationTokenLimitBytes = 128;
static_assert(kKvMetaMaxPhysicalObjectKeyBytes == 121);
static_assert(kKvMetaMaxPhysicalObjectKeyBytes <= kKvMetaExactAllocationTokenLimitBytes);

// The complete prefix is reserved for KVMeta. Legacy control-plane callers
// must not mutate even a malformed/future version under this namespace.
inline bool HasKvMetaReservedInstancePrefix(std::string_view instance_id) noexcept {
    return instance_id.size() >= kKvMetaInternalInstancePrefix.size() &&
           instance_id.compare(0, kKvMetaInternalInstancePrefix.size(), kKvMetaInternalInstancePrefix) == 0;
}

inline bool HasKvMetaInternalInstanceId(std::string_view instance_id) noexcept {
    if (instance_id.size() <= kKvMetaInternalInstancePrefix.size() || !HasKvMetaReservedInstancePrefix(instance_id)) {
        return false;
    }
    const std::size_t encoded_size = instance_id.size() - kKvMetaInternalInstancePrefix.size();
    if ((encoded_size & 1U) != 0) {
        return false;
    }
    for (std::size_t i = kKvMetaInternalInstancePrefix.size(); i < instance_id.size(); ++i) {
        const char c = instance_id[i];
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) {
            return false;
        }
    }
    return true;
}

inline std::string KvMetaHexEncode(std::string_view input) {
    if (input.size() > std::numeric_limits<std::size_t>::max() / 2) {
        return {};
    }
    static constexpr char kHex[] = "0123456789abcdef";
    std::string output(input.size() * 2, '\0');
    for (std::size_t i = 0; i < input.size(); ++i) {
        const auto value = static_cast<unsigned char>(input[i]);
        output[2 * i] = kHex[value >> 4];
        output[2 * i + 1] = kHex[value & 0x0f];
    }
    return output;
}

inline std::string BuildKvMetaInternalInstanceId(std::string_view public_instance_id) {
    if (public_instance_id.empty() ||
        public_instance_id.size() >
            (std::numeric_limits<std::size_t>::max() - kKvMetaInternalInstancePrefix.size()) / 2) {
        return {};
    }
    const std::string encoded = KvMetaHexEncode(public_instance_id);
    if (encoded.empty()) {
        return {};
    }
    std::string result;
    result.reserve(kKvMetaInternalInstancePrefix.size() + public_instance_id.size() * 2);
    result.append(kKvMetaInternalInstancePrefix);
    result.append(encoded);
    return result;
}

inline std::uint64_t HashKvMetaInternalInstancePath(std::string_view internal_instance_id) noexcept {
    return Hash64(internal_instance_id.data(), internal_instance_id.size(), kKvMetaInstancePathHashSeed);
}

inline std::string KvMetaUint64ToCanonicalHex(std::uint64_t value) {
    std::array<char, 16> buffer{};
    const auto converted = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value, 16);
    return converted.ec == std::errc{} ? std::string(buffer.data(), converted.ptr) : std::string{};
}

inline void KvMetaAppendUint64FixedHex(std::string &output, std::uint64_t value) {
    static constexpr char kHex[] = "0123456789abcdef";
    const std::size_t begin = output.size();
    output.resize(begin + 16);
    for (std::size_t i = 0; i < 16; ++i) {
        const std::size_t shift = (15 - i) * 4;
        output[begin + i] = kHex[(value >> shift) & 0x0fU];
    }
}

inline std::string BuildKvMetaInstancePathHash(std::string_view public_instance_id) {
    const std::string internal_instance_id = BuildKvMetaInternalInstanceId(public_instance_id);
    if (internal_instance_id.empty()) {
        return {};
    }
    return KvMetaUint64ToCanonicalHex(HashKvMetaInternalInstancePath(internal_instance_id));
}

inline std::string BuildKvMetaLegacyObjectKeyPrefixFromInternal(std::string_view internal_instance_id,
                                                                std::string_view public_key) {
    if (internal_instance_id.empty() || public_key.empty()) {
        return {};
    }
    const std::string instance_hash = KvMetaUint64ToCanonicalHex(HashKvMetaInternalInstancePath(internal_instance_id));
    if (instance_hash.empty()) {
        return {};
    }
    const std::string key_hash =
        KvMetaUint64ToCanonicalHex(Hash64(public_key.data(), public_key.size(), kKvMetaObjectKeyHashSeed));
    if (key_hash.empty()) {
        return {};
    }
    return "kvmeta/" + instance_hash + "/" + key_hash + "/";
}

inline std::string BuildKvMetaLegacyObjectKeyPrefix(std::string_view public_instance_id, std::string_view public_key) {
    const std::string internal_instance_id = BuildKvMetaInternalInstanceId(public_instance_id);
    return BuildKvMetaLegacyObjectKeyPrefixFromInternal(internal_instance_id, public_key);
}

// Prefix shared by every independently allocated generation of one logical
// key. The nonce follows this prefix and is intentionally unpredictable. The
// fingerprint includes the complete internal instance id and logical key, so
// even an instance-path-hash collision cannot make another tenant's URI pass
// the high-level client's pre-I/O ownership check.
inline std::string BuildKvMetaObjectKeyPrefixFromInternal(std::string_view internal_instance_id,
                                                          std::string_view public_key) {
    if (internal_instance_id.empty() || public_key.empty() ||
        internal_instance_id.size() >= std::numeric_limits<std::size_t>::max() ||
        public_key.size() > std::numeric_limits<std::size_t>::max() - internal_instance_id.size() - 1) {
        return {};
    }
    const std::string instance_hash = KvMetaUint64ToCanonicalHex(HashKvMetaInternalInstancePath(internal_instance_id));
    if (instance_hash.empty()) {
        return {};
    }

    // The NUL separator is unambiguous because an internal instance id has a
    // fixed ASCII prefix followed by canonical hexadecimal text. Logical keys
    // remain arbitrary bytes, including embedded NULs.
    std::string bound_key;
    bound_key.reserve(internal_instance_id.size() + 1 + public_key.size());
    bound_key.append(internal_instance_id);
    bound_key.push_back('\0');
    bound_key.append(public_key);

    std::string fingerprint;
    fingerprint.reserve(kKvMetaObjectFingerprintHexChars);
    for (const std::uint64_t seed : kKvMetaObjectFingerprintSeeds) {
        KvMetaAppendUint64FixedHex(fingerprint, Hash64(bound_key.data(), bound_key.size(), seed));
    }
    if (fingerprint.size() != kKvMetaObjectFingerprintHexChars) {
        return {};
    }
    return "kvmeta/" + instance_hash + "/" + fingerprint + "/";
}

inline std::string BuildKvMetaObjectKeyPrefix(std::string_view public_instance_id, std::string_view public_key) {
    const std::string internal_instance_id = BuildKvMetaInternalInstanceId(public_instance_id);
    return BuildKvMetaObjectKeyPrefixFromInternal(internal_instance_id, public_key);
}

} // namespace kv_cache_manager
