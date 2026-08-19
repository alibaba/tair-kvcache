// V6D key derivation, byte-for-byte compatible with the fixed Vineyard
// revision's Connector contract:
//
//   object_key = block_hash + "_" + group_id
//   block_key  = signed_big_endian_int64(SHA256(object_key)[0:8])
//
// The same content in the same group always yields the same key; the same
// content in different groups always yields different keys.
#pragma once

#include <cstdint>
#include <string>

namespace kvcm_swarm {

// Minimal SHA-256 (only the digest prefix is needed, but the full digest is
// produced so golden vectors can be asserted).
void Sha256(const void *data, size_t size, unsigned char out[32]);
std::string Sha256Hex(const std::string &value);

std::string MakeObjectKey(const std::string &block_hash, const std::string &group_id);
int64_t ObjectKeyToBlockKey(const std::string &object_key);

// Lowercase 16-char hex rendering of a 64-bit chain hash, mirroring
// `block_hash.hex()` on the engine side.
std::string BlockHashHex(uint64_t chain_hash);

} // namespace kvcm_swarm
