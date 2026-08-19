#include "tools/kvcm_swarm/clients/v6d/key_mapper.h"

#include <cstring>

namespace kvcm_swarm {
namespace {

constexpr uint32_t kSha256K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

inline uint32_t RotR(uint32_t value, uint32_t bits) { return (value >> bits) | (value << (32 - bits)); }

void Sha256Transform(uint32_t state[8], const unsigned char block[64]) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = (static_cast<uint32_t>(block[i * 4]) << 24) | (static_cast<uint32_t>(block[i * 4 + 1]) << 16) |
               (static_cast<uint32_t>(block[i * 4 + 2]) << 8) | static_cast<uint32_t>(block[i * 4 + 3]);
    }
    for (int i = 16; i < 64; ++i) {
        const uint32_t s0 = RotR(w[i - 15], 7) ^ RotR(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = RotR(w[i - 2], 17) ^ RotR(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    for (int i = 0; i < 64; ++i) {
        const uint32_t s1 = RotR(e, 6) ^ RotR(e, 11) ^ RotR(e, 25);
        const uint32_t ch = (e & f) ^ ((~e) & g);
        const uint32_t temp1 = h + s1 + ch + kSha256K[i] + w[i];
        const uint32_t s0 = RotR(a, 2) ^ RotR(a, 13) ^ RotR(a, 22);
        const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t temp2 = s0 + maj;
        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

} // namespace

void Sha256(const void *data, size_t size, unsigned char out[32]) {
    uint32_t state[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au, 0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    const auto *bytes = static_cast<const unsigned char *>(data);
    size_t offset = 0;
    while (offset + 64 <= size) {
        Sha256Transform(state, bytes + offset);
        offset += 64;
    }
    unsigned char tail[128];
    const size_t remaining = size - offset;
    std::memcpy(tail, bytes + offset, remaining);
    tail[remaining] = 0x80;
    size_t tail_size = remaining + 1;
    const size_t pad_to = (remaining + 1 <= 56) ? 56 : 120;
    std::memset(tail + tail_size, 0, pad_to - tail_size);
    tail_size = pad_to;
    const uint64_t bit_length = static_cast<uint64_t>(size) * 8;
    for (int i = 0; i < 8; ++i) {
        tail[tail_size + i] = static_cast<unsigned char>((bit_length >> (56 - 8 * i)) & 0xff);
    }
    tail_size += 8;
    for (size_t i = 0; i < tail_size; i += 64) {
        Sha256Transform(state, tail + i);
    }
    for (int i = 0; i < 8; ++i) {
        out[i * 4] = static_cast<unsigned char>((state[i] >> 24) & 0xff);
        out[i * 4 + 1] = static_cast<unsigned char>((state[i] >> 16) & 0xff);
        out[i * 4 + 2] = static_cast<unsigned char>((state[i] >> 8) & 0xff);
        out[i * 4 + 3] = static_cast<unsigned char>(state[i] & 0xff);
    }
}

std::string Sha256Hex(const std::string &value) {
    unsigned char digest[32];
    Sha256(value.data(), value.size(), digest);
    static const char *hex = "0123456789abcdef";
    std::string out;
    out.resize(64);
    for (int i = 0; i < 32; ++i) {
        out[i * 2] = hex[digest[i] >> 4];
        out[i * 2 + 1] = hex[digest[i] & 0x0f];
    }
    return out;
}

std::string MakeObjectKey(const std::string &block_hash, const std::string &group_id) {
    return block_hash + "_" + group_id;
}

int64_t ObjectKeyToBlockKey(const std::string &object_key) {
    unsigned char digest[32];
    Sha256(object_key.data(), object_key.size(), digest);
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value = (value << 8) | static_cast<uint64_t>(digest[i]);
    }
    return static_cast<int64_t>(value);
}

std::string BlockHashHex(uint64_t chain_hash) {
    static const char *hex = "0123456789abcdef";
    std::string out;
    out.resize(16);
    for (int i = 0; i < 16; ++i) {
        out[15 - i] = hex[(chain_hash >> (i * 4)) & 0x0f];
    }
    return out;
}

} // namespace kvcm_swarm
