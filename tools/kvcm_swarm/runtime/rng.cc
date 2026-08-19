#include "tools/kvcm_swarm/runtime/rng.h"

namespace kvcm_swarm {

uint64_t HashString(std::string_view value) {
    uint64_t hash = 0xcbf29ce484222325ULL;
    for (const char c : value) {
        hash ^= static_cast<uint64_t>(static_cast<unsigned char>(c));
        hash *= 0x100000001b3ULL;
    }
    uint64_t z = hash + 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

uint64_t SeedDeriver::Derive(std::string_view stream) const {
    Rng rng(root_seed_ ^ HashString(stream));
    return rng.Next();
}

uint64_t SeedDeriver::Derive(std::string_view stream, uint64_t ordinal) const {
    Rng rng((root_seed_ ^ HashString(stream)) ^ (ordinal * 0x9e3779b97f4a7c15ULL));
    return rng.Next();
}

} // namespace kvcm_swarm
