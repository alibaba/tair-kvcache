// Deterministic RNG streams.
//
// Every logical entity (deployment, process, arrival scheduler, session, and
// each independent sub-stream of a session) derives its own generator from the
// global seed plus a stable string id. Planned inputs are therefore
// reproducible without depending on real network completion order.
#pragma once

#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>

namespace kvcm_swarm {

// SplitMix64: used both as the seed mixer and as the generator core.
class Rng {
public:
    Rng() : state_(0x9e3779b97f4a7c15ULL) {}
    explicit Rng(uint64_t seed) : state_(seed) {}

    uint64_t Next() {
        state_ += 0x9e3779b97f4a7c15ULL;
        uint64_t z = state_;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

    // Uniform in [0, 1).
    double NextDouble() { return static_cast<double>(Next() >> 11) * (1.0 / 9007199254740992.0); }

    // Uniform in the closed integer interval [low, high].
    uint64_t NextInRange(uint64_t low, uint64_t high) {
        if (high <= low) {
            return low;
        }
        const uint64_t span = high - low + 1;
        if (span == 0) { // full 64-bit range
            return Next();
        }
        // Rejection sampling keeps the distribution exactly uniform.
        const uint64_t limit = UINT64_MAX - (UINT64_MAX % span);
        uint64_t value = Next();
        while (value >= limit) {
            value = Next();
        }
        return low + (value % span);
    }

    // Exponential inter-arrival time with the given rate (events per unit).
    double NextExponential(double rate) {
        if (rate <= 0.0) {
            return 0.0;
        }
        double u = NextDouble();
        if (u <= 0.0) {
            u = 1.0 / 9007199254740992.0;
        }
        return -std::log(u) / rate;
    }

    uint64_t state() const { return state_; }

private:
    uint64_t state_;
};

// Stable 64-bit hash of a string (FNV-1a followed by a SplitMix64 finaliser).
uint64_t HashString(std::string_view value);

// Derives independent seeds from the run seed and a stable entity path.
class SeedDeriver {
public:
    explicit SeedDeriver(uint64_t root_seed) : root_seed_(root_seed) {}

    uint64_t Derive(std::string_view stream) const;
    uint64_t Derive(std::string_view stream, uint64_t ordinal) const;
    Rng MakeRng(std::string_view stream) const { return Rng(Derive(stream)); }
    Rng MakeRng(std::string_view stream, uint64_t ordinal) const { return Rng(Derive(stream, ordinal)); }

    uint64_t root_seed() const { return root_seed_; }

private:
    uint64_t root_seed_;
};

} // namespace kvcm_swarm
