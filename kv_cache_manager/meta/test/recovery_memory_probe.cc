// Subprocess fixture for recovery_memory_test.py. No Redis or production data.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "kv_cache_manager/config/meta_storage_backend_config.h"
#include "kv_cache_manager/meta/meta_local_backend.h"
#include "kv_cache_manager/meta/meta_storage_backend_manager.h"
#include "kv_cache_manager/meta/utils.h"

namespace kv_cache_manager {
namespace {
constexpr size_t kKeys = 262144;
constexpr size_t kBatch = 256;
constexpr unsigned kArenas = 4;
using Control = int (*)(const char *, void *, size_t *, void *, size_t);
Control control;

void Require(bool ok, const char *message) {
    if (!ok) {
        std::fprintf(stderr, "recovery memory regression: %s\n", message);
        std::exit(1);
    }
}

template <class T>
T Read(const char *name) {
    T value{};
    size_t size = sizeof(value);
    Require(control(name, &value, &size, nullptr, 0) == 0 && size == sizeof(value), name);
    return value;
}

void Bind(unsigned arena) {
    Require(control("thread.arena", nullptr, nullptr, &arena, sizeof(arena)) == 0, "bind arena");
}

void Check(const std::vector<ErrorCode> &results, size_t expected) {
    Require(results.size() == expected &&
                std::all_of(results.begin(), results.end(), [](ErrorCode ec) { return ec == EC_OK; }),
            "backend operation failed");
}

FieldMap Payload(int generation) {
    auto location = std::make_shared<CacheLocation>();
    location->set_id(std::string(48, 'l'));
    location->set_status(CacheLocationStatus::CLS_SERVING);
    location->set_type(DataStorageType::DATA_STORAGE_TYPE_HF3FS);
    location->set_spec_size(1);
    location->set_location_specs({LocationSpec("default", std::string(120, 'u') + char('0' + generation))});
    return SerializeToFieldMap({{location->id(), location}}, {{"property", std::string(48, 'p')}});
}

// Generate SCAN/Get responses on demand, but use the production deserializer.
// Writes are acknowledged without retaining a second in-memory copy of data.
class SyntheticPersistentBackend : public MetaLocalBackend {
public:
    FieldMap payload = Payload(0);
    ErrorCode ListKeys(RequestContext *,
                       const std::string &cursor,
                       int64_t limit,
                       std::string &next,
                       KeyTypeVec &keys) noexcept override {
        const size_t begin = std::stoull(cursor);
        const size_t end = std::min(kKeys, begin + limit);
        keys.clear();
        for (size_t i = begin; i < end; ++i) {
            keys.push_back(i);
        }
        next = end == kKeys ? "0" : std::to_string(end);
        return EC_OK;
    }

    std::vector<ErrorCode> Get(RequestContext *,
                               const KeyTypeVec &keys,
                               CacheLocationMapVector &locations,
                               PropertyMapVector &properties) noexcept override {
        locations.resize(keys.size());
        properties.resize(keys.size());
        std::vector<ErrorCode> results;
        for (size_t i = 0; i < keys.size(); ++i) {
            results.push_back(DeserializeFieldMap(payload, locations[i], properties[i]));
        }
        return results;
    }

    std::vector<ErrorCode> Upsert(RequestContext *,
                                  const KeyTypeVec &keys,
                                  const CacheLocationMapVector &,
                                  const PropertyMapVector &) noexcept override {
        return std::vector<ErrorCode>(keys.size(), EC_OK);
    }
};

void Snapshot(const char *phase, MetaLocalBackend &cache) {
    uint64_t epoch = 1;
    Require(control("epoch", nullptr, nullptr, &epoch, sizeof(epoch)) == 0, "refresh stats");
    size_t rss = 0;
    std::ifstream rollup("/proc/self/smaps_rollup");
    std::string line;
    while (std::getline(rollup, line)) {
        if (line.compare(0, 4, "Rss:") == 0) {
            rss = std::stoull(line.substr(4)) * 1024;
        }
    }
    std::printf(
        "{\"phase\":\"%s\",\"keys\":%zu,\"charge\":%zu,\"allocated\":%zu,\"active\":%zu,\"rss\":%zu,\"arenas\":[",
        phase,
        cache.cache_->GetOccupancyCount(),
        cache.GetMemUsage(),
        Read<size_t>("stats.allocated"),
        Read<size_t>("stats.active"),
        rss);
    for (unsigned arena = 0; arena < kArenas; ++arena) {
        const auto prefix = "stats.arenas." + std::to_string(arena) + ".";
        size_t allocated =
            Read<size_t>((prefix + "small.allocated").c_str()) + Read<size_t>((prefix + "large.allocated").c_str());
        std::printf("%s{\"allocated\":%zu,\"active\":%zu}",
                    arena ? "," : "",
                    allocated,
                    Read<size_t>((prefix + "pactive").c_str()) * Read<size_t>("arenas.page"));
    }
    std::puts("]}");
}

void Update(MetaStorageBackendManager &manager, int generation) {
    std::mutex write_mutex;
    std::vector<std::thread> threads;
    for (unsigned worker = 0; worker < kArenas; ++worker) {
        threads.emplace_back([&, worker] {
            Bind(worker);
            const auto payload = Payload(generation);
            BatchMetaData batch;
            auto submit = [&] {
                if (batch.batch_keys.empty()) {
                    return;
                }
                // Respect the manager's caller-serialization contract. This is
                // a memory regression, not a throughput benchmark.
                std::lock_guard<std::mutex> lock(write_mutex);
                Check(manager.Upsert(nullptr, batch), batch.batch_keys.size());
                batch = BatchMetaData{};
            };
            for (size_t i = worker; i < kKeys; i += kArenas) {
                // Deterministic permutation; leave dispersed old objects live
                // rather than retiring a contiguous prefix of recovery slabs.
                const size_t key = (i * 65537 + 12345) % kKeys;
                if ((key / 4) % 8 == 0) {
                    continue;
                }
                batch.batch_indexs.push_back(batch.batch_keys.size());
                batch.batch_keys.push_back(key);
                batch.batch_locations.emplace_back();
                batch.batch_properties.emplace_back();
                Require(DeserializeFieldMap(payload, batch.batch_locations.back(), batch.batch_properties.back()) ==
                            EC_OK,
                        "deserialize update");
                if (batch.batch_keys.size() == kBatch) {
                    submit();
                }
            }
            submit();
        });
    }
    for (auto &thread : threads) {
        thread.join();
    }
    // Thread exit drains the workers' tcaches in both cases. No explicit purge;
    // partially occupied slabs remain active even after this quiescent point.
}

void Run() {
    control = reinterpret_cast<Control>(dlsym(RTLD_DEFAULT, "mallctl"));
    Require(control != nullptr, "requires jemalloc LD_PRELOAD");
    Dl_info malloc_info{}, control_info{};
    // The production binary is linked with -rdynamic. RTLD_DEFAULT can then
    // resolve malloc to the executable's exported PLT trampoline even though
    // its GOT target is jemalloc. Validate the real next provider instead.
    Require(dladdr(dlsym(RTLD_NEXT, "malloc"), &malloc_info) &&
                dladdr(reinterpret_cast<void *>(control), &control_info) &&
                malloc_info.dli_fbase == control_info.dli_fbase,
            "malloc and mallctl must belong to the same allocator");
    Require(Read<unsigned>("opt.narenas") == kArenas, "requires narenas:4");
    Require(std::string(Read<const char *>("opt.percpu_arena")) == "disabled", "percpu_arena must be disabled");
    Require(Read<bool>("config.stats"), "requires jemalloc stats");
    // Initialize the arenas so even an unused arena has readable statistics.
    for (unsigned arena = 0; arena < kArenas; ++arena) {
        Bind(arena);
    }
    Bind(0);
    std::printf("{\"config\":true,\"keys\":%zu,\"workers\":%u,\"page\":%zu,\"version\":\"%s\"}\n",
                kKeys,
                kArenas,
                Read<size_t>("arenas.page"),
                Read<const char *>("version"));
    MetaStorageBackendManager manager;
    auto config = std::make_shared<MetaStorageBackendConfig>();
    config->SetStorageUri("local://?capacity=1024&num_shard_bits=4");
    auto cache = std::make_unique<MetaLocalBackend>();
    Require(cache->Init("recovery-memory-test", config) == EC_OK && cache->Open() == EC_OK, "initialize cache");
    auto *local = cache.get();
    manager.cache_backend_ = std::move(cache);
    manager.persistent_backend_ = std::make_unique<SyntheticPersistentBackend>();
    std::thread recovery([&] {
        Bind(0);
        manager.AsyncRecoverTask();
    });
    recovery.join();
    Require(manager.GetRecoverState() == MetaStorageBackendManager::RecoverState::kRunning, "recovery incomplete");
    Snapshot("recovered", *local);
    for (int round = 1; round <= 2; ++round) {
        Update(manager, round);
        Snapshot(round == 1 ? "updated_1" : "updated_2", *local);
    }
    // Verify every key still exists and the latest update reached the cache.
    const auto original_json = Payload(0).begin()->second;
    const auto updated_json = Payload(2).begin()->second;
    for (size_t begin = 0; begin < kKeys; begin += kBatch) {
        KeyTypeVec keys;
        for (size_t key = begin; key < std::min(begin + kBatch, kKeys); ++key) {
            keys.push_back(key);
        }
        CacheLocationMapVector locations;
        PropertyMapVector properties;
        Check(local->Get(nullptr, keys, locations, properties), keys.size());
        for (size_t i = 0; i < keys.size(); ++i) {
            Require(locations[i].size() == 1, "location count changed");
            const auto &expected = (keys[i] / 4) % 8 == 0 ? original_json : updated_json;
            Require(locations[i].begin()->second->ToJsonString() == expected, "wrong final value");
        }
    }
}
} // namespace
} // namespace kv_cache_manager

int main() { kv_cache_manager::Run(); }
