#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/meta_cache_policy_config.h"
#include "kv_cache_manager/config/meta_indexer_config.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"
#include "kv_cache_manager/data_storage/snapshot_uri_utils.h"
#include "kv_cache_manager/manager/meta_searcher.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/query_executor.h"

namespace kv_cache_manager {
namespace {

class GetHostCacheStateBenchmark : public TESTBASE {};

// Manual, pure-memory benchmark for the manager-internal query chain. It
// deliberately excludes HTTP JSON parsing so a regression can be attributed
// to local metadata lookup/projection rather than request encoding. Run with:
//
// bazelisk test -c opt //kv_cache_manager/manager/test:GetHostCacheStateBenchmark
//   --test_output=streamed --test_arg=--gtest_also_run_disabled_tests
TEST_F(GetHostCacheStateBenchmark, DISABLED_MillionKeyPureLocalPrefixScenarios) {
    constexpr size_t kMaxKeyCount = 1'000'000;
    constexpr size_t kSetupBatchSize = 4096;
    constexpr size_t kEarlyStopIndex = 1024;
    constexpr int64_t kBaseKey = 900'000'000;
    constexpr std::string_view kHost = "benchmark-host:8080";
    constexpr std::string_view kOtherHost = "other-host:8080";

    auto backend_config = std::make_shared<MetaStorageBackendConfig>("local");
    auto cache_config = std::make_shared<MetaCachePolicyConfig>();
    cache_config->SetCapacity(0);
    auto indexer_config = std::make_shared<MetaIndexerConfig>();
    indexer_config->SetMaxKeyCount(kMaxKeyCount + 16);
    indexer_config->SetMutexShardNum(256);
    indexer_config->SetBatchKeySize(kSetupBatchSize);
    indexer_config->SetMetaStorageBackendConfig(backend_config);
    indexer_config->SetMetaCachePolicyConfig(cache_config);

    auto indexer = std::make_shared<MetaIndexer>();
    indexer->SetQueryExecutor(std::make_shared<QueryExecutor>(
        /*worker_count*/ 4, /*parallel_threshold*/ 256, /*chunk_size*/ 128, /*queue_capacity*/ 64));
    ASSERT_EQ(EC_OK, indexer->Init("get_host_million_key_benchmark", indexer_config));
    auto request_context = std::make_shared<RequestContext>("get_host_million_key_benchmark");

    auto make_location = [](std::string_view host) {
        const std::string host_text(host);
        auto location = std::make_shared<CacheLocation>(
            "kvs#event_report_l2#mem#" + host_text,
            CacheLocationStatus::CLS_SERVING,
            DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2,
            1,
            std::vector<LocationSpec>{LocationSpec("tp0", "event_report://" + host_text + "/mem")});
        // Model the in-memory ReportEvent write path, which has already
        // validated every URI and records the zero aggregate size.
        location->set_validated_total_size(0);
        return location;
    };
    const auto location = make_location(kHost);
    const auto other_location = make_location(kOtherHost);

    KeyVector all_hit_keys(kMaxKeyCount);
    std::iota(all_hit_keys.begin(), all_hit_keys.end(), kBaseKey);
    for (size_t begin = 0; begin < all_hit_keys.size(); begin += kSetupBatchSize) {
        const size_t end = std::min(all_hit_keys.size(), begin + kSetupBatchSize);
        KeyVector batch_keys(all_hit_keys.begin() + static_cast<ptrdiff_t>(begin),
                             all_hit_keys.begin() + static_cast<ptrdiff_t>(end));
        CacheLocationMapVector locations(batch_keys.size());
        for (auto &location_map : locations) {
            location_map.emplace(location->id(), location);
        }
        PropertyMapVector properties;
        ASSERT_EQ(EC_OK, indexer->Put(request_context.get(), batch_keys, locations, properties).ec)
            << "setup begin=" << begin;
    }

    // One existing key with another host supports a host-prefix early-stop
    // scenario without changing the all-hit data set.
    const KeyType other_host_key = kBaseKey + static_cast<KeyType>(kMaxKeyCount);
    KeyVector extra_keys{other_host_key};
    CacheLocationMapVector extra_locations(1);
    extra_locations[0].emplace(other_location->id(), other_location);
    PropertyMapVector extra_properties;
    ASSERT_EQ(EC_OK, indexer->Put(request_context.get(), extra_keys, extra_locations, extra_properties).ec);

    MetaSearcher searcher(indexer, [](const CacheLocation &) { return true; }, {});
    size_t active_worker_count = 4;
    const MetaSearcher::CheckHostCacheLocationFunc visibility_check = [](const CacheLocation &candidate,
                                                                         MetaSearcher::HostCacheLocationInfo &out) {
        out = {};
        std::string_view storage_type;
        std::string_view reporter_medium;
        std::string_view reporter_host;
        if (!SnapshotUriUtils::ParseEventReportLocationIdView(
                candidate.id(), storage_type, reporter_medium, reporter_host)) {
            return false;
        }
        const bool uri_structure_prevalidated = candidate.HasValidatedLocationSpecs();
        for (const auto &spec : candidate.location_specs()) {
            std::string_view version;
            if (!SnapshotUriUtils::InspectSnapshotUriForVisibility(spec.uri(), version, uri_structure_prevalidated)) {
                return false;
            }
        }
        out.has_reporter_identity = true;
        out.reporter_medium = reporter_medium;
        out.reporter_host = reporter_host;
        return true;
    };

    auto run_case = [&](const char *name, const KeyVector &query_keys, int64_t expected_prefix, int iterations) {
        std::vector<double> elapsed_ms;
        elapsed_ms.reserve(iterations);
        for (int iteration = -1; iteration < iterations; ++iteration) {
            std::vector<MetaSearcher::HostCacheMatch> matches;
            const auto begin = std::chrono::steady_clock::now();
            ASSERT_EQ(EC_OK,
                      searcher.PrefixMatchByHost(
                          request_context.get(), query_keys, false, {"mem"}, matches, &visibility_check));
            const auto elapsed =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
            const auto match = std::find_if(
                matches.begin(), matches.end(), [](const auto &item) { return item.host_ip_port == kHost; });
            ASSERT_NE(matches.end(), match) << name;
            ASSERT_EQ(expected_prefix, match->local) << name;
            if (iteration >= 0) {
                elapsed_ms.push_back(elapsed);
            }
        }
        std::sort(elapsed_ms.begin(), elapsed_ms.end());
        const double average =
            std::accumulate(elapsed_ms.begin(), elapsed_ms.end(), 0.0) / static_cast<double>(elapsed_ms.size());
        std::cout << "[GET_HOST_BENCH] case=" << name << " keys=" << query_keys.size()
                  << " workers=" << active_worker_count << " p50_ms=" << elapsed_ms[elapsed_ms.size() / 2]
                  << " avg_ms=" << average << std::endl;
    };

    auto run_metadata_only = [&](const KeyVector &query_keys, int iterations) {
        std::vector<double> elapsed_ms;
        elapsed_ms.reserve(iterations);
        for (int iteration = -1; iteration < iterations; ++iteration) {
            const auto begin = std::chrono::steady_clock::now();
            const auto result = indexer->VisitLocationValuesForPrefix(
                request_context.get(), query_keys, [&query_keys](size_t, const CompactLocationsPerKey &, size_t) {
                    return query_keys.size();
                });
            const auto elapsed =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
            ASSERT_EQ(EC_OK, result.terminal_ec);
            ASSERT_EQ(query_keys.size(), result.valid_key_count);
            if (iteration >= 0) {
                elapsed_ms.push_back(elapsed);
            }
        }
        std::sort(elapsed_ms.begin(), elapsed_ms.end());
        const double average =
            std::accumulate(elapsed_ms.begin(), elapsed_ms.end(), 0.0) / static_cast<double>(elapsed_ms.size());
        std::cout << "[GET_HOST_BENCH] case=metadata_only keys=" << query_keys.size()
                  << " workers=" << active_worker_count << " p50_ms=" << elapsed_ms[elapsed_ms.size() / 2]
                  << " avg_ms=" << average << std::endl;
    };

    for (const size_t key_count : {size_t{100'000}, size_t{500'000}, kMaxKeyCount}) {
        KeyVector query_keys(all_hit_keys.begin(), all_hit_keys.begin() + static_cast<ptrdiff_t>(key_count));
        run_metadata_only(query_keys, 3);
        run_case("all_hit", query_keys, static_cast<int64_t>(key_count), 3);
    }

    KeyVector early_host_stop = all_hit_keys;
    early_host_stop[kEarlyStopIndex] = other_host_key;
    run_case("early_host_stop", early_host_stop, kEarlyStopIndex, 5);

    KeyVector early_metadata_miss = all_hit_keys;
    early_metadata_miss[kEarlyStopIndex] = kBaseKey + static_cast<KeyType>(kMaxKeyCount + 10);
    run_case("early_metadata_miss", early_metadata_miss, kEarlyStopIndex, 5);

    for (const size_t worker_count : {size_t{1}, size_t{2}, size_t{8}, size_t{16}}) {
        active_worker_count = worker_count;
        indexer->SetQueryExecutor(
            std::make_shared<QueryExecutor>(worker_count, /*parallel_threshold*/ 256, /*chunk_size*/ 128, 64));
        run_metadata_only(all_hit_keys, 3);
        run_case("all_hit_worker_scaling", all_hit_keys, static_cast<int64_t>(kMaxKeyCount), 3);
    }

    // Skip million-key persistence in a manual latency benchmark; all tested
    // data lives solely in the local in-memory backend.
    ASSERT_EQ(EC_OK, indexer->backend_manager_->Close());
    indexer->backend_manager_.reset();
}

} // namespace
} // namespace kv_cache_manager
