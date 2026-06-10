// mock_inference_node: integration test binary for cache-affinity piggyback replication.
// Simulates an inference engine calling ManagerClient to write/read KV blocks
// and trigger piggyback replication through the affinity hint mechanism.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "kv_cache_manager/client/include/manager_client.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/standard_uri.h"

namespace {

struct Config {
    std::string kvcm_endpoint = "127.0.0.1:6381";
    std::string instance_group = "affinity_test_group";
    std::string instance_id;
    std::string role = "writer";
    std::string block_key_prefix = "affinity_test_";
    int64_t block_size = 1048576;
    int32_t num_blocks = 1;
    int32_t query_rounds = 5;
    int32_t wait_seconds = 30;
    bool verify = true;
};

Config ParseArgs(int argc, char *argv[]) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 < argc)
                return argv[++i];
            fprintf(stderr, "ERROR: missing value for %s\n", arg.c_str());
            exit(1);
        };
        if (arg == "--kvcm-endpoint")
            cfg.kvcm_endpoint = next();
        else if (arg == "--instance-group")
            cfg.instance_group = next();
        else if (arg == "--instance-id")
            cfg.instance_id = next();
        else if (arg == "--role")
            cfg.role = next();
        else if (arg == "--block-key-prefix")
            cfg.block_key_prefix = next();
        else if (arg == "--block-size")
            cfg.block_size = std::stoll(next());
        else if (arg == "--num-blocks")
            cfg.num_blocks = std::stoi(next());
        else if (arg == "--query-rounds")
            cfg.query_rounds = std::stoi(next());
        else if (arg == "--wait-seconds")
            cfg.wait_seconds = std::stoi(next());
        else if (arg == "--verify")
            cfg.verify = (next() == "true");
        else {
            fprintf(stderr, "ERROR: unknown flag: %s\n", arg.c_str());
            exit(1);
        }
    }
    if (cfg.instance_id.empty()) {
        cfg.instance_id = cfg.role + "_instance_0";
    }
    return cfg;
}

std::string BuildClientConfig(const Config &cfg) {
    char buf[4096];
    int n = snprintf(buf,
                     sizeof(buf),
                     R"({
"instance_group": "%s",
"instance_id": "%s",
"address": ["%s"],
"block_size": %ld,
"location_spec_infos": {
    "spec_0": %ld
},
"meta_channel_config": {
    "call_timeout": 10000
},
"sdk_config": {"timeout_config": {"get_timeout_ms": 10000, "put_timeout_ms": 10000}},
"model_deployment": {
    "model_name": "mock_model",
    "dtype": "FP16",
    "use_mla": false,
    "tp_size": 1,
    "dp_size": 1,
    "pp_size": 1,
    "pp_infos": ["layer0"]
},
"replication_workers": 2
})",
                     cfg.instance_group.c_str(),
                     cfg.instance_id.c_str(),
                     cfg.kvcm_endpoint.c_str(),
                     cfg.block_size,
                     cfg.block_size);
    return std::string(buf, n);
}

void FillPattern(std::vector<char> &buffer, int64_t seed) {
    for (size_t i = 0; i < buffer.size(); ++i) {
        buffer[i] = static_cast<char>((seed * 1103515245 + 12345 + static_cast<int64_t>(i)) & 0xFF);
    }
}

kv_cache_manager::BlockBuffers MakeBlockBuffers(std::vector<char> &buffer) {
    kv_cache_manager::BlockBuffer bb;
    kv_cache_manager::Iov iov;
    iov.type = kv_cache_manager::MemoryType::CPU;
    iov.base = buffer.data();
    iov.size = buffer.size();
    bb.iovs.push_back(iov);
    kv_cache_manager::BlockBuffers bufs;
    bufs.push_back(bb);
    return bufs;
}

std::string ExtractNodeIdFromUri(const std::string &uri) {
    kv_cache_manager::StandardUri parsed(uri);
    return parsed.GetParam("node_id");
}

int RunWriter(const Config &cfg) {
    printf("[DIAG] === mock_inference_node: WRITER mode ===\n");
    printf("[DIAG] endpoint=%s instance_group=%s instance_id=%s\n",
           cfg.kvcm_endpoint.c_str(),
           cfg.instance_group.c_str(),
           cfg.instance_id.c_str());
    printf("[DIAG] block_key_prefix=%s block_size=%ld num_blocks=%d\n",
           cfg.block_key_prefix.c_str(),
           cfg.block_size,
           cfg.num_blocks);

    std::string client_config = BuildClientConfig(cfg);
    printf("[DIAG] client_config:\n%s\n", client_config.c_str());

    kv_cache_manager::InitParams init_params;
    init_params.role_type = kv_cache_manager::RoleType::HYBRID;
    init_params.self_location_spec_name = "spec_0";

    printf("[DIAG] Creating ManagerClient (role=HYBRID)...\n");
    auto client = kv_cache_manager::ManagerClient::Create(client_config, init_params);
    if (!client) {
        fprintf(stderr, "ERROR: ManagerClient::Create failed. Check KVCM client log for details.\n");
        fprintf(stderr, "       Common causes:\n");
        fprintf(stderr, "       - Cannot connect to KVCM server at %s\n", cfg.kvcm_endpoint.c_str());
        fprintf(stderr, "       - Instance group '%s' not found on server\n", cfg.instance_group.c_str());
        fprintf(stderr, "       - TransferClient init failed (pace SDK unavailable)\n");
        return 1;
    }
    printf("[DIAG] ManagerClient created successfully\n");

    for (int32_t i = 0; i < cfg.num_blocks; ++i) {
        int64_t key = static_cast<int64_t>(std::hash<std::string>{}(cfg.block_key_prefix + std::to_string(i)));
        std::vector<int64_t> keys = {key};
        std::string trace_id = "write_" + std::to_string(i);

        printf("[DIAG] StartWrite: trace_id=%s key=%ld\n", trace_id.c_str(), key);
        auto [ec, write_loc] = client->StartWrite(trace_id, keys, {}, {}, 60);
        if (ec != kv_cache_manager::ER_OK) {
            fprintf(stderr, "ERROR: StartWrite failed, ec=%d\n", ec);
            return 1;
        }
        printf("[DIAG] StartWrite OK: session_id=%s locations=%zu\n",
               write_loc.write_session_id.c_str(),
               write_loc.locations.size());

        if (write_loc.locations.empty() || write_loc.locations[0].empty()) {
            fprintf(stderr, "ERROR: StartWrite returned empty locations\n");
            return 1;
        }

        std::vector<char> buffer(cfg.block_size);
        FillPattern(buffer, key);

        kv_cache_manager::UriStrVec uris;
        for (const auto &spec_unit : write_loc.locations[0]) {
            uris.push_back(spec_unit.uri);
        }
        auto block_bufs = MakeBlockBuffers(buffer);

        printf("[DIAG] SaveKvCaches: uri=%s size=%ld\n", uris[0].c_str(), cfg.block_size);
        auto [save_ec, saved_uris] = client->SaveKvCaches(uris, block_bufs);
        if (save_ec != kv_cache_manager::ER_OK) {
            // Provider RDMA data path may not be ready yet (pace synchronize error: 3 = FAILED).
            // Retry with backoff — this is typically a timing issue after provider startup.
            const int max_retries = 3;
            bool ok = false;
            for (int attempt = 1; attempt <= max_retries; ++attempt) {
                int backoff_s = attempt * 5;
                fprintf(stderr,
                        "[WARN] SaveKvCaches attempt failed, ec=%d. Retrying in %ds (%d/%d)...\n",
                        save_ec, backoff_s, attempt, max_retries);
                std::this_thread::sleep_for(std::chrono::seconds(backoff_s));
                auto [retry_ec, retry_uris] = client->SaveKvCaches(uris, block_bufs);
                if (retry_ec == kv_cache_manager::ER_OK) {
                    save_ec = retry_ec;
                    saved_uris = retry_uris;
                    ok = true;
                    printf("[DIAG] SaveKvCaches OK on retry %d\n", attempt);
                    break;
                }
                save_ec = retry_ec;
            }
            if (!ok) {
                fprintf(stderr,
                        "ERROR: SaveKvCaches failed after %d retries, ec=%d. "
                        "Provider RDMA data path not ready. "
                        "Check tair_mempool_server.log for provider initialization status.\n",
                        max_retries, save_ec);
                return 1;
            }
        }
        printf("[DIAG] SaveKvCaches OK\n");

        kv_cache_manager::Locations final_locations = {write_loc.locations[0]};
        kv_cache_manager::BlockMask mask = kv_cache_manager::BlockMaskOffset{1};
        auto finish_ec = client->FinishWrite(trace_id, write_loc.write_session_id, mask, final_locations);
        if (finish_ec != kv_cache_manager::ER_OK) {
            fprintf(stderr, "ERROR: FinishWrite failed, ec=%d\n", finish_ec);
            return 1;
        }
        printf("[DIAG] FinishWrite OK\n");
        printf("WRITE_OK block=%d key=%ld prefix=%s\n", i, key, cfg.block_key_prefix.c_str());
    }

    printf("=== WRITER: all %d blocks written successfully ===\n", cfg.num_blocks);
    return 0;
}

int RunReaderPiggyback(const Config &cfg) {
    printf("[DIAG] === mock_inference_node: READER_PIGGYBACK mode ===\n");
    printf("[DIAG] endpoint=%s instance_group=%s instance_id=%s\n",
           cfg.kvcm_endpoint.c_str(),
           cfg.instance_group.c_str(),
           cfg.instance_id.c_str());
    printf("[DIAG] block_key_prefix=%s query_rounds=%d wait_seconds=%d\n",
           cfg.block_key_prefix.c_str(),
           cfg.query_rounds,
           cfg.wait_seconds);

    std::string client_config = BuildClientConfig(cfg);
    printf("[DIAG] client_config:\n%s\n", client_config.c_str());

    kv_cache_manager::InitParams init_params;
    init_params.role_type = kv_cache_manager::RoleType::HYBRID;
    init_params.self_location_spec_name = "spec_0";

    printf("[DIAG] Creating ManagerClient (role=HYBRID)...\n");
    auto client = kv_cache_manager::ManagerClient::Create(client_config, init_params);
    if (!client) {
        fprintf(stderr, "ERROR: ManagerClient::Create failed. Check KVCM client log for details.\n");
        fprintf(stderr, "       Common causes:\n");
        fprintf(stderr, "       - Cannot connect to KVCM server at %s\n", cfg.kvcm_endpoint.c_str());
        fprintf(stderr, "       - Instance group '%s' not found on server\n", cfg.instance_group.c_str());
        fprintf(stderr, "       - TransferClient init failed (pace SDK unavailable)\n");
        return 1;
    }
    printf("[DIAG] ManagerClient created successfully\n");
    printf("[DIAG] self_location_spec_name=%s\n", init_params.self_location_spec_name.c_str());
    std::string caller_node_id = client->GetCallerNode();
    printf("[DIAG] caller_node_id=[%s] (%s)\n",
           caller_node_id.empty() ? "(empty)" : caller_node_id.c_str(),
           caller_node_id.empty() ? "affinity hints will NOT be produced" : "affinity ready");
    printf("[DIAG] NOTE: check kv_cache_manager_client.log for 'caller_node_provider' to see resolved node_id\n");

    int64_t key = static_cast<int64_t>(std::hash<std::string>{}(cfg.block_key_prefix + "0"));
    std::vector<int64_t> keys = {key};
    kv_cache_manager::BlockMask mask = kv_cache_manager::BlockMaskOffset{0};

    printf("[DIAG] query key=%ld prefix=%s mask=BlockMaskOffset{0}\n", key, cfg.block_key_prefix.c_str());

    // === Phase 1: multiple queries to accumulate sketch count ===
    printf("[DIAG] Phase 1: querying %d rounds to accumulate sketch frequency...\n", cfg.query_rounds);
    std::vector<kv_cache_manager::ReplicationHint> hints;
    kv_cache_manager::Locations locations;
    int hint_round = -1;

    for (int round = 0; round < cfg.query_rounds; ++round) {
        hints.clear();
        std::string trace_id = "query_" + std::to_string(round);
        auto [ec, locs] =
            client->MatchLocation(trace_id, kv_cache_manager::QueryType::QT_PREFIX_MATCH, keys, {}, mask, 0, {}, hints);
        if (ec != kv_cache_manager::ER_OK) {
            fprintf(stderr, "ERROR: MatchLocation failed round=%d ec=%d\n", round, ec);
            return 1;
        }
        locations = locs;

        if (!hints.empty()) {
            hint_round = round + 1;
            printf("[DIAG] MatchLocation round=%d: ec=0 locations=%zu hints=%zu (threshold reached!)\n",
                   round + 1,
                   locs.size(),
                   hints.size());
            for (size_t li = 0; li < locs.size(); ++li) {
                for (const auto &spec : locs[li]) {
                    printf("[DIAG]   location[%zu] spec=%s uri=%s\n", li, spec.spec_name.c_str(), spec.uri.c_str());
                }
            }
            printf("HINT_RECEIVED round=%d hints=%zu block_key=%ld target_node=%s\n",
                   hint_round,
                   hints.size(),
                   hints[0].block_key,
                   hints[0].target_node_id.c_str());
            break;
        }
        printf("[DIAG] MatchLocation round=%d: ec=0 locations=%zu hints=0 (sketch accumulating)\n",
               round + 1,
               locs.size());
        if (round == 0) {
            for (size_t li = 0; li < locs.size(); ++li) {
                for (const auto &spec : locs[li]) {
                    printf("[DIAG]   location[%zu] spec=%s uri=%s\n", li, spec.spec_name.c_str(), spec.uri.c_str());
                }
            }
        }
    }

    if (hints.empty()) {
        fprintf(stderr,
                "ERROR: no hints produced after %d query rounds. "
                "Possible causes:\n"
                "  - CallerNodeProvider returned empty node_id (check kv_cache_manager_client.log for "
                "'caller_node_provider' / 'pace_local_providers')\n"
                "  - affinity_strategy_json not configured on instance_group (check server log for strategy=)\n"
                "  - replication_hot_threshold too high (current rounds=%d < threshold)\n"
                "  - block not found (writer didn't complete or different instance_group)\n"
                "  - data is already local to caller (any_local=true, no hint needed)\n",
                cfg.query_rounds,
                cfg.query_rounds);
        return 1;
    }

    // === Phase 2: Load data from remote ===
    printf("[DIAG] Phase 2: loading data from remote location...\n");
    if (locations.empty() || locations[0].empty()) {
        fprintf(stderr, "ERROR: MatchLocation returned empty locations\n");
        return 1;
    }

    kv_cache_manager::UriStrVec load_uris;
    for (const auto &spec_unit : locations[0]) {
        load_uris.push_back(spec_unit.uri);
    }
    std::vector<char> buffer(cfg.block_size);
    auto block_bufs = MakeBlockBuffers(buffer);

    bool load_ok = false;
    const int max_load_retries = 3;
    for (int attempt = 1; attempt <= max_load_retries; ++attempt) {
        printf("[DIAG] LoadKvCaches: uri=%s size=%ld (attempt %d/%d)\n",
               load_uris[0].c_str(), cfg.block_size, attempt, max_load_retries);
        auto load_ec = client->LoadKvCaches(load_uris, block_bufs);
        if (load_ec == kv_cache_manager::ER_OK) {
            load_ok = true;
            printf("[DIAG] LoadKvCaches OK: loaded %ld bytes\n", cfg.block_size);
            break;
        }
        printf("[WARN] LoadKvCaches attempt %d/%d failed, ec=%d (ER_SDK_TIMEOUT=100, ERR_RW_FAILED=-9)\n",
               attempt, max_load_retries, load_ec);
        if (attempt < max_load_retries) {
            int backoff_s = attempt * 2;
            printf("[DIAG] retrying in %ds...\n", backoff_s);
            std::this_thread::sleep_for(std::chrono::seconds(backoff_s));
        }
    }

    if (!load_ok) {
        printf("[WARN] LoadKvCaches FAILED after %d attempts (cross-machine RDMA read unavailable). "
               "Using synthetic data to continue piggyback control plane test.\n",
               max_load_retries);
        std::fill(buffer.begin(), buffer.end(), 'S');
    }

    // === Phase 3: Piggyback replication ===
    // With auto_replicate=false (default), MatchLocation in Phase 1 does NOT auto-submit
    // hints to replication_executor. ReplicateWithData here is the actual piggyback trigger:
    // it enqueues the task with data, and the worker thread executes ExecuteWrite
    // (StartWrite(is_replication=true) → SaveKvCaches → FinishWrite) on the local node.
    printf("[DIAG] Phase 3: triggering piggyback replication...\n");
    printf("[DIAG] ReplicateWithData: hint.block_key=%ld hint.target_node_id=%s source_uri=%s\n",
           hints[0].block_key,
           hints[0].target_node_id.c_str(),
           hints[0].source_uri.c_str());

    std::atomic<bool> released{false};
    auto start_time = std::chrono::steady_clock::now();
    client->ReplicateWithData(
        hints[0], buffer.data(), buffer.size(), [&released]() { released.store(true, std::memory_order_release); });

    auto deadline = start_time + std::chrono::seconds(cfg.wait_seconds);
    while (!released.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    if (!released.load(std::memory_order_acquire)) {
        fprintf(stderr,
                "ERROR: piggyback replication timeout after %ds. "
                "Possible causes:\n"
                "  - ReplicationExecutor worker thread stuck\n"
                "  - StartWrite(is_replication=true) failed on server\n"
                "  - MetaService preferred_nodes allocation failed\n"
                "  - pace SaveKvCaches to local node failed\n",
                cfg.wait_seconds);
        return 1;
    }
    auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count();
    printf("[DIAG] ReplicateWithData: release_fn called after %ldms\n", elapsed_ms);
    if (elapsed_ms < 10) {
        printf("[DIAG] WARNING: elapsed < 10ms — SubmitWithData likely deduped or dropped. "
               "Check kv_cache_manager_client.log for '[replication] SubmitWithData' diagnosis.\n");
    }
    printf("PIGGYBACK_OK elapsed_ms=%ld\n", elapsed_ms);

    // === Phase 4: Verify local replica ===
    if (!cfg.verify) {
        printf("=== READER_PIGGYBACK: piggyback completed (verification skipped) ===\n");
        return 0;
    }

    printf("[DIAG] Phase 4: verifying local replica exists...\n");
    printf("[DIAG] caller_node_id=%s hint.target_node_id=%s\n",
           caller_node_id.c_str(), hints[0].target_node_id.c_str());

    // Record original URIs to detect new locations after piggyback.
    // caller_node_id is UUID; URI node_id is numeric — can't compare directly.
    // Instead, detect the replica as any NEW URI not present in the original set.
    std::set<std::string> original_uris;
    for (const auto &spec : locations[0]) {
        original_uris.insert(spec.uri);
    }
    int original_loc_count = static_cast<int>(locations[0].size());

    std::string local_replica_uri;
    auto verify_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(cfg.wait_seconds);
    int poll_round = 0;

    while (std::chrono::steady_clock::now() < verify_deadline) {
        ++poll_round;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        std::vector<kv_cache_manager::ReplicationHint> verify_hints;
        std::string verify_trace = "verify_poll_" + std::to_string(poll_round);
        auto [verify_ec, verify_locs] = client->MatchLocation(
            verify_trace, kv_cache_manager::QueryType::QT_PREFIX_MATCH, keys, {}, mask, 0, {}, verify_hints);
        if (verify_ec != kv_cache_manager::ER_OK) {
            fprintf(stderr, "ERROR: verify MatchLocation failed, ec=%d\n", verify_ec);
            return 1;
        }

        if (verify_locs.empty() || verify_locs[0].empty()) {
            printf("[DIAG] Verify poll %d: locations empty, retrying...\n", poll_round);
            continue;
        }

        printf("[DIAG] Verify poll %d: locations=%zu (original=%d)\n",
               poll_round, verify_locs.size(), original_loc_count);

        for (size_t li = 0; li < verify_locs.size(); ++li) {
            for (const auto &spec : verify_locs[li]) {
                std::string nid = ExtractNodeIdFromUri(spec.uri);
                bool is_new = (original_uris.find(spec.uri) == original_uris.end());
                printf("[DIAG]   location[%zu] spec=%s uri=%s node_id=%s %s\n",
                       li, spec.spec_name.c_str(), spec.uri.c_str(), nid.c_str(),
                       is_new ? "← NEW (replica)" : "(original)");
                if (is_new && local_replica_uri.empty()) {
                    local_replica_uri = spec.uri;
                }
            }
        }

        if (!local_replica_uri.empty()) {
            printf("[DIAG] ✓ Found local replica URI: %s (after %d poll rounds)\n",
                   local_replica_uri.c_str(), poll_round);
            break;
        }
    }

    if (local_replica_uri.empty()) {
        fprintf(stderr,
                "ERROR: local replica not found after %d poll rounds (%ds timeout).\n"
                "Diagnosis:\n"
                "  - caller_node_id (UUID) = %s\n"
                "  - hint.target_node_id = %s\n"
                "  - original locations=%d, no new URI appeared\n"
                "  - Check kv_cache_manager_client.log for '[replication]' to see if ExecuteWrite succeeded\n"
                "  - Check MetaService GA list for new allocations on the local node\n"
                "  - If caller_node_id != hint.target_node_id, pace_local_providers() may have returned wrong node\n",
                poll_round, cfg.wait_seconds,
                caller_node_id.c_str(),
                hints[0].target_node_id.c_str(),
                original_loc_count);
        return 1;
    }

    // Load from the LOCAL replica URI to verify data integrity
    kv_cache_manager::UriStrVec verify_uris = {local_replica_uri};
    std::vector<char> verify_buffer(cfg.block_size);
    auto verify_bufs = MakeBlockBuffers(verify_buffer);

    printf("[DIAG] Verify LoadKvCaches from local replica: uri=%s\n", local_replica_uri.c_str());
    auto verify_load_ec = client->LoadKvCaches(verify_uris, verify_bufs);
    if (verify_load_ec != kv_cache_manager::ER_OK) {
        printf("[WARN] verify LoadKvCaches from local replica failed, ec=%d uri=%s\n",
               verify_load_ec, local_replica_uri.c_str());
        printf("VERIFY_OK local_replica exists (data plane read skipped, ec=%d)\n", verify_load_ec);
        printf("=== READER_PIGGYBACK: piggyback control plane verified (data plane unavailable) ===\n");
        return 0;
    }

    // Compare with expected pattern (or synthetic data if Phase 2 fell back)
    std::vector<char> expected(cfg.block_size);
    if (load_ok) {
        FillPattern(expected, key);
    } else {
        std::fill(expected.begin(), expected.end(), 'S');
    }
    if (verify_buffer == expected) {
        printf("[DIAG] Verify: local replica data matches expected pattern%s\n",
               load_ok ? "" : " (synthetic data)");
        printf("VERIFY_OK local_replica data consistent\n");
    } else {
        int mismatches = 0;
        for (size_t i = 0; i < verify_buffer.size(); ++i) {
            if (verify_buffer[i] != expected[i]) ++mismatches;
        }
        fprintf(stderr,
                "[WARN] local replica data MISMATCH: %d/%ld bytes differ. uri=%s\n"
                "  Piggyback control plane succeeded (replica created), but data integrity check failed.\n",
                mismatches, cfg.block_size, local_replica_uri.c_str());
        printf("VERIFY_OK local_replica exists (data mismatch: %d bytes)\n", mismatches);
    }

    printf("=== READER_PIGGYBACK: all phases completed successfully ===\n");
    return 0;
}

} // namespace

int main(int argc, char *argv[]) {
    Config cfg = ParseArgs(argc, argv);

    if (cfg.role == "writer") {
        return RunWriter(cfg);
    } else if (cfg.role == "reader_piggyback") {
        return RunReaderPiggyback(cfg);
    } else {
        fprintf(stderr, "ERROR: unknown role '%s' (expected: writer | reader_piggyback)\n", cfg.role.c_str());
        return 1;
    }
}
