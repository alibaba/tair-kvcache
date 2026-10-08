// Strict single-spec data-plane acceptance driver. Run writer on A and readers on B.
// A callback releasing a buffer is NOT proof that replication succeeded: query the
// expected physical node and compare every byte before emitting E2E_OK.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "kv_cache_manager/client/include/manager_client.h"
#include "kv_cache_manager/common/standard_uri.h"

namespace {
using namespace kv_cache_manager;

struct Config {
    std::string kvcm_endpoint = "127.0.0.1:6381";
    std::string instance_group = "affinity_test_group";
    std::string instance_id;
    std::string role = "writer";
    std::string block_key_prefix = "affinity_test_";
    std::string expected_node_id; // PACE numeric ID from MetaService, not Provider UUID.
    int64_t block_size = 1048576;
    int32_t num_blocks = 3;
    int32_t query_rounds = 5;
    int32_t expected_hint_round = 0;
    int32_t wait_seconds = 30;
    int32_t repeat_reads = 3;
};

void Require(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

int64_t Positive(const std::string &value) {
    size_t consumed = 0;
    auto n = std::stoll(value, &consumed);
    Require(consumed == value.size() && n > 0 && n <= std::numeric_limits<int32_t>::max(),
            "invalid positive integer: " + value);
    return n;
}

Config ParseArgs(int argc, char **argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string flag = argv[i];
        Require(i + 1 < argc, "missing value for " + flag);
        std::string value = argv[++i];
        if (flag == "--kvcm-endpoint") cfg.kvcm_endpoint = value;
        else if (flag == "--instance-group") cfg.instance_group = value;
        else if (flag == "--instance-id") cfg.instance_id = value;
        else if (flag == "--role") cfg.role = value;
        else if (flag == "--block-key-prefix") cfg.block_key_prefix = value;
        else if (flag == "--expected-node-id") cfg.expected_node_id = value;
        else if (flag == "--block-size") cfg.block_size = Positive(value);
        else if (flag == "--num-blocks") cfg.num_blocks = Positive(value);
        else if (flag == "--query-rounds") cfg.query_rounds = Positive(value);
        else if (flag == "--expected-hint-round") cfg.expected_hint_round = Positive(value);
        else if (flag == "--wait-seconds") cfg.wait_seconds = Positive(value);
        else if (flag == "--repeat-reads") cfg.repeat_reads = Positive(value);
        else if (flag == "--verify") Require(value == "true", "verification cannot be disabled");
        else throw std::runtime_error("unknown flag: " + flag);
    }
    Require(!cfg.instance_id.empty(), "--instance-id is required and must match writer/reader");
    Require(cfg.expected_hint_round <= cfg.query_rounds, "hint threshold exceeds query rounds");
    Require(cfg.role == "writer" || cfg.role == "writer_abort" || cfg.role == "reader_piggyback" ||
                cfg.role == "reader_async" || cfg.role == "reader_local" || cfg.role == "reader_miss" ||
                cfg.role == "remove", "unknown role: " + cfg.role);
    if (cfg.role != "reader_miss" && cfg.role != "remove") {
        Require(!cfg.expected_node_id.empty(), "--expected-node-id is required to prove physical locality");
        size_t consumed = 0;
        auto node = std::stoul(cfg.expected_node_id, &consumed);
        Require(consumed == cfg.expected_node_id.size() && node <= 65535, "invalid PACE node ID");
        cfg.expected_node_id = std::to_string(node);
    }
    return cfg;
}

std::string Quote(const std::string &s) {
    std::string result = "\"";
    for (unsigned char c : s) {
        Require(c >= 32, "control character in configuration");
        if (c == '\\' || c == '"') result += '\\';
        result += static_cast<char>(c);
    }
    return result + "\"";
}

std::string BuildClientConfig(const Config &cfg) {
    return "{\"instance_group\":" + Quote(cfg.instance_group) +
           ",\"instance_id\":" + Quote(cfg.instance_id) +
           ",\"address\":[" + Quote(cfg.kvcm_endpoint) +
           "],\"block_size\":" + std::to_string(cfg.block_size) +
           ",\"location_spec_infos\":{\"spec_0\":" + std::to_string(cfg.block_size) +
           "},\"meta_channel_config\":{\"call_timeout\":10000}," +
           "\"sdk_config\":{\"timeout_config\":{\"get_timeout_ms\":10000,\"put_timeout_ms\":10000}}," +
           "\"model_deployment\":{\"model_name\":\"mock_model\",\"dtype\":\"FP16\"," +
           "\"use_mla\":false,\"tp_size\":1,\"dp_size\":1,\"pp_size\":1}," +
           "\"replication_workers\":2,\"auto_replicate\":" +
           (cfg.role == "reader_async" ? "true}" : "false}");
}

int64_t BlockKey(const Config &cfg, int index) {
    // Stable across compilers/processes; unsigned arithmetic avoids signed overflow.
    uint64_t h = 14695981039346656037ULL;
    for (unsigned char c : cfg.block_key_prefix + std::to_string(index)) {
        h = (h ^ c) * 1099511628211ULL;
    }
    return static_cast<int64_t>(h & 0x7fffffffffffffffULL);
}

void FillPattern(std::vector<char> &buffer, int64_t key) {
    uint64_t state = static_cast<uint64_t>(key);
    for (auto &byte : buffer) {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        byte = static_cast<char>((state * 2685821657736338717ULL) >> 56);
    }
}

BlockBuffers Buffers(std::vector<char> &buffer) {
    BlockBuffer block;
    block.iovs.push_back(Iov{MemoryType::CPU, buffer.data(), buffer.size(), false});
    return {block};
}

std::string NodeId(const std::string &uri) {
    return StandardUri(uri).GetParam("node_id");
}

const std::string &SingleUri(const Locations &locations) {
    Require(locations.size() == 1 && locations[0].size() == 1 && locations[0][0].spec_name == "spec_0",
            "expected exactly one block and spec_0; multi-spec requires separate acceptance tests");
    Require(!locations[0][0].uri.empty(), "empty URI");
    return locations[0][0].uri;
}

Locations Query(ManagerClient &client, int64_t key, std::vector<ClientReplicationHint> &hints) {
    hints.clear();
    auto result = client.MatchLocation("affinity_query", QueryType::QT_PREFIX_MATCH, {key}, {},
                                       BlockMaskOffset{0}, 0, {}, hints);
    Require(result.first == ER_OK, "MatchLocation failed: " + std::to_string(result.first));
    return result.second;
}

bool IsMiss(const Locations &locations) {
    for (const auto &location : locations) {
        if (!location.empty()) return false;
    }
    return true;
}

std::vector<char> ReadAndCheck(ManagerClient &client, const Config &cfg, int64_t key, const std::string &uri) {
    std::vector<char> buffer(cfg.block_size, 0);
    auto ec = client.LoadKvCaches({uri}, Buffers(buffer));
    Require(ec == ER_OK, "LoadKvCaches failed: " + std::to_string(ec) + " uri=" + uri);
    std::vector<char> expected(cfg.block_size);
    FillPattern(expected, key);
    Require(buffer == expected, "data mismatch for key=" + std::to_string(key) + " uri=" + uri);
    return buffer;
}

void CheckMiss(ManagerClient &client, int64_t key) {
    std::vector<ClientReplicationHint> hints;
    Require(IsMiss(Query(client, key, hints)) && hints.empty(), "unpublished/removed/isolated key is visible");
}

void CheckLocal(ManagerClient &client, const Config &cfg, int64_t key) {
    std::string first_uri;
    for (int i = 0; i < cfg.repeat_reads; ++i) {
        std::vector<ClientReplicationHint> hints;
        auto locations = Query(client, key, hints);
        const auto &uri = SingleUri(locations);
        Require(NodeId(uri) == cfg.expected_node_id, "read was not routed to expected local Provider: " + uri);
        Require(hints.empty(), "local hit unexpectedly emitted replication hint");
        Require(first_uri.empty() || first_uri == uri, "local URI changed during repeated reads");
        first_uri = uri;
        ReadAndCheck(client, cfg, key, uri);
    }
}

void Write(ManagerClient &client, const Config &cfg, int64_t key, bool abort) {
    CheckMiss(client, key); // Fresh prefix also prevents accidental reuse of an earlier test run.
    auto result = client.StartWrite("affinity_write", {key}, {}, {}, 60);
    Require(result.first == ER_OK, "StartWrite failed: " + std::to_string(result.first));
    auto &write = result.second;
    const auto &uri = SingleUri(write.locations);
    Require(!write.write_session_id.empty(), "empty write session");
    Require(NodeId(uri) == cfg.expected_node_id, "write allocated on wrong physical Provider: " + uri);
    CheckMiss(client, key); // CLS_WRITING must not count as a readable hit.
    if (abort) {
        Require(client.FinishWrite("affinity_abort", write.write_session_id, BlockMaskVector{false}, {}) == ER_OK,
                "failed-write FinishWrite failed");
        CheckMiss(client, key);
        return;
    }
    std::vector<char> buffer(cfg.block_size);
    FillPattern(buffer, key);
    auto saved = client.SaveKvCaches({uri}, Buffers(buffer));
    Require(saved.first == ER_OK && saved.second.size() == 1, "SaveKvCaches failed or missing actual URI");
    Require(NodeId(saved.second[0]) == cfg.expected_node_id, "saved URI is on wrong Provider");
    Locations finished = {{{"spec_0", saved.second[0]}}};
    Require(client.FinishWrite("affinity_finish", write.write_session_id, BlockMaskOffset{1}, finished) == ER_OK,
            "FinishWrite failed");
    CheckLocal(client, cfg, key);
}

struct ReplicationBuffer {
    std::vector<char> data;
    std::atomic<bool> released{false};
    std::atomic<bool> completed{false};
    ReplicationResult result;
};

void Replicate(ManagerClient &client, const Config &cfg, int64_t key) {
    const auto caller = client.GetCallerNode();
    Require(!caller.empty(), "caller Provider UUID is empty");
    ClientReplicationHint hint;
    bool received = false;
    std::string source;
    std::vector<char> source_data;
    for (int round = 1; round <= cfg.query_rounds; ++round) {
        std::vector<ClientReplicationHint> hints;
        auto locations = Query(client, key, hints);
        const auto &uri = SingleUri(locations);
        Require(!NodeId(uri).empty() && NodeId(uri) != cfg.expected_node_id,
                "pre-replication read is not remote: " + uri);
        if (source.empty()) {
            source = uri;
            source_data = ReadAndCheck(client, cfg, key, source);
        }
        Require(source == uri, "source changed while accumulating read frequency");
        if (hints.empty()) continue;
        Require(hints.size() == 1, "expected one hint per block");
        hint = hints[0];
        Require(hint.block_key == key && hint.target_node_id == caller && hint.source_uri == source,
                "hint key/source/target does not match the request");
        Require(cfg.expected_hint_round == 0 || round == cfg.expected_hint_round,
                "hint emitted at unexpected query round " + std::to_string(round));
        printf("HINT_RECEIVED key=%lld round=%d target=%s\n", static_cast<long long>(key), round, caller.c_str());
        received = true;
        break;
    }
    Require(received, "no replication hint within query-round budget");

    auto owned = std::make_shared<ReplicationBuffer>();
    if (cfg.role == "reader_piggyback") {
        owned->data = std::move(source_data);
        // Capture ownership: even a timeout must not free memory still used by the executor.
        const bool admitted = client.ReplicateWithDataAsync(
            hint,
            owned->data.data(),
            owned->data.size(),
            [owned]() { owned->released.store(true, std::memory_order_release); },
            [owned](const ReplicationResult &result) {
                owned->result = result;
                owned->completed.store(true, std::memory_order_release);
            });
        Require(admitted, "piggyback replication was not admitted");
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(cfg.wait_seconds);
    bool local = false;
    while (std::chrono::steady_clock::now() < deadline) {
        std::vector<ClientReplicationHint> hints;
        auto locations = Query(client, key, hints);
        if (!IsMiss(locations)) {
            const auto &uri = SingleUri(locations);
            if (NodeId(uri) == cfg.expected_node_id) {
                Require(uri != source, "replica URI must differ from source");
                if (cfg.role != "reader_piggyback" ||
                    (owned->released.load(std::memory_order_acquire) &&
                     owned->completed.load(std::memory_order_acquire))) {
                    local = true;
                    break;
                }
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    Require(local, "replica not published on expected local Provider before deadline");
    if (cfg.role == "reader_piggyback") {
        Require(owned->completed.load(std::memory_order_acquire), "replication result callback was not invoked");
        Require(owned->result.block_key == key && owned->result.target_node_id == caller,
                "replication result key/target mismatch");
        Require(owned->result.error_code == ER_OK &&
                    (owned->result.outcome == ReplicationOutcome::SUCCEEDED ||
                     owned->result.outcome == ReplicationOutcome::ALREADY_EXISTS),
                "replication result reported failure: " + std::to_string(owned->result.error_code));
        Require(owned->result.copied_bytes == static_cast<uint64_t>(cfg.block_size),
                "replication result copied byte count mismatch");
        printf("REPLICATION_RESULT key=%lld outcome=%d bytes=%llu latency_us=%llu\n",
               static_cast<long long>(key),
               static_cast<int>(owned->result.outcome),
               static_cast<unsigned long long>(owned->result.copied_bytes),
               static_cast<unsigned long long>(owned->result.latency_us));
    }
    CheckLocal(client, cfg, key);
}

void Run(const Config &cfg) {
    InitParams params;
    params.role_type = RoleType::HYBRID;
    params.self_location_spec_name = "spec_0";
    auto client = ManagerClient::Create(BuildClientConfig(cfg), params);
    Require(client != nullptr, "ManagerClient::Create failed");
    for (int i = 0; i < cfg.num_blocks; ++i) {
        auto key = BlockKey(cfg, i);
        if (cfg.role == "writer" || cfg.role == "writer_abort") {
            Write(*client, cfg, key, cfg.role == "writer_abort");
        } else if (cfg.role == "reader_piggyback" || cfg.role == "reader_async") {
            Replicate(*client, cfg, key);
        } else if (cfg.role == "reader_local") {
            CheckLocal(*client, cfg, key);
        } else if (cfg.role == "reader_miss") {
            CheckMiss(*client, key);
        } else {
            Require(client->RemoveCache("affinity_remove", {key}, {}, BlockMaskOffset{0}) == ER_OK,
                    "RemoveCache failed");
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(cfg.wait_seconds);
            bool removed = false;
            do {
                std::vector<ClientReplicationHint> hints;
                if (IsMiss(Query(*client, key, hints)) && hints.empty()) { removed = true; break; }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            } while (std::chrono::steady_clock::now() < deadline);
            Require(removed, "removed key still query-visible");
        }
        printf("BLOCK_OK role=%s block=%d key=%lld\n", cfg.role.c_str(), i, static_cast<long long>(key));
    }
    printf("E2E_OK role=%s blocks=%d\n", cfg.role.c_str(), cfg.num_blocks);
}
} // namespace

int main(int argc, char **argv) {
    try {
        Run(ParseArgs(argc, argv));
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "E2E_FAILED: %s\n", e.what());
        return 1;
    }
}
