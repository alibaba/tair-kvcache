#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/optimizer/config/optimizer_lite_hit_config.h"
#include "kv_cache_manager/optimizer/config/optimizer_registry_manager.h"
#include "kv_cache_manager/optimizer/liteHit/facts_csv.h"
#include "kv_cache_manager/optimizer/liteHit/facts_query.h"
#include "kv_cache_manager/optimizer/manager/lite_hit_offline_runner.h"
#include "kv_cache_manager/optimizer/manager/online_runtime/online_optimizer_manager.h"

namespace kv_cache_manager {

class LiteHitOfflineRunnerTest : public TESTBASE {
protected:
    static OptimizerInstanceGroup MakeGroup(const std::string &name = "g1") {
        OptimizerInstanceGroup group;
        group.set_name(name);
        // Capacities are irrelevant to the facts replay; they exist only to
        // satisfy the shared registration path.
        group.set_capacity_gb({1.0});
        group.set_eviction_policy("lru");
        group.set_enable_theoretical_max_cache(false);
        group.set_ttl_seconds(0);
        return group;
    }

    static OptimizerInstanceInfo MakeInfo(const std::string &instance_id, int32_t block_size = 4) {
        return OptimizerInstanceInfo("g1",
                                     instance_id,
                                     block_size,
                                     {LocationSpecInfo("tp0", 8192), LocationSpecInfo("tp1", 8192)},
                                     {LocationSpecGroup("full", {"tp0", "tp1"})},
                                     0,
                                     OptimizerStateInfo("full", ""));
    }

    // Hybrid full+linear instance: full charge 16384, mamba charge 4096.
    static OptimizerInstanceInfo
    MakeHybridInfo(const std::string &instance_id, int32_t block_size = 4, int32_t linear_step_tokens = 12) {
        return OptimizerInstanceInfo(
            "g1",
            instance_id,
            block_size,
            {LocationSpecInfo("tp0_F0", 8192),
             LocationSpecInfo("tp1_F0", 8192),
             LocationSpecInfo("tp0_L1", 2048),
             LocationSpecInfo("tp1_L1", 2048)},
            {LocationSpecGroup("F0", {"tp0_F0", "tp1_F0"}), LocationSpecGroup("L1", {"tp0_L1", "tp1_L1"})},
            linear_step_tokens,
            OptimizerStateInfo("F0", "L1"));
    }

    static std::string TraceLine(const std::string &instance_id,
                                 const std::string &trace_id,
                                 int64_t timestamp_ns,
                                 const std::vector<int64_t> &keys,
                                 int64_t input_len) {
        std::ostringstream line;
        line << R"({"type":"request","instance_id":")" << instance_id << R"(","trace_id":")" << trace_id
             << R"(","timestamp_ns":)" << timestamp_ns << R"(,"keys":[)";
        for (std::size_t i = 0; i < keys.size(); ++i) {
            if (i > 0) {
                line << ',';
            }
            line << keys[i];
        }
        line << R"(],"input_len":)" << input_len << R"(,"query_type":"prefix_match","block_mask":[]})";
        return line.str();
    }

    std::string WriteTrace(const std::string &name, const std::vector<std::string> &lines) {
        const std::string path = GetTestTempRootPath() + "/" + name;
        std::ofstream out(path);
        for (const auto &line : lines) {
            out << line << '\n';
        }
        out.close();
        return path;
    }

    OptimizerLiteHitConfig MakeConfig(const std::string &trace_path, const std::string &output_dir) {
        OptimizerLiteHitConfig config;
        config.set_trace_file_path(trace_path);
        config.set_output_result_path(output_dir);
        config.set_instance_groups({MakeGroup()});
        config.set_instances({MakeInfo("i1")});
        // The fixtures produce 4-token blocks; the production default is 256.
        config.set_block_size(4);
        return config;
    }

    static std::vector<std::string> ReadLines(const std::string &path) {
        std::ifstream in(path);
        std::vector<std::string> lines;
        std::string line;
        while (std::getline(in, line)) {
            lines.push_back(line);
        }
        return lines;
    }
};

TEST_F(LiteHitOfflineRunnerTest, FactsCsvRowRoundTrips) {
    LiteHitFactRecord record;
    record.trace_id = "id,with\"quote";
    record.instance_id = "engine-0";
    record.timestamp_ns = 1720000000000;
    record.input_token_len = 900;
    record.block_size_tokens = 256;
    record.block_bytes = 4194304;
    record.is_full_rle = true;
    record.full_rle_fact.hit_curve = {{1, 2}, {4, 1}};

    const std::string row = SerializeLiteHitFactRow(record);
    LiteHitFactRecord parsed;
    std::string error;
    ASSERT_TRUE(ParseLiteHitFactRow(row, parsed, error)) << error;
    EXPECT_EQ(record.trace_id, parsed.trace_id);
    EXPECT_EQ(record.instance_id, parsed.instance_id);
    EXPECT_EQ(record.timestamp_ns, parsed.timestamp_ns);
    EXPECT_EQ(record.input_token_len, parsed.input_token_len);
    EXPECT_EQ(record.block_size_tokens, parsed.block_size_tokens);
    EXPECT_EQ(record.block_bytes, parsed.block_bytes);
    EXPECT_TRUE(parsed.is_full_rle);
    EXPECT_EQ(record.full_rle_fact.hit_curve, parsed.full_rle_fact.hit_curve);

    record.full_rle_fact.hit_curve.clear();
    LiteHitFactRecord parsed_empty;
    ASSERT_TRUE(ParseLiteHitFactRow(SerializeLiteHitFactRow(record), parsed_empty, error)) << error;
    EXPECT_TRUE(parsed_empty.full_rle_fact.hit_curve.empty());

    LiteHitFactRecord bad;
    EXPECT_FALSE(ParseLiteHitFactRow("a,b,c", bad, error));
    EXPECT_FALSE(ParseLiteHitFactRow("t,i,1,2,3,4,\"[[1]]\"", bad, error));
    EXPECT_FALSE(ParseLiteHitFactRow("t,i,x,2,3,4,\"[]\"", bad, error));
}

TEST_F(LiteHitOfflineRunnerTest, TtlGroupLayersTtlOntoTheHitCurve) {
    // Group TTL is 3s; timestamps in ns make ages 2s/1s/3s at the later
    // requests. TTL rows are ordinary hit-curve facts: the fixed TTL was
    // applied during the replay, the capacity axis stays queryable.
    const std::string trace_path = WriteTrace("facts_ttl.jsonl",
                                              {
                                                  TraceLine("i1", "r1", 1000000000, {1, 2, 3}, 13),
                                                  TraceLine("i1", "r2", 3000000000, {1, 2, 9}, 12),
                                                  TraceLine("i1", "r3", 4000000000, {1, 2, 3}, 13),
                                              });
    const std::string output_dir = GetTestTempRootPath() + "/ttl";
    ASSERT_EQ(0, ::system(("mkdir -p " + output_dir).c_str()));
    OptimizerLiteHitConfig config = MakeConfig(trace_path, output_dir);
    OptimizerInstanceGroup group = MakeGroup();
    group.set_ttl_seconds(3);
    config.set_instance_groups({group});
    ASSERT_TRUE(LiteHitOfflineRunner(config).Run());

    const std::vector<std::string> lines = ReadLines(output_dir + "/" + kLiteHitFactsFileName);
    ASSERT_EQ(4, lines.size());
    std::string error;
    LiteHitFactRecord r1;
    ASSERT_TRUE(ParseLiteHitFactRow(lines[1], r1, error)) << error;
    EXPECT_TRUE(r1.full_rle_fact.hit_curve.empty()); // cold

    LiteHitFactRecord r2;
    ASSERT_TRUE(ParseLiteHitFactRow(lines[2], r2, error)) << error;
    // Blocks 1,2 are 2s old (alive); 9 was never seen.
    EXPECT_EQ((std::vector<HitCurveSegment>{{1, 2}}), r2.full_rle_fact.hit_curve);

    LiteHitFactRecord r3;
    ASSERT_TRUE(ParseLiteHitFactRow(lines[3], r3, error)) << error;
    // Blocks 1,2 refreshed by r2 (1s old); block 3 is 3s old: deadline
    // reached, the prefix stops there for every capacity. Without the TTL
    // this request would produce thresholds 1,2,4.
    EXPECT_EQ((std::vector<HitCurveSegment>{{1, 2}}), r3.full_rle_fact.hit_curve);

    // Capacity stays a query-time axis on top of the fixed TTL.
    const double capacity_1_block_gb = 16384.0 / (1024.0 * 1024.0 * 1024.0);
    const std::string query_log = output_dir + "/query.jsonl";
    ASSERT_TRUE(
        RunLiteHitFactsQuery(output_dir + "/" + kLiteHitFactsFileName, {capacity_1_block_gb, -1.0}, query_log, error))
        << error;
    const std::vector<std::string> query_lines = ReadLines(query_log);
    ASSERT_EQ(5, query_lines.size()); // 3 requests + i1 summary + overall
    EXPECT_NE(std::string::npos, query_lines[2].find("\"hit_blocks\":[1,2]"));
    EXPECT_NE(std::string::npos, query_lines[4].find("\"total_hit_blocks\":[2,4]"));
    EXPECT_NE(std::string::npos, query_lines[4].find("\"total_input_tokens\":38"));
}

TEST_F(LiteHitOfflineRunnerTest, RejectsNegativeTtlGroup) {
    const std::string trace_path = WriteTrace("facts_ttl_bad.jsonl", {TraceLine("i1", "r1", 1000, {1}, 4)});
    OptimizerLiteHitConfig config = MakeConfig(trace_path, GetTestTempRootPath());
    OptimizerInstanceGroup group = MakeGroup();
    group.set_ttl_seconds(-1);
    config.set_instance_groups({group});
    EXPECT_FALSE(LiteHitOfflineRunner(config).Run());
}

TEST_F(LiteHitOfflineRunnerTest, ByteStepAndFullRleFactsCsvRowsRoundTrip) {
    LiteHitFactRecord record;
    record.trace_id = "m1";
    record.instance_id = "mamba-0";
    record.timestamp_ns = 1720000000001;
    record.input_token_len = 700;
    record.block_size_tokens = 128;
    record.block_bytes = 65536;
    record.fact.points = {{4096, 1}, {131072, 5}};

    const std::string row = SerializeLiteHitFactRow(record);
    EXPECT_NE(std::string::npos, row.find("bytes:"));

    LiteHitFactRecord parsed;
    std::string error;
    ASSERT_TRUE(ParseLiteHitFactRow(row, parsed, error)) << error;
    EXPECT_FALSE(parsed.is_full_rle);
    EXPECT_EQ(record.fact.points, parsed.fact.points);
    EXPECT_TRUE(parsed.full_rle_fact.hit_curve.empty());

    record.fact.points.clear();
    LiteHitFactRecord parsed_empty;
    ASSERT_TRUE(ParseLiteHitFactRow(SerializeLiteHitFactRow(record), parsed_empty, error)) << error;
    EXPECT_FALSE(parsed_empty.is_full_rle);
    EXPECT_TRUE(parsed_empty.fact.points.empty());

    // New Full-only rows are explicitly tagged RLE.
    LiteHitFactRecord new_full;
    new_full.is_full_rle = true;
    new_full.full_rle_fact.hit_curve = {{1, 2}};
    EXPECT_NE(std::string::npos, SerializeLiteHitFactRow(new_full).find("rle:"));

    // Legacy unprefixed Full RLE and "mamba:" byte-step rows stay readable.
    LiteHitFactRecord full_row;
    ASSERT_TRUE(ParseLiteHitFactRow("t,i,1,2,3,4,\"[[1,2]]\"", full_row, error)) << error;
    EXPECT_TRUE(full_row.is_full_rle);
    EXPECT_EQ((std::vector<HitCurveSegment>{{1, 2}}), full_row.full_rle_fact.hit_curve);
    LiteHitFactRecord legacy_byte_step;
    ASSERT_TRUE(ParseLiteHitFactRow("t,i,1,2,3,4,\"mamba:[[4096,1]]\"", legacy_byte_step, error)) << error;
    EXPECT_FALSE(legacy_byte_step.is_full_rle);
    EXPECT_EQ((std::vector<ByteStepPoint>{{4096, 1}}), legacy_byte_step.fact.points);
}

TEST_F(LiteHitOfflineRunnerTest, PublishesFactsAndMatchesOnlineReplay) {
    const std::string trace_path = WriteTrace("facts_ok.jsonl",
                                              {
                                                  TraceLine("i1", "r1", 1000, {1, 2, 3}, 13),
                                                  TraceLine("i1", "r2", 2000, {1, 2, 9}, 12),
                                                  TraceLine("i1", "r3", 3000, {1, 2, 3}, 13),
                                              });
    const std::string output_dir = GetTestTempRootPath();
    LiteHitOfflineRunner runner(MakeConfig(trace_path, output_dir));
    ASSERT_TRUE(runner.Run());

    const std::string facts_path = output_dir + "/" + kLiteHitFactsFileName;
    const std::vector<std::string> lines = ReadLines(facts_path);
    ASSERT_EQ(4, lines.size());
    EXPECT_EQ(kLiteHitFactsCsvHeader, lines[0]);

    LiteHitFactRecord r3;
    std::string error;
    ASSERT_TRUE(ParseLiteHitFactRow(lines[3], r3, error)) << error;
    EXPECT_EQ("r3", r3.trace_id);
    EXPECT_EQ(13, r3.input_token_len);
    EXPECT_EQ(4, r3.block_size_tokens);
    EXPECT_EQ(16384, r3.block_bytes);
    // Fork [1,2,9] interleaved key 9 between 2 and 3: thresholds 1,2,4.
    EXPECT_EQ((std::vector<HitCurveSegment>{{1, 2}, {4, 1}}), r3.full_rle_fact.hit_curve);

    // Reprojection of the facts must match an online replay of the same
    // trace with the same capacities.
    const double capacity_2_blocks_gb = 2.0 * 16384.0 / (1024.0 * 1024.0 * 1024.0);
    const std::string query_log = output_dir + "/query.jsonl";
    ASSERT_TRUE(RunLiteHitFactsQuery(facts_path, {capacity_2_blocks_gb, -1.0}, query_log, error)) << error;
    const std::vector<std::string> query_lines = ReadLines(query_log);
    ASSERT_EQ(5, query_lines.size()); // 3 requests + i1 summary + overall summary
    EXPECT_NE(std::string::npos, query_lines[3].find("\"summary\":true,\"instance_id\":\"i1\""));
    EXPECT_NE(std::string::npos, query_lines[4].find("\"summary\":true"));
    EXPECT_NE(std::string::npos, query_lines[4].find("\"total_hit_blocks\":[4,5]"));
    EXPECT_NE(std::string::npos, query_lines[4].find("\"total_input_tokens\":38"));

    auto registry = std::make_shared<OptimizerRegistryManager>("");
    OnlineOptimizerManager manager(registry);
    OptimizerInstanceGroup group = MakeGroup();
    group.set_capacity_gb({capacity_2_blocks_gb});
    group.set_enable_theoretical_max_cache(true);
    ASSERT_EQ(EC_OK, registry->CreateInstanceGroup(group));
    RegisterInstanceResult reg_result;
    ASSERT_EQ(EC_OK, manager.RegisterInstance(MakeInfo("i1"), reg_result));

    int64_t online_cap2_hits = 0;
    int64_t online_infinite_hits = 0;
    for (const auto &request :
         std::vector<std::pair<std::vector<int64_t>, int64_t>>{{{1, 2, 3}, 13}, {{1, 2, 9}, 12}, {{1, 2, 3}, 13}}) {
        TraceQueryResult result;
        ASSERT_EQ(EC_OK, manager.TraceQuery("i1", request.first, request.second, result));
        online_cap2_hits += result.hit_count_per_capacity.at(0);
        online_infinite_hits += result.max_hit_count;
    }
    EXPECT_EQ(4, online_cap2_hits);
    EXPECT_EQ(5, online_infinite_hits);
}

TEST_F(LiteHitOfflineRunnerTest, PublishesMambaFactsAndMatchesOnlineReplay) {
    // block_size 4, linear_step 12 tokens -> Linear state every 3 blocks plus
    // the forced last block. 17 tokens -> 4 complete blocks.
    const std::string trace_path = WriteTrace("facts_mamba.jsonl",
                                              {
                                                  TraceLine("m1", "r1", 1000, {1, 2, 3, 4}, 17),
                                                  TraceLine("m1", "r2", 2000, {1, 2, 3, 4}, 17),
                                              });
    const std::string output_dir = GetTestTempRootPath() + "/mamba";
    ASSERT_EQ(0, ::system(("mkdir -p " + output_dir).c_str()));
    OptimizerLiteHitConfig config = MakeConfig(trace_path, output_dir);
    config.set_instances({MakeHybridInfo("m1")});
    ASSERT_TRUE(LiteHitOfflineRunner(config).Run());

    const std::vector<std::string> lines = ReadLines(output_dir + "/" + kLiteHitFactsFileName);
    ASSERT_EQ(3, lines.size());
    std::string error;
    LiteHitFactRecord r1;
    ASSERT_TRUE(ParseLiteHitFactRow(lines[1], r1, error)) << error;
    EXPECT_FALSE(r1.is_full_rle);
    EXPECT_TRUE(r1.fact.points.empty()); // cold: no recoverable Linear state
    EXPECT_EQ(16384, r1.block_bytes);    // per-row charge stays the Full charge

    // Warm request. Shared-pool recency after r1 (oldest->newest):
    // M3,F3,M2,F2,F1,F0. Linear state p=2: max(full prefix 3x16384, Mamba
    // 3x16384+4096) = 53248 -> 3 blocks; p=3: 4x16384+2x4096 = 73728 -> 4.
    LiteHitFactRecord r2;
    ASSERT_TRUE(ParseLiteHitFactRow(lines[2], r2, error)) << error;
    EXPECT_FALSE(r2.is_full_rle);
    EXPECT_EQ((std::vector<ByteStepPoint>{{53248, 3}, {73728, 4}}), r2.fact.points);

    const double capacity_gb = 53248.0 / (1024.0 * 1024.0 * 1024.0);
    const std::string query_log = output_dir + "/query.jsonl";
    ASSERT_TRUE(RunLiteHitFactsQuery(output_dir + "/" + kLiteHitFactsFileName, {capacity_gb, -1.0}, query_log, error))
        << error;
    const std::vector<std::string> query_lines = ReadLines(query_log);
    ASSERT_EQ(4, query_lines.size()); // 2 requests + m1 summary + overall
    EXPECT_NE(std::string::npos, query_lines[3].find("\"total_hit_blocks\":[3,4]"));
    EXPECT_NE(std::string::npos, query_lines[3].find("\"total_input_tokens\":34"));

    // Same trace through the online manager must agree.
    auto registry = std::make_shared<OptimizerRegistryManager>("");
    OnlineOptimizerManager manager(registry);
    OptimizerInstanceGroup group = MakeGroup();
    group.set_capacity_gb({capacity_gb});
    group.set_enable_theoretical_max_cache(true);
    ASSERT_EQ(EC_OK, registry->CreateInstanceGroup(group));
    RegisterInstanceResult reg_result;
    ASSERT_EQ(EC_OK, manager.RegisterInstance(MakeHybridInfo("m1"), reg_result));

    TraceQueryResult first;
    ASSERT_EQ(EC_OK, manager.TraceQuery("m1", {1, 2, 3, 4}, 17, first));
    EXPECT_EQ(0, first.hit_count_per_capacity.at(0));
    TraceQueryResult second;
    ASSERT_EQ(EC_OK, manager.TraceQuery("m1", {1, 2, 3, 4}, 17, second));
    EXPECT_EQ(3, second.hit_count_per_capacity.at(0));
    EXPECT_EQ(4, second.max_hit_count);
}

TEST_F(LiteHitOfflineRunnerTest, LinearInstanceHonorsGroupTtl) {
    const std::string trace_path = WriteTrace("facts_mamba_ttl.jsonl",
                                              {
                                                  TraceLine("m1", "r1", 1, {1}, 4),
                                                  TraceLine("m1", "r2", 1000000001, {1}, 4),
                                                  TraceLine("m1", "r3", 1000000002, {1}, 4),
                                              });
    const std::string output_dir = GetTestTempRootPath() + "/mamba_ttl";
    ASSERT_EQ(0, ::system(("mkdir -p " + output_dir).c_str()));
    OptimizerLiteHitConfig config = MakeConfig(trace_path, output_dir);
    OptimizerInstanceGroup group = MakeGroup();
    group.set_ttl_seconds(1);
    config.set_instance_groups({group});
    config.set_instances({MakeHybridInfo("m1")});
    ASSERT_TRUE(LiteHitOfflineRunner(config).Run());

    const std::vector<std::string> lines = ReadLines(output_dir + "/" + kLiteHitFactsFileName);
    ASSERT_EQ(4u, lines.size());
    std::string error;
    LiteHitFactRecord expired;
    ASSERT_TRUE(ParseLiteHitFactRow(lines[2], expired, error)) << error;
    EXPECT_TRUE(expired.fact.points.empty()); // strict TTL boundary
    LiteHitFactRecord revived;
    ASSERT_TRUE(ParseLiteHitFactRow(lines[3], revived, error)) << error;
    EXPECT_EQ((std::vector<ByteStepPoint>{{20480, 1}}), revived.fact.points);
}

TEST_F(LiteHitOfflineRunnerTest, ParallelPipelineMatchesSerialOutput) {
    std::vector<std::string> lines;
    for (int i = 0; i < 2000; ++i) {
        const int64_t base = (i % 13) * 100;
        lines.push_back(TraceLine("i1", "t" + std::to_string(i), 1000 + i, {base + 1, base + 2, base + 3}, 13));
    }
    const std::string trace_path = WriteTrace("facts_parallel.jsonl", lines);

    const std::string serial_dir = GetTestTempRootPath() + "/serial";
    const std::string parallel_dir = GetTestTempRootPath() + "/parallel";
    ASSERT_EQ(0, ::system(("mkdir -p " + serial_dir + " " + parallel_dir).c_str()));

    OptimizerLiteHitConfig serial_config = MakeConfig(trace_path, serial_dir);
    serial_config.set_pipeline_worker_count(1);
    ASSERT_TRUE(LiteHitOfflineRunner(serial_config).Run());

    OptimizerLiteHitConfig parallel_config = MakeConfig(trace_path, parallel_dir);
    parallel_config.set_pipeline_worker_count(4);
    ASSERT_TRUE(LiteHitOfflineRunner(parallel_config).Run());

    EXPECT_EQ(ReadLines(serial_dir + "/" + kLiteHitFactsFileName),
              ReadLines(parallel_dir + "/" + kLiteHitFactsFileName));
}

TEST_F(LiteHitOfflineRunnerTest, AppliesOverrideInstanceIdAndPrefixHash) {
    // Raw per-block hashes; the instance group enables prefix hashing, so
    // the two pods share one lane and the second request re-hits the shared
    // prefix.
    const std::string trace_path = WriteTrace("facts_override.jsonl",
                                              {
                                                  TraceLine("pod-a", "r1", 1000, {7, 8}, 8),
                                                  TraceLine("pod-b", "r2", 2000, {7, 8}, 8),
                                              });
    OptimizerLiteHitConfig config = MakeConfig(trace_path, GetTestTempRootPath());
    OptimizerInstanceGroup group = MakeGroup();
    group.set_enable_prefix_hash(true);
    config.set_instance_groups({group});
    config.set_instances({MakeInfo("service")});
    config.set_override_instance_id("service");
    ASSERT_TRUE(LiteHitOfflineRunner(config).Run());

    const std::vector<std::string> lines = ReadLines(GetTestTempRootPath() + "/" + kLiteHitFactsFileName);
    ASSERT_EQ(3, lines.size());
    LiteHitFactRecord r2;
    std::string error;
    ASSERT_TRUE(ParseLiteHitFactRow(lines[2], r2, error)) << error;
    EXPECT_EQ("service", r2.instance_id);
    EXPECT_EQ((std::vector<HitCurveSegment>{{1, 2}}), r2.full_rle_fact.hit_curve);
}

TEST_F(LiteHitOfflineRunnerTest, OverrideInstanceIdMustMatchConfiguredInstance) {
    const std::string trace_path = WriteTrace("facts_override_bad.jsonl", {TraceLine("pod-a", "r1", 1000, {1}, 4)});
    OptimizerLiteHitConfig config = MakeConfig(trace_path, GetTestTempRootPath());
    config.set_override_instance_id("missing");
    EXPECT_FALSE(LiteHitOfflineRunner(config).Run());
}

TEST_F(LiteHitOfflineRunnerTest, FailsFastWithoutPublishingFacts) {
    const std::string output_dir = GetTestTempRootPath() + "/failfast";
    ASSERT_EQ(0, ::system(("mkdir -p " + output_dir).c_str()));
    const std::string facts_path = output_dir + "/" + kLiteHitFactsFileName;

    // Out-of-order timestamps.
    const std::string unsorted = WriteTrace("facts_unsorted.jsonl",
                                            {
                                                TraceLine("i1", "r1", 2000, {1}, 4),
                                                TraceLine("i1", "r2", 1000, {2}, 4),
                                            });
    EXPECT_FALSE(LiteHitOfflineRunner(MakeConfig(unsorted, output_dir)).Run());
    EXPECT_FALSE(std::ifstream(facts_path).is_open());

    // Unknown instance.
    const std::string unknown = WriteTrace("facts_unknown.jsonl", {TraceLine("nope", "r1", 1000, {1}, 4)});
    EXPECT_FALSE(LiteHitOfflineRunner(MakeConfig(unknown, output_dir)).Run());
    EXPECT_FALSE(std::ifstream(facts_path).is_open());

    // Length contract violation.
    const std::string bad_len = WriteTrace("facts_badlen.jsonl", {TraceLine("i1", "r1", 1000, {1, 2}, 4)});
    EXPECT_FALSE(LiteHitOfflineRunner(MakeConfig(bad_len, output_dir)).Run());
    EXPECT_FALSE(std::ifstream(facts_path).is_open());

    // No valid Request/Get events at all.
    const std::string empty = WriteTrace("facts_empty.jsonl", {});
    EXPECT_FALSE(LiteHitOfflineRunner(MakeConfig(empty, output_dir)).Run());
    EXPECT_FALSE(std::ifstream(facts_path).is_open());
}

TEST_F(LiteHitOfflineRunnerTest, IgnoresWriteEvents) {
    const std::string trace_path = WriteTrace(
        "facts_write.jsonl",
        {
            TraceLine("i1", "r1", 1000, {1}, 4),
            R"({"type":"write","instance_id":"i1","trace_id":"w1","timestamp_ns":1500,"keys":[9],"ttl_us":0})",
            TraceLine("i1", "r2", 2000, {1}, 4),
        });
    const std::string output_dir = GetTestTempRootPath() + "/writes";
    ASSERT_EQ(0, ::system(("mkdir -p " + output_dir).c_str()));
    ASSERT_TRUE(LiteHitOfflineRunner(MakeConfig(trace_path, output_dir)).Run());

    const std::vector<std::string> lines = ReadLines(output_dir + "/" + kLiteHitFactsFileName);
    ASSERT_EQ(3, lines.size());
    LiteHitFactRecord r2;
    std::string error;
    ASSERT_TRUE(ParseLiteHitFactRow(lines[2], r2, error)) << error;
    // The write event neither produced a fact row nor touched the LRU.
    EXPECT_EQ((std::vector<HitCurveSegment>{{1, 1}}), r2.full_rle_fact.hit_curve);
}

TEST_F(LiteHitOfflineRunnerTest, FanoutSweepsMultipleBlockSizes) {
    // Trace granularity 4 tokens/block. Two lanes: bs4 replays as-is, bs8
    // re-blocks by keeping every 2nd prefix-chained key.
    const std::string trace_path = WriteTrace("facts_fanout.jsonl",
                                              {
                                                  TraceLine("ignored", "r1", 1000, {1, 2, 3}, 13),
                                                  TraceLine("ignored", "r2", 2000, {1, 2, 3}, 13),
                                              });
    const std::string output_dir = GetTestTempRootPath() + "/fanout";
    ASSERT_EQ(0, ::system(("mkdir -p " + output_dir).c_str()));
    OptimizerLiteHitConfig config = MakeConfig(trace_path, output_dir);
    config.set_instances({MakeInfo("bs4", 4), MakeInfo("bs8", 8)});
    config.set_fanout_all_instances(true);
    ASSERT_TRUE(LiteHitOfflineRunner(config).Run());

    const std::vector<std::string> lines = ReadLines(output_dir + "/" + kLiteHitFactsFileName);
    ASSERT_EQ(5, lines.size()); // header + 2 requests x 2 lanes

    std::string error;
    uint64_t bs4_rows = 0;
    uint64_t bs8_rows = 0;
    for (std::size_t i = 1; i < lines.size(); ++i) {
        LiteHitFactRecord record;
        ASSERT_TRUE(ParseLiteHitFactRow(lines[i], record, error)) << error;
        EXPECT_EQ(13, record.input_token_len);
        if (record.instance_id == "bs4") {
            ++bs4_rows;
            EXPECT_EQ(4, record.block_size_tokens);
            if (record.trace_id == "r2") {
                EXPECT_EQ((std::vector<HitCurveSegment>{{1, 3}}), record.full_rle_fact.hit_curve);
            }
        } else {
            ++bs8_rows;
            EXPECT_EQ("bs8", record.instance_id);
            EXPECT_EQ(8, record.block_size_tokens);
            if (record.trace_id == "r2") {
                // 13 tokens = 1 complete 8-token block (chained key at index 1).
                EXPECT_EQ((std::vector<HitCurveSegment>{{1, 1}}), record.full_rle_fact.hit_curve);
            }
        }
    }
    EXPECT_EQ(2, bs4_rows);
    EXPECT_EQ(2, bs8_rows);

    // The query emits one summary per lane plus the overall one.
    const std::string query_log = output_dir + "/query.jsonl";
    ASSERT_TRUE(RunLiteHitFactsQuery(output_dir + "/" + kLiteHitFactsFileName, {-1.0}, query_log, error)) << error;
    const std::vector<std::string> query_lines = ReadLines(query_log);
    ASSERT_EQ(7, query_lines.size()); // 4 requests + 2 instance summaries + overall
    EXPECT_NE(std::string::npos, query_lines[4].find("\"summary\":true,\"instance_id\":\"bs4\""));
    EXPECT_NE(std::string::npos, query_lines[5].find("\"summary\":true,\"instance_id\":\"bs8\""));
    EXPECT_NE(std::string::npos, query_lines[6].find("\"summary\":true,\"requests\":4"));
}

TEST_F(LiteHitOfflineRunnerTest, RejectsNonMultipleAnalysisBlockSize) {
    const std::string trace_path = WriteTrace("facts_badbs.jsonl", {TraceLine("i1", "r1", 1000, {1}, 4)});
    OptimizerLiteHitConfig config = MakeConfig(trace_path, GetTestTempRootPath());
    config.set_instances({MakeInfo("i1", 6)}); // 6 % 4 != 0: coarsening only
    EXPECT_FALSE(LiteHitOfflineRunner(config).Run());
}

} // namespace kv_cache_manager
