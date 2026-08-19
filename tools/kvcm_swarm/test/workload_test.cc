// Workload and key-derivation tests, including the golden SHA-256 vector that
// pins the V6D Connector key contract.
#include <gtest/gtest.h>
#include <set>
#include <string>

#include "tools/kvcm_swarm/clients/v6d/config.h"
#include "tools/kvcm_swarm/clients/v6d/key_mapper.h"
#include "tools/kvcm_swarm/clients/v6d/workload.h"

namespace kvcm_swarm {
namespace {

std::vector<CacheGroupSpec> MakeGroups() {
    CacheGroupSpec full;
    full.group_id = "full-0";
    full.kind = CacheGroupKind::kFullAttention;
    full.block_size_tokens = 4;
    full.object_size_bytes = 4096;
    full.spec_name = "v6d_4096";
    full.lookup_selector = FullSelector::kPrefix;

    CacheGroupSpec mamba;
    mamba.group_id = "mamba-0";
    mamba.kind = CacheGroupKind::kMamba;
    mamba.block_size_tokens = 8;
    mamba.object_size_bytes = 1024;
    mamba.spec_name = "v6d_1024";
    mamba.key_presence_rate = 0.5;
    return {full, mamba};
}

SessionClass MakeClass(uint64_t initial, uint64_t appended, uint64_t rewrite) {
    SessionClass session_class;
    session_class.name = "chat";
    session_class.weight = 1.0;
    session_class.turns = IntSpec(4);
    session_class.turn_interval = DurationSpec(Duration(std::chrono::milliseconds(10)));
    session_class.initial_tokens = IntSpec(initial);
    session_class.new_tokens_per_turn = IntSpec(appended);
    session_class.rewrite_tail_tokens = IntSpec(rewrite);
    session_class.shared_prefix_probability = 0.0;
    return session_class;
}

TEST(KeyMapperTest, Sha256GoldenVectors) {
    EXPECT_EQ(Sha256Hex(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    EXPECT_EQ(Sha256Hex("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    EXPECT_EQ(Sha256Hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    // 64-byte and 119-byte inputs exercise both padding branches.
    EXPECT_EQ(Sha256Hex(std::string(64, 'a')), "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb");
    EXPECT_EQ(Sha256Hex(std::string(119, 'a')), "31eba51c313a5c08226adf18d4a359cfdfd8d2e816b13f4af952f7ea6584dcfb")
        << "padding must cross into a second block";
}

TEST(KeyMapperTest, ObjectKeyAndBlockKeyFollowTheConnectorContract) {
    // object_key = block_hash + "_" + group_id
    EXPECT_EQ(MakeObjectKey("0011223344556677", "full-0"), "0011223344556677_full-0");
    // block_key = signed big-endian int64 of SHA256(object_key)[0:8]
    const std::string object_key = "0011223344556677_full-0";
    const std::string digest = Sha256Hex(object_key);
    uint64_t expected = 0;
    for (int i = 0; i < 8; ++i) {
        const std::string byte_hex = digest.substr(static_cast<size_t>(i) * 2, 2);
        expected = (expected << 8) | static_cast<uint64_t>(std::stoul(byte_hex, nullptr, 16));
    }
    EXPECT_EQ(ObjectKeyToBlockKey(object_key), static_cast<int64_t>(expected));

    // The same content in different groups yields different keys.
    EXPECT_NE(ObjectKeyToBlockKey(MakeObjectKey("abcd", "g1")), ObjectKeyToBlockKey(MakeObjectKey("abcd", "g2")));
    // Signed interpretation must be preserved: at least one key is negative.
    bool saw_negative = false;
    for (int i = 0; i < 200 && !saw_negative; ++i) {
        saw_negative = ObjectKeyToBlockKey(MakeObjectKey(BlockHashHex(static_cast<uint64_t>(i)), "g")) < 0;
    }
    EXPECT_TRUE(saw_negative);
}

TEST(WorkloadTest, OnlyCompleteBlocksProduceObjects) {
    const auto groups = MakeGroups();
    SharedPrefixPool pool_config;
    SeedDeriver seeds(1);
    SharedPrefixPoolState pool(pool_config, seeds, "b");
    Rng content(1);
    Rng shape(2);
    SessionWorkload workload;
    // 10 tokens: full block_size 4 -> 2 complete blocks, incomplete tail of 2.
    workload.Init(MakeClass(10, 0, 0), groups, pool, false, content, shape);
    ASSERT_EQ(workload.groups().size(), 2u);
    EXPECT_EQ(workload.groups()[0].complete_blocks(), 2u);
    // Mamba block_size 8 -> 1 complete block; a key may or may not exist.
    EXPECT_EQ(workload.groups()[1].complete_blocks(), 1u);
    for (const auto &object : workload.groups()[0].objects()) {
        EXPECT_LE(object.boundary_tokens, 8u);
        EXPECT_EQ(object.spec_name, "v6d_4096");
        EXPECT_EQ(object.object_size, 4096u);
    }
}

TEST(WorkloadTest, WorkingSetBytesSumsEveryCurrentGroupObject) {
    CacheGroupSpec left;
    left.group_id = "left";
    left.kind = CacheGroupKind::kFullAttention;
    left.block_size_tokens = 4;
    left.object_size_bytes = 4096;
    left.spec_name = "v6d_4096";
    left.lookup_selector = FullSelector::kPrefix;
    CacheGroupSpec right = left;
    right.group_id = "right";
    right.object_size_bytes = 1024;
    right.spec_name = "v6d_1024";

    SharedPrefixPool pool_config;
    SeedDeriver seeds(2);
    SharedPrefixPoolState pool(pool_config, seeds, "b");
    Rng content(3);
    Rng shape(4);
    SessionWorkload workload;
    workload.Init(MakeClass(16, 0, 0), {left, right}, pool, false, content, shape);

    // Four complete objects from each group. The budget is a simple sum even
    // when groups originate from the same token history.
    EXPECT_EQ(workload.WorkingSetBytes(), 4u * 4096u + 4u * 1024u);
}

TEST(WorkloadTest, FullAttentionKeysAreStableAndMambaKeysAreSparseButStable) {
    const auto groups = MakeGroups();
    SharedPrefixPool pool_config;
    SeedDeriver seeds(3);
    SharedPrefixPoolState pool(pool_config, seeds, "b");

    Rng content_a(11);
    Rng shape_a(12);
    SessionWorkload a;
    a.Init(MakeClass(64, 0, 0), groups, pool, false, content_a, shape_a);

    Rng content_b(11);
    Rng shape_b(12);
    SessionWorkload b;
    b.Init(MakeClass(64, 0, 0), groups, pool, false, content_b, shape_b);

    ASSERT_EQ(a.groups()[0].objects().size(), b.groups()[0].objects().size());
    for (size_t i = 0; i < a.groups()[0].objects().size(); ++i) {
        EXPECT_EQ(a.groups()[0].objects()[i].block_key, b.groups()[0].objects()[i].block_key);
    }
    // Full Attention produces a key for every complete block.
    EXPECT_EQ(a.groups()[0].objects().size(), a.groups()[0].complete_blocks());
    // Mamba is sparse: not every complete block has a key, and the selection is
    // fully determined by content, not by completion order.
    EXPECT_LT(a.groups()[1].objects().size(), a.groups()[1].complete_blocks());
    ASSERT_EQ(a.groups()[1].objects().size(), b.groups()[1].objects().size());
    for (size_t i = 0; i < a.groups()[1].objects().size(); ++i) {
        EXPECT_EQ(a.groups()[1].objects()[i].block_key, b.groups()[1].objects()[i].block_key);
    }
}

TEST(WorkloadTest, TailRewriteOnlyChangesAffectedBlocks) {
    const auto groups = MakeGroups();
    SharedPrefixPool pool_config;
    SeedDeriver seeds(5);
    SharedPrefixPoolState pool(pool_config, seeds, "b");
    Rng content(21);
    Rng shape(22);
    SessionWorkload workload;
    const SessionClass session_class = MakeClass(32, 8, 4);
    workload.Init(session_class, groups, pool, false, content, shape);
    const std::vector<GroupObject> before = workload.groups()[0].objects();
    ASSERT_EQ(before.size(), 8u);

    workload.ApplyTurn(session_class, content, shape);
    const std::vector<GroupObject> after = workload.groups()[0].objects();
    EXPECT_EQ(workload.token_count(), 40u);
    ASSERT_EQ(after.size(), 10u);
    // 4 rewritten tail tokens start at token 28 -> block 7 onwards changes.
    for (size_t block = 0; block < 7; ++block) {
        EXPECT_EQ(before[block].block_key, after[block].block_key) << "block " << block;
    }
    EXPECT_NE(before[7].block_key, after[7].block_key);
}

TEST(WorkloadTest, SharedPrefixRootProducesIdenticalKeysAcrossSessions) {
    const auto groups = MakeGroups();
    SharedPrefixPool pool_config;
    pool_config.root_count = 4;
    pool_config.prefix_tokens = IntSpec(16);
    SeedDeriver seeds(9);
    SharedPrefixPoolState pool(pool_config, seeds, "behavior");
    ASSERT_EQ(pool.root_count(), 4u);

    // Two sessions whose shape stream selects the same root and whose private
    // suffix is identical must share the leading block keys.
    Rng content_a(31);
    Rng shape_a(32);
    SessionWorkload a;
    a.Init(MakeClass(32, 0, 0), groups, pool, true, content_a, shape_a);
    Rng content_b(31);
    Rng shape_b(32);
    SessionWorkload b;
    b.Init(MakeClass(32, 0, 0), groups, pool, true, content_b, shape_b);
    ASSERT_TRUE(a.used_shared_prefix());
    EXPECT_EQ(a.shared_root_index(), b.shared_root_index());
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(a.groups()[0].objects()[i].block_key, b.groups()[0].objects()[i].block_key);
    }

    // A session on a different root must not collide on the first block.
    Rng content_c(41);
    Rng shape_c(99);
    SessionWorkload c;
    c.Init(MakeClass(32, 0, 0), groups, pool, true, content_c, shape_c);
    if (c.shared_root_index() != a.shared_root_index()) {
        EXPECT_NE(a.groups()[0].objects()[0].block_key, c.groups()[0].objects()[0].block_key);
    }
}

TEST(WorkloadTest, DifferentGroupsUseIndependentHashNamespaces) {
    CacheGroupSpec left;
    left.group_id = "left";
    left.kind = CacheGroupKind::kFullAttention;
    left.block_size_tokens = 4;
    left.object_size_bytes = 4096;
    left.spec_name = "v6d_4096";
    left.lookup_selector = FullSelector::kPrefix;
    CacheGroupSpec right = left;
    right.group_id = "right";

    SharedPrefixPool pool_config;
    SeedDeriver seeds(13);
    SharedPrefixPoolState pool(pool_config, seeds, "b");
    Rng content(51);
    Rng shape(52);
    SessionWorkload workload;
    workload.Init(MakeClass(16, 0, 0), {left, right}, pool, false, content, shape);
    ASSERT_EQ(workload.groups().size(), 2u);
    ASSERT_EQ(workload.groups()[0].objects().size(), workload.groups()[1].objects().size());
    std::set<int64_t> keys;
    for (const auto &object : workload.groups()[0].objects()) {
        keys.insert(object.block_key);
    }
    for (const auto &object : workload.groups()[1].objects()) {
        EXPECT_EQ(keys.count(object.block_key), 0u) << "identical content in different groups must not share a key";
    }
}

TEST(WorkloadTest, RewriteIsClampedToTheCurrentContext) {
    const auto groups = MakeGroups();
    SharedPrefixPool pool_config;
    SeedDeriver seeds(17);
    SharedPrefixPoolState pool(pool_config, seeds, "b");
    Rng content(61);
    Rng shape(62);
    SessionWorkload workload;
    SessionClass session_class = MakeClass(8, 0, 100);
    workload.Init(session_class, groups, pool, false, content, shape);
    workload.ApplyTurn(session_class, content, shape);
    EXPECT_EQ(workload.last_rewrite_tokens(), 8u);
    EXPECT_EQ(workload.token_count(), 8u);
}

} // namespace
} // namespace kvcm_swarm
