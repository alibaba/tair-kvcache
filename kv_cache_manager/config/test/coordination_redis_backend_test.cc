#include <gtest/gtest.h>
#include <memory>

#include "kv_cache_manager/common/redis_client_ext.h"
#include "kv_cache_manager/config/test/coordination_backend_test_base.h"

namespace kv_cache_manager {

const std::string kRedisUri = "redis://test_redis_user:test_redis_password@localhost:6379/"
                              "?timeout_ms=1000&retry_count=3&client_max_pool_size=2";

const std::string kRedisUriWithClusterName = "redis://test_redis_user:test_redis_password@localhost:6379/"
                                             "?timeout_ms=1000&retry_count=3&client_max_pool_size=2"
                                             "&cluster_name=test_cluster";

CoordinationBackendTestConfig redis_backend_config{
    .get_test_uri = [](CoordinationBackendTest *test_base) { return kRedisUri; },
    .set_up_ =
        [](CoordinationBackendTest *test_base) {
            RedisClientExt client(StandardUri::FromUri(kRedisUri));
            client.Open();
            client.FlushAll();
            client.Close();
        },
    .tear_down_ =
        [](CoordinationBackendTest *test_base) {
            RedisClientExt client(StandardUri::FromUri(kRedisUri));
            client.Open();
            client.FlushAll();
            client.Close();
        }};

CoordinationBackendTestConfig redis_backend_with_cluster_name_config{
    .get_test_uri = [](CoordinationBackendTest *test_base) { return kRedisUriWithClusterName; },
    .set_up_ =
        [](CoordinationBackendTest *test_base) {
            RedisClientExt client(StandardUri::FromUri(kRedisUriWithClusterName));
            client.Open();
            client.FlushAll();
            client.Close();
        },
    .tear_down_ =
        [](CoordinationBackendTest *test_base) {
            RedisClientExt client(StandardUri::FromUri(kRedisUriWithClusterName));
            client.Open();
            client.FlushAll();
            client.Close();
        }};

INSTANTIATE_TEST_SUITE_P(CoordinationBackendRedisTest,
                         CoordinationBackendTest,
                         testing::Values(redis_backend_config));

INSTANTIATE_TEST_SUITE_P(CoordinationBackendRedisWithClusterNameTest,
                         CoordinationBackendTest,
                         testing::Values(redis_backend_with_cluster_name_config));

// ============================================================
// cluster_name 隔离性 / 冲突性 综合测试
// 覆盖4种组合:
//   1. 一个为空 + 一个非空 → 隔离（不同 key 前缀）
//   2. 两个都为空 → 冲突（共享 key 空间）
//   3. 相同 cluster_name → 冲突（共享 key 空间）
//   4. 不同 cluster_name → 隔离（不同 key 前缀）
// ============================================================

const std::string kRedisUriWithClusterNameB = "redis://test_redis_user:test_redis_password@localhost:6379/"
                                              "?timeout_ms=1000&retry_count=3&client_max_pool_size=2"
                                              "&cluster_name=another_cluster";

// ---------- 场景1: 一个为空 + 一个非空 → 隔离 ----------
class CoordinationRedisClusterNameIsolationTest : public TESTBASE {
protected:
    void SetUp() override {
        RedisClientExt client(StandardUri::FromUri(kRedisUri));
        client.Open();
        client.FlushAll();
        client.Close();

        backend_no_cluster_ = CoordinationBackendFactory::CreateAndInitCoordinationBackend(kRedisUri);
        ASSERT_NE(backend_no_cluster_, nullptr);

        backend_with_cluster_ = CoordinationBackendFactory::CreateAndInitCoordinationBackend(kRedisUriWithClusterName);
        ASSERT_NE(backend_with_cluster_, nullptr);
    }

    void TearDown() override {
        RedisClientExt client(StandardUri::FromUri(kRedisUri));
        client.Open();
        client.FlushAll();
        client.Close();
    }

    std::unique_ptr<CoordinationBackend> backend_no_cluster_;
    std::unique_ptr<CoordinationBackend> backend_with_cluster_;
};

TEST_F(CoordinationRedisClusterNameIsolationTest, TestLockIsolation) {
    const std::string lock_key = "shared_lock";
    const int64_t ttl_ms = 5000;

    ASSERT_EQ(EC_OK, backend_no_cluster_->TryLock(lock_key, "holder_a", ttl_ms));
    ASSERT_EQ(EC_OK, backend_with_cluster_->TryLock(lock_key, "holder_b", ttl_ms));

    ASSERT_EQ(EC_OK, backend_no_cluster_->Unlock(lock_key, "holder_a"));
    ASSERT_EQ(EC_OK, backend_with_cluster_->Unlock(lock_key, "holder_b"));
}

TEST_F(CoordinationRedisClusterNameIsolationTest, TestKVIsolation) {
    const std::string key = "shared_key";

    ASSERT_EQ(EC_OK, backend_no_cluster_->SetValue(key, "value_no_cluster"));
    ASSERT_EQ(EC_OK, backend_with_cluster_->SetValue(key, "value_with_cluster"));

    std::string out;
    ASSERT_EQ(EC_OK, backend_no_cluster_->GetValue(key, out));
    EXPECT_EQ("value_no_cluster", out);

    ASSERT_EQ(EC_OK, backend_with_cluster_->GetValue(key, out));
    EXPECT_EQ("value_with_cluster", out);
}

// ---------- 场景2: 两个都为空 → 冲突 ----------
class CoordinationRedisBothEmptyClusterTest : public TESTBASE {
protected:
    void SetUp() override {
        RedisClientExt client(StandardUri::FromUri(kRedisUri));
        client.Open();
        client.FlushAll();
        client.Close();

        backend_a_ = CoordinationBackendFactory::CreateAndInitCoordinationBackend(kRedisUri);
        ASSERT_NE(backend_a_, nullptr);

        backend_b_ = CoordinationBackendFactory::CreateAndInitCoordinationBackend(kRedisUri);
        ASSERT_NE(backend_b_, nullptr);
    }

    void TearDown() override {
        RedisClientExt client(StandardUri::FromUri(kRedisUri));
        client.Open();
        client.FlushAll();
        client.Close();
    }

    std::unique_ptr<CoordinationBackend> backend_a_;
    std::unique_ptr<CoordinationBackend> backend_b_;
};

TEST_F(CoordinationRedisBothEmptyClusterTest, TestLockConflict) {
    const std::string lock_key = "shared_lock";
    const int64_t ttl_ms = 5000;

    ASSERT_EQ(EC_OK, backend_a_->TryLock(lock_key, "holder_a", ttl_ms));
    ASSERT_EQ(EC_EXIST, backend_b_->TryLock(lock_key, "holder_b", ttl_ms));

    ASSERT_EQ(EC_OK, backend_a_->Unlock(lock_key, "holder_a"));
    ASSERT_EQ(EC_OK, backend_b_->TryLock(lock_key, "holder_b", ttl_ms));
    ASSERT_EQ(EC_OK, backend_b_->Unlock(lock_key, "holder_b"));
}

TEST_F(CoordinationRedisBothEmptyClusterTest, TestKVConflict) {
    const std::string key = "shared_key";

    ASSERT_EQ(EC_OK, backend_a_->SetValue(key, "value_from_a"));

    std::string out;
    ASSERT_EQ(EC_OK, backend_b_->GetValue(key, out));
    EXPECT_EQ("value_from_a", out);

    ASSERT_EQ(EC_OK, backend_b_->SetValue(key, "value_from_b"));
    ASSERT_EQ(EC_OK, backend_a_->GetValue(key, out));
    EXPECT_EQ("value_from_b", out);
}

// ---------- 场景3: 相同 cluster_name → 冲突 ----------
class CoordinationRedisSameClusterNameTest : public TESTBASE {
protected:
    void SetUp() override {
        RedisClientExt client(StandardUri::FromUri(kRedisUri));
        client.Open();
        client.FlushAll();
        client.Close();

        backend_a_ = CoordinationBackendFactory::CreateAndInitCoordinationBackend(kRedisUriWithClusterName);
        ASSERT_NE(backend_a_, nullptr);

        backend_b_ = CoordinationBackendFactory::CreateAndInitCoordinationBackend(kRedisUriWithClusterName);
        ASSERT_NE(backend_b_, nullptr);
    }

    void TearDown() override {
        RedisClientExt client(StandardUri::FromUri(kRedisUri));
        client.Open();
        client.FlushAll();
        client.Close();
    }

    std::unique_ptr<CoordinationBackend> backend_a_;
    std::unique_ptr<CoordinationBackend> backend_b_;
};

TEST_F(CoordinationRedisSameClusterNameTest, TestLockConflict) {
    const std::string lock_key = "shared_lock";
    const int64_t ttl_ms = 5000;

    ASSERT_EQ(EC_OK, backend_a_->TryLock(lock_key, "holder_a", ttl_ms));
    ASSERT_EQ(EC_EXIST, backend_b_->TryLock(lock_key, "holder_b", ttl_ms));

    ASSERT_EQ(EC_OK, backend_a_->Unlock(lock_key, "holder_a"));
    ASSERT_EQ(EC_OK, backend_b_->TryLock(lock_key, "holder_b", ttl_ms));
    ASSERT_EQ(EC_OK, backend_b_->Unlock(lock_key, "holder_b"));
}

TEST_F(CoordinationRedisSameClusterNameTest, TestKVConflict) {
    const std::string key = "shared_key";

    ASSERT_EQ(EC_OK, backend_a_->SetValue(key, "value_from_a"));

    std::string out;
    ASSERT_EQ(EC_OK, backend_b_->GetValue(key, out));
    EXPECT_EQ("value_from_a", out);

    ASSERT_EQ(EC_OK, backend_b_->SetValue(key, "value_from_b"));
    ASSERT_EQ(EC_OK, backend_a_->GetValue(key, out));
    EXPECT_EQ("value_from_b", out);
}

// ---------- 场景4: 不同 cluster_name → 隔离 ----------
class CoordinationRedisDiffClusterNameTest : public TESTBASE {
protected:
    void SetUp() override {
        RedisClientExt client(StandardUri::FromUri(kRedisUri));
        client.Open();
        client.FlushAll();
        client.Close();

        backend_cluster_a_ = CoordinationBackendFactory::CreateAndInitCoordinationBackend(kRedisUriWithClusterName);
        ASSERT_NE(backend_cluster_a_, nullptr);

        backend_cluster_b_ = CoordinationBackendFactory::CreateAndInitCoordinationBackend(kRedisUriWithClusterNameB);
        ASSERT_NE(backend_cluster_b_, nullptr);
    }

    void TearDown() override {
        RedisClientExt client(StandardUri::FromUri(kRedisUri));
        client.Open();
        client.FlushAll();
        client.Close();
    }

    std::unique_ptr<CoordinationBackend> backend_cluster_a_;
    std::unique_ptr<CoordinationBackend> backend_cluster_b_;
};

TEST_F(CoordinationRedisDiffClusterNameTest, TestLockIsolation) {
    const std::string lock_key = "shared_lock";
    const int64_t ttl_ms = 5000;

    ASSERT_EQ(EC_OK, backend_cluster_a_->TryLock(lock_key, "holder_a", ttl_ms));
    ASSERT_EQ(EC_OK, backend_cluster_b_->TryLock(lock_key, "holder_b", ttl_ms));

    ASSERT_EQ(EC_OK, backend_cluster_a_->Unlock(lock_key, "holder_a"));
    ASSERT_EQ(EC_OK, backend_cluster_b_->Unlock(lock_key, "holder_b"));
}

TEST_F(CoordinationRedisDiffClusterNameTest, TestKVIsolation) {
    const std::string key = "shared_key";

    ASSERT_EQ(EC_OK, backend_cluster_a_->SetValue(key, "value_cluster_a"));
    ASSERT_EQ(EC_OK, backend_cluster_b_->SetValue(key, "value_cluster_b"));

    std::string out;
    ASSERT_EQ(EC_OK, backend_cluster_a_->GetValue(key, out));
    EXPECT_EQ("value_cluster_a", out);

    ASSERT_EQ(EC_OK, backend_cluster_b_->GetValue(key, out));
    EXPECT_EQ("value_cluster_b", out);
}

TEST_F(CoordinationRedisDiffClusterNameTest, TestLeaderElectionKeyIsolation) {
    const std::string leader_lock_key = "_TAIR_KVCM_LEADER_KEY";
    const std::string node_info_key = "_TAIR_KVCM_NODE_INFO_node1";
    const int64_t ttl_ms = 5000;

    ASSERT_EQ(EC_OK, backend_cluster_a_->TryLock(leader_lock_key, "node_a", ttl_ms));
    ASSERT_EQ(EC_OK, backend_cluster_a_->SetValue(node_info_key, "{\"host\":\"10.0.0.1\"}"));

    ASSERT_EQ(EC_OK, backend_cluster_b_->TryLock(leader_lock_key, "node_b", ttl_ms));
    ASSERT_EQ(EC_OK, backend_cluster_b_->SetValue(node_info_key, "{\"host\":\"10.0.0.2\"}"));

    std::string out;
    ASSERT_EQ(EC_OK, backend_cluster_a_->GetValue(node_info_key, out));
    EXPECT_EQ("{\"host\":\"10.0.0.1\"}", out);

    ASSERT_EQ(EC_OK, backend_cluster_b_->GetValue(node_info_key, out));
    EXPECT_EQ("{\"host\":\"10.0.0.2\"}", out);

    ASSERT_EQ(EC_OK, backend_cluster_a_->Unlock(leader_lock_key, "node_a"));
    ASSERT_EQ(EC_OK, backend_cluster_b_->Unlock(leader_lock_key, "node_b"));
}

} // namespace kv_cache_manager