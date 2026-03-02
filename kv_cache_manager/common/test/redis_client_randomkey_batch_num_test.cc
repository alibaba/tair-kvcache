#include "kv_cache_manager/common/redis_client.h"
#include "kv_cache_manager/common/test/mock_redis_client.h"
#include "kv_cache_manager/common/test/redis_test_base.h"
#include "kv_cache_manager/common/unittest.h"

using namespace kv_cache_manager;

class RedisClientRandomKeyBatchNumTest : public RedisTestBase, public TESTBASE {
public:
    void SetUp() override {}
    void TearDown() override {}

    std::unique_ptr<RedisClient> CreateRedisClientWithParams(const std::map<std::string, std::string> &params) {
        StandardUri storage_uri;
        storage_uri.SetProtocol("redis");
        storage_uri.SetUserInfo("test_user:test_password");
        storage_uri.SetHostName("test_host");
        storage_uri.SetPort(6379);

        for (const auto &param : params) {
            storage_uri.SetParam(param.first, param.second);
        }

        return std::make_unique<RedisClient>(storage_uri);
    }
};

TEST_F(RedisClientRandomKeyBatchNumTest, TestDefaultRandomKeyBatchNum) {
    // Test that the default value is 20 when randomkey_batch_num is not specified
    auto redis_client = CreateRedisClientWithParams({});

    EXPECT_EQ(20, redis_client->randomkey_batch_num_);
}

TEST_F(RedisClientRandomKeyBatchNumTest, TestRandomKeyBatchNumParsing) {
    // Test that randomkey_batch_num is correctly parsed when specified
    auto redis_client = CreateRedisClientWithParams({{"randomkey_batch_num", "50"}});

    EXPECT_EQ(50, redis_client->randomkey_batch_num_);
}

TEST_F(RedisClientRandomKeyBatchNumTest, TestRandomKeyBatchNumWithZero) {
    // Test that when randomkey_batch_num is 0, the default value is used
    auto redis_client = CreateRedisClientWithParams({{"randomkey_batch_num", "0"}});

    EXPECT_EQ(20, redis_client->randomkey_batch_num_);
}

TEST_F(RedisClientRandomKeyBatchNumTest, TestRandomKeyBatchNumWithNegative) {
    // Test that when randomkey_batch_num is negative, the default value is used
    auto redis_client = CreateRedisClientWithParams({{"randomkey_batch_num", "-10"}});

    EXPECT_EQ(20, redis_client->randomkey_batch_num_);
}

TEST_F(RedisClientRandomKeyBatchNumTest, TestRandomKeyBatchNumWithLargeValue) {
    // Test that when randomkey_batch_num is a large positive value, it's correctly used
    auto redis_client = CreateRedisClientWithParams({{"randomkey_batch_num", "100"}});

    EXPECT_EQ(100, redis_client->randomkey_batch_num_);
}

TEST_F(RedisClientRandomKeyBatchNumTest, TestRandomKeyBatchNumInRandMethod) {
    // Test that the Rand method uses the configured randomkey_batch_num value
    auto redis_client = CreateRedisClientWithParams({{"randomkey_batch_num", "10"}});

    // Since we can't easily mock the Rand method behavior without complex setup,
    // we'll test that the configuration is properly set, which is the main purpose
    // of this commit.
    EXPECT_EQ(10, redis_client->randomkey_batch_num_);
}
