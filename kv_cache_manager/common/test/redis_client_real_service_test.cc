#include "kv_cache_manager/common/redis_client.h"
#include "kv_cache_manager/common/unittest.h"

namespace kv_cache_manager {

class RedisClientRealServiceTest : public TESTBASE {
public:
    void SetUp() override;

    void TearDown() override {}

public:
    void CheckReplyType(const redisReply *r, const int expected_type) const;

private:
    std::unique_ptr<RedisClient> redis_client_;
};

void RedisClientRealServiceTest::SetUp() {
    StandardUri storage_uri;
    storage_uri.user_info_ = "test_user:test_password";
    storage_uri.hostname_ = "localhost";
    storage_uri.port_ = 6379;
    storage_uri.params_["timeout_ms"] = "2000";
    storage_uri.params_["retry_count"] = "2";
    redis_client_ = std::make_unique<RedisClient>(storage_uri);
}

void RedisClientRealServiceTest::CheckReplyType(const redisReply *r, const int expected_type) const {
    ASSERT_TRUE(r);
    ASSERT_EQ(expected_type, r->type);
}

TEST_F(RedisClientRealServiceTest, TestOpen) {
    ASSERT_TRUE(redis_client_->Open());
    redis_client_->Close();
}

TEST_F(RedisClientRealServiceTest, TestSimple) {
    ASSERT_TRUE(redis_client_->Open());

    // use prefix to avoid key name conflict
    std::string key1 = "redis_client_test_simple_key1";
    std::string key2 = "redis_client_test_simple_key2";

    // put & get
    std::vector<std::string> keys{key1, key2};
    std::vector<std::map<std::string, std::string>> field_maps{{{"f1", "v1-1-0"}, {"f2", "v1-2-0"}},
                                                               {{"f1", "v2-1-0"}, {"f2", "v2-2-0"}}};
    auto ec_per_key = redis_client_->Set(keys, field_maps);
    ASSERT_EQ(std::vector<ErrorCode>(keys.size(), EC_OK), ec_per_key);

    std::vector<std::map<std::string, std::string>> expected_field_maps = field_maps;
    std::vector<std::map<std::string, std::string>> out_field_maps;
    ec_per_key = redis_client_->Get(keys, /*field_names*/ {"f1", "f2"}, out_field_maps);
    ASSERT_EQ(std::vector<ErrorCode>(keys.size(), EC_OK), ec_per_key);
    ASSERT_EQ(expected_field_maps, out_field_maps);

    // update & get
    std::vector<std::map<std::string, std::string>> update_field_maps{{{"f2", "v1-2-1"}}, {{"f1", "v2-1-1"}}};
    ec_per_key = redis_client_->Update(keys, update_field_maps);
    ASSERT_EQ(std::vector<ErrorCode>(keys.size(), EC_OK), ec_per_key);

    expected_field_maps = std::vector<std::map<std::string, std::string>>{{{"f1", "v1-1-0"}, {"f2", "v1-2-1"}},
                                                                          {{"f1", "v2-1-1"}, {"f2", "v2-2-0"}}};
    ec_per_key = redis_client_->Get(keys, /*field_names*/ {"f1", "f2"}, out_field_maps);
    ASSERT_EQ(std::vector<ErrorCode>(keys.size(), EC_OK), ec_per_key);
    ASSERT_EQ(expected_field_maps, out_field_maps);

    // get all fields
    ec_per_key = redis_client_->GetAllFields(keys, out_field_maps);
    ASSERT_EQ(std::vector<ErrorCode>(keys.size(), EC_OK), ec_per_key);
    ASSERT_EQ(expected_field_maps, out_field_maps);

    // exists
    std::vector<bool> out_is_exist_vec;
    ec_per_key = redis_client_->Exists(keys, out_is_exist_vec);
    ASSERT_EQ(std::vector<ErrorCode>(keys.size(), EC_OK), ec_per_key);
    ASSERT_EQ(std::vector<bool>(keys.size(), true), out_is_exist_vec);

    // scan
    std::vector<std::string> all_out_keys;
    std::string current_cursor = "0";
    std::string next_cursor;
    int64_t try_count = 0;
    while (try_count < 30) {
        std::vector<std::string> out_keys;
        auto ec = redis_client_->Scan(
            /*matching_prefix*/ "redis_client_test_simple_", current_cursor, /*limit*/ 50, next_cursor, out_keys);
        ASSERT_EQ(EC_OK, ec);
        try_count++;
        for (std::string &key : out_keys) {
            auto it = std::find(all_out_keys.begin(), all_out_keys.end(), key);
            ASSERT_TRUE(it == all_out_keys.end());
            all_out_keys.emplace_back(std::move(key));
        }
        if (next_cursor == "0") {
            break;
        }
        current_cursor = next_cursor;
    }
    ASSERT_LT(try_count, 30);
    ASSERT_THAT(all_out_keys, UnorderedElementsAre(key1, key2));

    // rand
    auto ec = redis_client_->Rand(/*matching_prefix*/ "redis_client_test_simple_", /*count*/ 2, all_out_keys);
    ASSERT_EQ(EC_OK, ec);
    ASSERT_THAT(all_out_keys, IsSubsetOf({key1, key2}));

    // delete & exists
    ec_per_key = redis_client_->Delete(keys);
    ASSERT_EQ(std::vector<ErrorCode>(keys.size(), EC_OK), ec_per_key);
    ec_per_key = redis_client_->Delete(keys);
    ASSERT_EQ(std::vector<ErrorCode>(keys.size(), EC_NOENT), ec_per_key);

    ec_per_key = redis_client_->Exists(keys, out_is_exist_vec);
    ASSERT_EQ(std::vector<ErrorCode>(keys.size(), EC_OK), ec_per_key);
    ASSERT_EQ(std::vector<bool>(keys.size(), false), out_is_exist_vec);

    redis_client_->Close();
}

TEST_F(RedisClientRealServiceTest, TestNotExistKey) {
    ASSERT_TRUE(redis_client_->Open());

    // use prefix to avoid key name conflict
    std::string key1 = "redis_client_test_not_exist_key_key1";
    std::string key2 = "redis_client_test_not_exist_key_key2";
    std::string key3 = "redis_client_test_not_exist_key_key3"; // not exist
    std::string key4 = "redis_client_test_not_exist_key_key4"; // not exist

    std::vector<std::string> keys{key1, key2};
    std::vector<std::map<std::string, std::string>> field_maps{{{"f1", "v1-1-0"}, {"f2", "v1-2-0"}},
                                                               {{"f1", "v2-1-0"}, {"f2", "v2-2-0"}}};
    auto ec_per_key = redis_client_->Set(keys, field_maps);
    ASSERT_EQ(std::vector<ErrorCode>(keys.size(), EC_OK), ec_per_key);

    // update
    std::vector<std::string> update_keys{key3, key4};
    std::vector<std::map<std::string, std::string>> update_field_maps{{{"f2", "v3-2-1"}}, {{"f1", "v4-1-1"}}};
    ec_per_key = redis_client_->Update(update_keys, update_field_maps);
    ASSERT_EQ(std::vector<ErrorCode>(keys.size(), EC_NOENT), ec_per_key);

    // get
    std::vector<std::string> all_keys{key1, key2, key3, key4};
    std::vector<std::map<std::string, std::string>> expected_field_maps = field_maps;
    expected_field_maps[0]["f3"] = "";
    expected_field_maps[1]["f3"] = "";
    expected_field_maps.emplace_back(
        std::map<std::string, std::string>{{"f1", ""}, {"f2", ""}, {"f3", ""}}); // not exist
    expected_field_maps.emplace_back(
        std::map<std::string, std::string>{{"f1", ""}, {"f2", ""}, {"f3", ""}}); // not exist
    std::vector<ErrorCode> expected_ec_per_key = {EC_OK, EC_OK, EC_OK, EC_OK};
    std::vector<std::map<std::string, std::string>> out_field_maps;
    ec_per_key = redis_client_->Get(all_keys, /*field_names*/ {"f1", "f2", "f3"}, out_field_maps);
    ASSERT_EQ(expected_ec_per_key, ec_per_key);
    ASSERT_EQ(expected_field_maps, out_field_maps);

    // get all fields
    std::vector<ErrorCode> expected_ec_per_key_2 = {EC_OK, EC_OK, EC_NOENT, EC_NOENT};
    expected_field_maps = field_maps;
    expected_field_maps.resize(4); // last 2 is empty
    ec_per_key = redis_client_->GetAllFields(all_keys, out_field_maps);
    ASSERT_EQ(expected_ec_per_key_2, ec_per_key);
    ASSERT_EQ(expected_field_maps, out_field_maps);

    // exists
    std::vector<bool> expected_is_exist_vec = {true, true, false, false};
    std::vector<bool> out_is_exist_vec;
    ec_per_key = redis_client_->Exists(all_keys, out_is_exist_vec);
    ASSERT_EQ(std::vector<ErrorCode>(all_keys.size(), EC_OK), ec_per_key);
    ASSERT_EQ(expected_is_exist_vec, out_is_exist_vec);

    redis_client_->Close();
}

TEST_F(RedisClientRealServiceTest, TestScanAndRandEmpty) {
    ASSERT_TRUE(redis_client_->Open());

    // scan
    std::vector<std::string> all_out_keys;
    std::string current_cursor = "0";
    std::string next_cursor;
    int64_t try_count = 0;
    while (try_count < 30) {
        std::vector<std::string> out_keys;
        auto ec = redis_client_->Scan(
            /*matching_prefix*/ "redis_client_test_simple_", current_cursor, /*limit*/ 50, next_cursor, out_keys);
        ASSERT_EQ(EC_OK, ec);
        try_count++;
        for (std::string &key : out_keys) {
            auto it = std::find(all_out_keys.begin(), all_out_keys.end(), key);
            ASSERT_TRUE(it == all_out_keys.end());
            all_out_keys.emplace_back(std::move(key));
        }
        if (next_cursor == "0") {
            break;
        }
        current_cursor = next_cursor;
    }
    ASSERT_LT(try_count, 30);
    ASSERT_TRUE(all_out_keys.empty());

    // rand
    auto ec =
        redis_client_->Rand(/*matching_prefix*/ "redis_client_test_scan_and_rand_empty_", /*count*/ 2, all_out_keys);
    ASSERT_EQ(EC_OK, ec);
    ASSERT_TRUE(all_out_keys.empty());

    redis_client_->Close();
}

TEST_F(RedisClientRealServiceTest, TestKeyAndFieldWithSpace) {
    ASSERT_TRUE(redis_client_->Open());

    // use prefix to avoid key name conflict
    std::string key1 = "redis client test key with space 1";
    std::string key2 = "redis client test key with space 2";

    // Test fields with spaces
    std::string field_with_space1 = "field with space 1";
    std::string field_with_space2 = "field with space 2";

    // put & get
    std::vector<std::string> keys{key1, key2};
    std::vector<std::map<std::string, std::string>> field_maps{
        {{field_with_space1, "value 1-1"}, {field_with_space2, "value 1-2"}},
        {{field_with_space1, "value 2-1"}, {field_with_space2, "value 2-2"}}};

    auto ec_per_key = redis_client_->Set(keys, field_maps);
    ASSERT_EQ(std::vector<ErrorCode>(keys.size(), EC_OK), ec_per_key);

    std::vector<std::map<std::string, std::string>> expected_field_maps = field_maps;
    std::vector<std::map<std::string, std::string>> out_field_maps;
    ec_per_key = redis_client_->Get(keys, /*field_names*/ {field_with_space1, field_with_space2}, out_field_maps);
    ASSERT_EQ(std::vector<ErrorCode>(keys.size(), EC_OK), ec_per_key);
    ASSERT_EQ(expected_field_maps, out_field_maps);

    // update & get
    std::vector<std::map<std::string, std::string>> update_field_maps{{{field_with_space2, "updated value 1-2"}},
                                                                      {{field_with_space1, "updated value 2-1"}}};

    ec_per_key = redis_client_->Update(keys, update_field_maps);
    ASSERT_EQ(std::vector<ErrorCode>(keys.size(), EC_OK), ec_per_key);

    expected_field_maps = std::vector<std::map<std::string, std::string>>{
        {{field_with_space1, "value 1-1"}, {field_with_space2, "updated value 1-2"}},
        {{field_with_space1, "updated value 2-1"}, {field_with_space2, "value 2-2"}}};

    ec_per_key = redis_client_->Get(keys, /*field_names*/ {field_with_space1, field_with_space2}, out_field_maps);
    ASSERT_EQ(std::vector<ErrorCode>(keys.size(), EC_OK), ec_per_key);
    ASSERT_EQ(expected_field_maps, out_field_maps);

    // get all fields
    ec_per_key = redis_client_->GetAllFields(keys, out_field_maps);
    ASSERT_EQ(std::vector<ErrorCode>(keys.size(), EC_OK), ec_per_key);
    ASSERT_EQ(expected_field_maps, out_field_maps);

    // exists
    std::vector<bool> out_is_exist_vec;
    ec_per_key = redis_client_->Exists(keys, out_is_exist_vec);
    ASSERT_EQ(std::vector<ErrorCode>(keys.size(), EC_OK), ec_per_key);
    ASSERT_EQ(std::vector<bool>(keys.size(), true), out_is_exist_vec);

    // scan
    std::vector<std::string> all_out_keys;
    std::string current_cursor = "0";
    std::string next_cursor;
    int64_t try_count = 0;
    while (try_count < 30) {
        std::vector<std::string> out_keys;
        auto ec = redis_client_->Scan(
            /*matching_prefix*/ "redis client test key with space",
            current_cursor,
            /*limit*/ 50,
            next_cursor,
            out_keys);
        ASSERT_EQ(EC_OK, ec);
        try_count++;
        for (std::string &key : out_keys) {
            auto it = std::find(all_out_keys.begin(), all_out_keys.end(), key);
            ASSERT_TRUE(it == all_out_keys.end());
            all_out_keys.emplace_back(std::move(key));
        }
        if (next_cursor == "0") {
            break;
        }
        current_cursor = next_cursor;
    }
    ASSERT_LT(try_count, 30);
    ASSERT_THAT(all_out_keys, UnorderedElementsAre(key1, key2));

    // rand
    auto ec = redis_client_->Rand(/*matching_prefix*/ "redis client test key with space", /*count*/ 2, all_out_keys);
    ASSERT_EQ(EC_OK, ec);
    ASSERT_THAT(all_out_keys, IsSubsetOf({key1, key2}));

    // delete & exists
    ec_per_key = redis_client_->Delete(keys);
    ASSERT_EQ(std::vector<ErrorCode>(keys.size(), EC_OK), ec_per_key);
    ec_per_key = redis_client_->Delete(keys);
    ASSERT_EQ(std::vector<ErrorCode>(keys.size(), EC_NOENT), ec_per_key);

    ec_per_key = redis_client_->Exists(keys, out_is_exist_vec);
    ASSERT_EQ(std::vector<ErrorCode>(keys.size(), EC_OK), ec_per_key);
    ASSERT_EQ(std::vector<bool>(keys.size(), false), out_is_exist_vec);

    redis_client_->Close();
}

} // namespace kv_cache_manager
