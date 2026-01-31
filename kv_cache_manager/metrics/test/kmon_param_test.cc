#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>

#include "kv_cache_manager/common/env_util.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/metrics/kmon_param.h"

using namespace ::testing;

namespace kv_cache_manager {

class KmonParamTest : public ::testing::Test {
protected:
    void SetUp() override { param_ = std::make_unique<KmonParam>(); }

    void TearDown() override { param_.reset(); }

    std::unique_ptr<KmonParam> param_;
};

TEST_F(KmonParamTest, TestConstructor) {
    // 测试构造函数是否正常工作
    EXPECT_NE(param_, nullptr);
    // 检查默认值
    EXPECT_FALSE(param_->kmonitor_enable_log_file_sink);
    EXPECT_FALSE(param_->kmonitor_manually_mode);
    EXPECT_EQ(param_->kmonitor_normal_sample_period, 1);
}

TEST_F(KmonParamTest, TestInitWithDefaultValues) {
    // 测试使用默认环境变量初始化
    bool result = param_->Init();
    EXPECT_TRUE(result);

    // 检查默认值
    EXPECT_EQ(param_->kmonitor_port, "4141");
    EXPECT_EQ(param_->kmonitor_sink_address, "127.0.0.1");
    EXPECT_EQ(param_->kmonitor_tenant, "default");
}

TEST_F(KmonParamTest, TestInitWithEnvironmentVariables) {
    // 测试使用环境变量初始化
    ScopedEnv env2("kmonitorPort", "8080");
    ScopedEnv env3("kmonitorEnableLogFileSink", "true");
    ScopedEnv env4("kmonitorNormalSamplePeriod", "10");

    bool result = param_->Init();
    EXPECT_TRUE(result);

    EXPECT_EQ(param_->kmonitor_port, "8080");
    EXPECT_TRUE(param_->kmonitor_enable_log_file_sink);
    EXPECT_EQ(param_->kmonitor_normal_sample_period, 10);
}

TEST_F(KmonParamTest, TestParseKmonitorTagsSuccess) {
    // 测试成功解析标签
    std::map<std::string, std::string> tags_map;
    bool result = KmonParam::ParseKmonitorTags("key1^value1@key2^value2", tags_map);
    EXPECT_TRUE(result);
    EXPECT_EQ(tags_map.size(), 2);
    EXPECT_EQ(tags_map["key1"], "value1");
    EXPECT_EQ(tags_map["key2"], "value2");
}

TEST_F(KmonParamTest, TestParseKmonitorTagsWithSpaces) {
    // 测试解析带空格的标签
    std::map<std::string, std::string> tags_map;
    bool result = KmonParam::ParseKmonitorTags("key1 ^ value1 @ key2 ^ value2 ", tags_map);
    EXPECT_TRUE(result);
    EXPECT_EQ(tags_map.size(), 2);
    EXPECT_EQ(tags_map["key1"], "value1");
    EXPECT_EQ(tags_map["key2"], "value2");
}

TEST_F(KmonParamTest, TestParseKmonitorTagsFailure) {
    // 测试解析失败的情况
    std::map<std::string, std::string> tags_map;
    bool result = KmonParam::ParseKmonitorTags("key1:value1@key2^value2", tags_map);
    EXPECT_FALSE(result);
}

TEST_F(KmonParamTest, TestInitWithTags) {
    // 测试使用标签环境变量初始化
    ScopedEnv env1("kmonitorTags", "env^test@version^1.0");

    bool result = param_->Init();
    EXPECT_TRUE(result);
    EXPECT_EQ(param_->kmonitor_tags.size(), 2);
    EXPECT_EQ(param_->kmonitor_tags["env"], "test");
    EXPECT_EQ(param_->kmonitor_tags["version"], "1.0");
}

TEST_F(KmonParamTest, TestInitWithEmptyTags) {
    // 测试使用空标签环境变量初始化
    ScopedEnv env1("kmonitorTags", "");

    bool result = param_->Init();
    EXPECT_TRUE(result);
    EXPECT_TRUE(param_->kmonitor_tags.empty());
}

TEST_F(KmonParamTest, TestInitWithInvalidTags) {
    // 测试使用无效标签环境变量初始化
    ScopedEnv env1("kmonitorTags", "invalid_tag_format");

    bool result = param_->Init();
    EXPECT_FALSE(result);
}

} // namespace kv_cache_manager