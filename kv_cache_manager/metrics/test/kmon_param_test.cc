#include <cstdlib>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <sstream>

#include "kv_cache_manager/common/env_util.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/metrics/kmon_param.h"

using namespace ::testing;

namespace kv_cache_manager {

class KmonParamTest : public ::testing::Test {
protected:
    void SetUp() override {
        for (const auto *name : {"HIPPO_SLAVE_IP", "kmonitorSinkAddress", "kmonitorPort"}) {
            if (const auto *value = std::getenv(name)) {
                saved_environment_[name] = value;
            }
            unsetenv(name);
        }
        param_ = std::make_unique<KmonParam>();
    }

    void TearDown() override {
        param_.reset();
        for (const auto *name : {"HIPPO_SLAVE_IP", "kmonitorSinkAddress", "kmonitorPort"}) {
            unsetenv(name);
        }
        for (const auto &[name, value] : saved_environment_) {
            setenv(name.c_str(), value.c_str(), 1);
        }
    }

    std::unique_ptr<KmonParam> param_;
    std::map<std::string, std::string> saved_environment_;
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
    EXPECT_TRUE(param_->kmonitor_sink_address.empty());
    EXPECT_EQ(param_->hippo_slave_ip, "127.0.0.1");
    EXPECT_EQ(param_->kmonitor_tenant, "default");
    EXPECT_TRUE(param_->GetSinkAddress("host-from-sdk").empty());
    EXPECT_EQ(param_->GetSinkAddress(""), "127.0.0.1:4141");
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

TEST_F(KmonParamTest, TestHippoAddressTakesPriorityOverSdkDiscovery) {
    for (const auto *ip : {"10.0.0.1", "127.0.0.1"}) {
        ScopedEnv hippo_address("HIPPO_SLAVE_IP", ip);
        SCOPED_TRACE(ip);

        ASSERT_TRUE(param_->Init());

        const auto expected_address = std::string(ip) + ":4141";
        EXPECT_EQ(param_->hippo_slave_ip, ip);
        EXPECT_TRUE(param_->kmonitor_sink_address.empty());
        EXPECT_EQ(param_->GetSinkAddress("host-from-sdk"), expected_address);
        EXPECT_EQ(param_->GetSinkAddress("host-from-sdk", "10.94.0.97"), expected_address);
        EXPECT_EQ(param_->ResolveSinkAddress(), expected_address);
    }
}

TEST_F(KmonParamTest, TestExplicitSinkOverridesSdkDiscoveryAndHippoAddress) {
    ScopedEnv sink_address("kmonitorSinkAddress", "10.0.0.2");
    ScopedEnv hippo_address("HIPPO_SLAVE_IP", "10.0.0.1");

    ASSERT_TRUE(param_->Init());

    EXPECT_EQ(param_->GetSinkAddress("host-from-sdk"), "10.0.0.2:4141");
    EXPECT_EQ(param_->GetSinkAddress(""), "10.0.0.2:4141");
    EXPECT_EQ(param_->GetSinkAddress("host-from-sdk", "10.94.0.97"), "10.0.0.2:4141");
    EXPECT_EQ(param_->GetSinkAddress("", "10.94.0.97"), "10.0.0.2:4141");
    EXPECT_EQ(param_->ResolveSinkAddress(), "10.0.0.2:4141");
}

TEST_F(KmonParamTest, TestExplicitSinkUsesCustomPort) {
    ScopedEnv sink_address("kmonitorSinkAddress", "receiver.example.com");
    ScopedEnv port("kmonitorPort", "5141");

    ASSERT_TRUE(param_->Init());

    EXPECT_EQ(param_->GetSinkAddress("host-from-sdk"), "receiver.example.com:5141");
}

TEST_F(KmonParamTest, TestExplicitSinkPreservesEmptyPort) {
    ScopedEnv port("kmonitorPort", "");
    ScopedEnv hippo_address("HIPPO_SLAVE_IP", "10.0.0.1");
    for (const auto *address : {"127.0.0.1:5141", "receiver.example.com:5141", "10.0.0.2"}) {
        ScopedEnv sink_address("kmonitorSinkAddress", address);
        SCOPED_TRACE(address);

        ASSERT_TRUE(param_->Init());

        EXPECT_TRUE(param_->kmonitor_port.empty());
        EXPECT_EQ(param_->GetSinkAddress("host-from-sdk", "10.94.0.97"), address);
        EXPECT_EQ(param_->ResolveSinkAddress(), address);
    }
}

TEST_F(KmonParamTest, TestSdkDiscoveryUsesCustomPort) {
    ScopedEnv port("kmonitorPort", "5141");

    ASSERT_TRUE(param_->Init());

    EXPECT_EQ(param_->GetSinkAddress("host-from-sdk"), "host-from-sdk:5141");
    EXPECT_EQ(param_->GetSinkAddress("host-from-sdk", "10.94.0.97"), "host-from-sdk:5141");
}

TEST_F(KmonParamTest, TestHippoAddressTakesPriorityOverHostInfoIp) {
    ScopedEnv hippo_address("HIPPO_SLAVE_IP", "10.0.0.1");

    ASSERT_TRUE(param_->Init());

    EXPECT_EQ(param_->GetSinkAddress("", "10.94.0.97"), "10.0.0.1:4141");
}

TEST_F(KmonParamTest, TestHostInfoIpUsesDefaultAndCustomPort) {
    ASSERT_TRUE(param_->Init());
    EXPECT_EQ(param_->GetSinkAddress("", "10.94.0.97"), "10.94.0.97:4141");

    ScopedEnv port("kmonitorPort", "5141");
    ASSERT_TRUE(param_->Init());
    EXPECT_EQ(param_->GetSinkAddress("", "10.94.0.97"), "10.94.0.97:5141");
}

TEST_F(KmonParamTest, TestHippoAddressUsesCustomPort) {
    ScopedEnv hippo_address("HIPPO_SLAVE_IP", "10.0.0.1");
    ScopedEnv port("kmonitorPort", "5141");

    ASSERT_TRUE(param_->Init());

    EXPECT_EQ(param_->GetSinkAddress(""), "10.0.0.1:5141");
    EXPECT_EQ(param_->GetSinkAddress("host-from-sdk", "10.94.0.97"), "10.0.0.1:5141");
    EXPECT_EQ(param_->ResolveSinkAddress(), "10.0.0.1:5141");
}

TEST_F(KmonParamTest, TestHippoAddressPreservesEmptyPort) {
    ScopedEnv port("kmonitorPort", "");
    for (const auto *ip : {"10.0.0.1", "127.0.0.1"}) {
        ScopedEnv hippo_address("HIPPO_SLAVE_IP", ip);
        SCOPED_TRACE(ip);

        ASSERT_TRUE(param_->Init());

        EXPECT_TRUE(param_->kmonitor_port.empty());
        EXPECT_EQ(param_->GetSinkAddress("host-from-sdk", "10.94.0.97"), ip);
        EXPECT_EQ(param_->ResolveSinkAddress(), ip);
    }
}

TEST_F(KmonParamTest, TestEmptySinkAndPortUseSdkDefaults) {
    ScopedEnv sink_address("kmonitorSinkAddress", "");
    ScopedEnv port("kmonitorPort", "");
    ScopedEnv hippo_address("HIPPO_SLAVE_IP", "");

    ASSERT_TRUE(param_->Init());

    EXPECT_TRUE(param_->kmonitor_sink_address.empty());
    EXPECT_TRUE(param_->kmonitor_port.empty());
    EXPECT_TRUE(param_->GetSinkAddress("host-from-sdk").empty());
    EXPECT_EQ(param_->GetSinkAddress("", "10.94.0.97"), "10.94.0.97:4141");
    EXPECT_EQ(param_->GetSinkAddress(""), "127.0.0.1:4141");
}

TEST_F(KmonParamTest, TestParseHostInfoIp) {
    const std::map<std::string, std::string> cases = {
        {"host-name\n10.94.0.97\n", "10.94.0.97"},
        {"\n\nhost-name\n\n10.94.0.97\n\n", "10.94.0.97"},
        {" \t\n  host-name \r\n\t \n \t10.94.0.97 \r\n", "10.94.0.97"},
        {"host-name\nnot-an-ip\n10.94.0.97\n", ""},
        {"host-name\n999.94.0.97\n", ""},
        {"host-name\n10.94.0.97:4141\n", ""},
        {"host-name\n::1\n", ""},
        {"host-name\n", ""},
        {"", ""},
        {" \t\r\n\n", ""},
    };
    for (const auto &[content, expected] : cases) {
        SCOPED_TRACE(content);
        std::istringstream input(content);
        EXPECT_EQ(KmonParam::ParseHostInfoIp(input), expected);
    }
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
