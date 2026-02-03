#include <future>
#include <thread>

#include "kv_cache_manager/common/env_util.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/unittest.h"

using namespace kv_cache_manager;

class LoggerTest : public TESTBASE {
public:
    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(LoggerTest, TestSimple) {
    {
        LoggerBroker::InitLogger("");
        KVCM_LOG_DEBUG("test debug");
        KVCM_LOG_INFO("test info");
        KVCM_LOG_WARN("test warn");
        KVCM_LOG_ERROR("test error");
        LoggerBroker::DestroyLogger();
    }

    {
        ScopedEnv env("KVCM_LOG_TO_CONSOLE", "1");
        LoggerBroker::InitLogger("");
        KVCM_LOG_DEBUG("test debug");
        KVCM_LOG_INFO("test info");
        KVCM_LOG_WARN("test warn");
        KVCM_LOG_ERROR("test error");
        LoggerBroker::DestroyLogger();
    }

    {
        LoggerBroker::InitLogger("");
        KVCM_INTERVAL_LOG_DEBUG(2, "test interval debug1");
        KVCM_INTERVAL_LOG_DEBUG(2, "test interval debug2");
        KVCM_INTERVAL_LOG_DEBUG(2, "test interval debug3");
        KVCM_INTERVAL_LOG_INFO(1, "test interval info");
        KVCM_INTERVAL_LOG_WARN(1, "test interval warn");
        KVCM_INTERVAL_LOG_ERROR(1, "test interval error");
        LoggerBroker::DestroyLogger();
    }

    {
        LoggerBroker::InitLogger("");
        KVCM_PERIOD_LOG_DEBUG(2, "test period debug1");
        KVCM_PERIOD_LOG_DEBUG(2, "test period debug2");
        KVCM_PERIOD_LOG_INFO(1, "test period info");
        KVCM_PERIOD_LOG_WARN(1, "test period warn");
        KVCM_PERIOD_LOG_ERROR(1, "test period error");
        LoggerBroker::DestroyLogger();
    }
}

TEST_F(LoggerTest, TestSetLogLevel) {
    {
        LoggerBroker::InitLogger("");
        KVCM_LOG_DEBUG("test debug");
        KVCM_LOG_INFO("test info");
        KVCM_LOG_WARN("test warn");
        KVCM_LOG_ERROR("test error");
        LoggerBroker::SetLogLevel(Logger::LEVEL_DEBUG);
        KVCM_LOG_DEBUG("test debug");
        KVCM_LOG_INFO("test info");
        KVCM_LOG_WARN("test warn");
        KVCM_LOG_ERROR("test error");
        ASSERT_EQ(Logger::LEVEL_DEBUG, LoggerBroker::base_log_level_);
        LoggerBroker::DestroyLogger();
    }

    {
        ScopedEnv env("KVCM_LOG_LEVEL", "WARN");
        LoggerBroker::InitLogger("");
        ASSERT_EQ(Logger::LEVEL_WARN, LoggerBroker::base_log_level_);
        KVCM_LOG_DEBUG("test debug");
        KVCM_LOG_INFO("test info");
        KVCM_LOG_WARN("test warn");
        KVCM_LOG_ERROR("test error");
        LoggerBroker::SetLogLevel(Logger::LEVEL_DEBUG);
        KVCM_LOG_DEBUG("test debug");
        KVCM_LOG_INFO("test info");
        KVCM_LOG_WARN("test warn");
        KVCM_LOG_ERROR("test error");
        ASSERT_EQ(Logger::LEVEL_DEBUG, LoggerBroker::base_log_level_);
        LoggerBroker::SetLogLevel(9999);
        ASSERT_EQ(Logger::LEVEL_DEBUG, LoggerBroker::base_log_level_);
        KVCM_LOG_DEBUG("test debug");
        KVCM_LOG_INFO("test info");
        KVCM_LOG_WARN("test warn");
        KVCM_LOG_ERROR("test error");
        LoggerBroker::DestroyLogger();
    }

    {
        ScopedEnv env("KVCM_LOG_LEVEL", "INVALID_LEVEL");
        LoggerBroker::InitLogger("");
        ASSERT_EQ(Logger::LEVEL_UNSET, LoggerBroker::base_log_level_);
        KVCM_LOG_DEBUG("test debug");
        KVCM_LOG_INFO("test info");
        KVCM_LOG_WARN("test warn");
        KVCM_LOG_ERROR("test error");
        LoggerBroker::SetLogLevel(9999);
        ASSERT_EQ(Logger::LEVEL_UNSET, LoggerBroker::base_log_level_);
        KVCM_LOG_DEBUG("test debug");
        KVCM_LOG_INFO("test info");
        KVCM_LOG_WARN("test warn");
        KVCM_LOG_ERROR("test error");
        LoggerBroker::DestroyLogger();
    }

    {
        ScopedEnv env("KVCM_LOG_LEVEL", "100");
        LoggerBroker::InitLogger("");
        ASSERT_EQ(Logger::LEVEL_UNSET, LoggerBroker::base_log_level_);
        KVCM_LOG_DEBUG("test debug");
        KVCM_LOG_INFO("test info");
        KVCM_LOG_WARN("test warn");
        KVCM_LOG_ERROR("test error");
        LoggerBroker::DestroyLogger();
    }
}

TEST_F(LoggerTest, TestClientLogger) {
    {
        LoggerBroker::InitLoggerForClient();
        ASSERT_EQ(Logger::LEVEL_UNSET, LoggerBroker::base_log_level_);
        KVCM_LOG_DEBUG("test debug");
        KVCM_LOG_INFO("test info");
        KVCM_LOG_WARN("test warn");
        KVCM_LOG_ERROR("test error");
        LoggerBroker::DestroyLogger();
    }

    {
        ScopedEnv env("KVCM_LOG_LEVEL", "DEBUG");
        LoggerBroker::InitLoggerForClient();
        ASSERT_EQ(Logger::LEVEL_DEBUG, LoggerBroker::base_log_level_);
        KVCM_LOG_DEBUG("test debug");
        KVCM_LOG_INFO("test info");
        KVCM_LOG_WARN("test warn");
        KVCM_LOG_ERROR("test error");
        LoggerBroker::SetLogLevel(Logger::LEVEL_INFO);
        ASSERT_EQ(Logger::LEVEL_INFO, LoggerBroker::base_log_level_);
        KVCM_LOG_DEBUG("test debug");
        KVCM_LOG_INFO("test info");
        KVCM_LOG_WARN("test warn");
        KVCM_LOG_ERROR("test error");
        LoggerBroker::DestroyLogger();
    }

    {
        ScopedEnv env("KVCM_LOG_LEVEL", "ERROR");
        LoggerBroker::InitLoggerForClient();
        ASSERT_EQ(Logger::LEVEL_ERROR, LoggerBroker::base_log_level_);
        KVCM_LOG_DEBUG("test debug");
        KVCM_LOG_INFO("test info");
        KVCM_LOG_WARN("test warn");
        KVCM_LOG_ERROR("test error");
        LoggerBroker::DestroyLogger();
    }

    {
        ScopedEnv env("KVCM_LOG_LEVEL", "INVALID_LEVEL");
        LoggerBroker::InitLoggerForClient();
        ASSERT_EQ(Logger::LEVEL_UNSET, LoggerBroker::base_log_level_);
        KVCM_LOG_DEBUG("test debug");
        KVCM_LOG_INFO("test info");
        KVCM_LOG_WARN("test warn");
        KVCM_LOG_ERROR("test error");
        LoggerBroker::DestroyLogger();
    }

    {
        ScopedEnv env("KVCM_LOG_LEVEL", "100");
        LoggerBroker::InitLoggerForClient();
        ASSERT_EQ(Logger::LEVEL_UNSET, LoggerBroker::base_log_level_);
        KVCM_LOG_DEBUG("test debug");
        KVCM_LOG_INFO("test info");
        KVCM_LOG_WARN("test warn");
        KVCM_LOG_ERROR("test error");
        LoggerBroker::DestroyLogger();
    }
}

TEST_F(LoggerTest, TestInitLoggerForClientOnce) {

    ScopedEnv env("KVCM_LOG_LEVEL", "DEBUG");
    LoggerBroker::InitLoggerForClientOnce();
    constexpr int kThreadCount = 10; // 测试线程数
    constexpr int kIterations = 100; // 每个线程调用次数
    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    // 创建多个线程
    for (int i = 0; i < kThreadCount; ++i) {
        threads.emplace_back([&]() {
            // 等待所有线程准备就绪
            ready++;
            while (!go) {
                std::this_thread::yield();
            }
            // 每个线程多次调用初始化
            for (int j = 0; j < kIterations; ++j) {
                LoggerBroker::InitLoggerForClientOnce();
                if (LoggerBroker::logger_) {
                    success_count++;
                }
            }
            KVCM_LOG_DEBUG("test debug");
        });
    }
    // 等待所有线程准备就绪
    while (ready < kThreadCount) {
        std::this_thread::yield();
    }
    go = true; // 同时启动所有线程
    // 等待所有线程完成
    for (auto &t : threads) {
        t.join();
    }
    // 验证结果
    ASSERT_GT(success_count, 0);
    ASSERT_TRUE(LoggerBroker::logger_);
    ASSERT_EQ(Logger::LEVEL_DEBUG, LoggerBroker::base_log_level_);
    LoggerBroker::DestroyLogger();
}

TEST_F(LoggerTest, TestUpdateLogLevel) {
    LoggerBroker::InitLogger("");
    LoggerBroker::SetLogLevel(Logger::LEVEL_DEBUG);
    std::atomic<bool> stop{false};
    std::future f = std::async(std::launch::async, [&]() {
        while (!stop) {
            KVCM_LOG_DEBUG("test debug, wait log level update");
        }
    });
    ASSERT_EQ(Logger::LEVEL_DEBUG, LoggerBroker::base_log_level_);
    usleep(1000);
    LoggerBroker::SetLogLevel(Logger::LEVEL_INFO);
    KVCM_LOG_DEBUG("test debug, logger info update");
    KVCM_LOG_INFO("test info, logger info update");
    stop = true;
    ASSERT_EQ(Logger::LEVEL_INFO, LoggerBroker::base_log_level_);
}