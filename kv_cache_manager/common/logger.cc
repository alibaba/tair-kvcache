#include "kv_cache_manager/common/logger.h"

#include <algorithm>
#include <iostream>
#include <sstream>

#include "alog/Appender.h"
#include "alog/Configurator.h"
#include "alog/Logger.h"
#include "kv_cache_manager/common/env_util.h"

namespace kv_cache_manager {

static const std::string DEFAULT_ALOG_CONF = R"(
alog.rootLogger=INFO, rootAppender
alog.max_msg_len=204800

alog.appender.rootAppender=FileAppender
alog.appender.rootAppender.fileName=logs/kv_cache_manager.log
alog.appender.rootAppender.layout=PatternLayout
alog.appender.rootAppender.layout.LogPattern=[%%d] [%%l] [%%p:%%t] %%m
alog.appender.rootAppender.async_flush=false
alog.appender.rootAppender.flush=true
alog.appender.rootAppender.compress=false

alog.logger.access=INFO, accessAppender
inherit.access=false
alog.appender.accessAppender=FileAppender
alog.appender.accessAppender.fileName=logs/access.log
alog.appender.accessAppender.layout=PatternLayout
alog.appender.accessAppender.layout.LogPattern=%%m
alog.appender.accessAppender.async_flush=true
alog.appender.accessAppender.flush_threshold=100
alog.appender.accessAppender.flush_interval=100
alog.appender.accessAppender.compress=false
alog.appender.accessAppender.max_file_size=256
alog.appender.accessAppender.log_keep_count=20

alog.logger.metrics=INFO, metricsAppender
inherit.metrics=false
alog.appender.metricsAppender=FileAppender
alog.appender.metricsAppender.fileName=logs/metrics.log
alog.appender.metricsAppender.layout=PatternLayout
alog.appender.metricsAppender.layout.LogPattern=%%m
alog.appender.metricsAppender.async_flush=false
alog.appender.metricsAppender.flush_threshold=10240
alog.appender.metricsAppender.flush_interval=100
alog.appender.metricsAppender.compress=false
alog.appender.metricsAppender.max_file_size=512
alog.appender.metricsAppender.log_keep_count=10

alog.logger.publisher=INFO, publisherAppender
inherit.publisher=false
alog.appender.publisherAppender=FileAppender
alog.appender.publisherAppender.fileName=logs/event_publisher.log
alog.appender.publisherAppender.layout=PatternLayout
alog.appender.publisherAppender.layout.LogPattern=%%m
alog.appender.publisherAppender.async_flush=true
alog.appender.publisherAppender.flush_threshold=10240
alog.appender.publisherAppender.flush_interval=100
alog.appender.publisherAppender.compress=false
alog.appender.publisherAppender.max_file_size=256
alog.appender.publisherAppender.log_keep_count=20

alog.logger.console=INFO, consoleAppender
inherit.console=false
alog.appender.consoleAppender=ConsoleAppender
alog.appender.consoleAppender.layout=PatternLayout
alog.appender.consoleAppender.layout.LogPattern=[%%d] [%%l] [%%p:%%t] %%m
)";

uint32_t LoggerBroker::base_log_level_ = 0;
std::recursive_mutex LoggerBroker::logger_mutex_;
std::unique_ptr<Logger> LoggerBroker::logger_;
std::unique_ptr<Logger> LoggerBroker::access_logger_;
std::unique_ptr<Logger> LoggerBroker::metrics_logger_;
std::unique_ptr<Logger> LoggerBroker::publisher_logger_;

bool LoggerBroker::InitLogger(const std::string &log_config_file) {
    std::lock_guard<std::recursive_mutex> logger_guard(logger_mutex_);
    if (logger_) {
        std::cerr << "Logger already inited, you should call DestroyLogger() first." << std::endl;
        return false;
    }
    try {
        if (log_config_file.empty()) {
            alog::Configurator::configureLoggerFromString(DEFAULT_ALOG_CONF.c_str());
        } else {
            alog::Configurator::configureLogger(log_config_file.c_str());
        }
    } catch (std::exception &e) {
        std::cerr << "Failed to configure logger, log_config_file: " << log_config_file << std::endl;
        return false;
    }

    // 初始化logger
    if (EnvUtil::GetEnv("KVCM_LOG_TO_CONSOLE", 0) == 1) {
        auto alog_console_logger = alog::Logger::getLogger("console");
        logger_.reset(new Logger(alog_console_logger));
        access_logger_.reset(new Logger(alog_console_logger));
        metrics_logger_.reset(new Logger(alog_console_logger));
        publisher_logger_.reset(new Logger(alog_console_logger));
    } else {
        auto alog_logger = alog::Logger::getLogger("rootLogger");
        auto alog_access_logger = alog::Logger::getLogger("access");
        auto alog_metrics_logger = alog::Logger::getLogger("metrics");
        auto alog_publisher_logger = alog::Logger::getLogger("publisher");
        logger_.reset(new Logger(alog_logger));
        access_logger_.reset(new Logger(alog_access_logger));
        metrics_logger_.reset(new Logger(alog_metrics_logger));
        publisher_logger_.reset(new Logger(alog_publisher_logger));
    }
    InitLogLevelFromEnv();
    return true;
}

bool LoggerBroker::InitLoggerForClient() {
    std::lock_guard<std::recursive_mutex> logger_guard(logger_mutex_);
    if (logger_) {
        std::cerr << "kv_cache_manager_client logger already inited." << std::endl;
        return true;
    }
    auto alog_logger = alog::Logger::getLogger("kv_cache_manager_client");
    if (alog_logger == nullptr) {
        // can not reach here
        std::cerr << "Failed to get kv_cache_manager_client logger." << std::endl;
        return false;
    }
    static std::string CLIENT_APPENDER_NAME = "logs/kv_cache_manager_client.log";
    auto fileAppender = (alog::FileAppender *)alog::FileAppender::getAppender(CLIENT_APPENDER_NAME.c_str());
    fileAppender->setMaxSize(100);
    fileAppender->setCacheLimit(1024);
    fileAppender->setHistoryLogKeepCount(5);
    fileAppender->setAsyncFlush(false);
    fileAppender->setCompress(true);
    auto layout = new alog::PatternLayout();
    layout->setLogPattern("[%%d] [%%l] [%%p:%%t] %%m");
    fileAppender->setLayout(layout);
    alog_logger->setAppender(fileAppender);
    alog_logger->setInheritFlag(false);
    alog_logger->setLevel(alog::LOG_LEVEL_INFO);
    logger_.reset(new Logger(alog_logger));
    InitLogLevelFromEnv();
    return true;
}

void LoggerBroker::InitLoggerForClientOnce() {
    static std::once_flag once_flag;
    std::call_once(once_flag, []() {
        std::cout << "init kv cache manager client logger." << std::endl;
        if (!LoggerBroker::InitLoggerForClient()) {
            std::cerr << "Init kv cache manager client logger failed." << std::endl;
        }
    });
}

void LoggerBroker::InitLogLevelFromEnv() {
    auto env_log_level_str = EnvUtil::GetEnv("KVCM_LOG_LEVEL", "");
    if (env_log_level_str.empty()) {
        return;
    }
    uint32_t level = Logger::StringToLevel(env_log_level_str);
    if (level != Logger::LEVEL_UNSET) {
        SetLogLevel(level);
    } else {
        std::cerr << "Invalid kvcm log level env [" << env_log_level_str << "], ignored." << std::endl;
    }
}

void LoggerBroker::DestroyLogger() {
    std::lock_guard<std::recursive_mutex> logger_guard(logger_mutex_);
    logger_.reset();
    access_logger_.reset();
    metrics_logger_.reset();
    publisher_logger_.reset();
    base_log_level_ = 0;
    alog::Logger::shutdown();
}

void LoggerBroker::SetLogLevel(uint32_t level) {
    if (level < Logger::LEVEL_ERROR || level > Logger::LEVEL_DEBUG) {
        std::cerr << "Invalid log level: " << level << std::endl;
        return;
    }
    std::cout << "kv_cache_manager set log level: " << Logger::LevelToString(level) << std::endl;
    std::lock_guard<std::recursive_mutex> logger_guard(logger_mutex_);
    base_log_level_ = level;
    if (logger_) {
        logger_->SetLogLevel(level);
    }
    if (access_logger_) {
        access_logger_->SetLogLevel(level);
    }
    if (metrics_logger_) {
        metrics_logger_->SetLogLevel(level);
    }
    if (publisher_logger_) {
        publisher_logger_->SetLogLevel(level);
    }
}

#define CHECK_LOGGER_INITED()                                                                                          \
    if (!logger_) {                                                                                                    \
        std::lock_guard<std::recursive_mutex> logger_guard(logger_mutex_);                                             \
        if (!logger_) {                                                                                                \
            std::stringstream ss;                                                                                      \
            ss << "kvcm logger not inited, try print to stderr:";                                                      \
            ss << "file=" << file << ", line=" << line;                                                                \
            ss << ", func=" << func << ", format=" << format;                                                          \
            std::cerr << ss.str() << std::endl;                                                                        \
            return;                                                                                                    \
        }                                                                                                              \
    }

#define CHECK_LOGGER_INITED_2()                                                                                        \
    if (!logger_) {                                                                                                    \
        std::lock_guard<std::recursive_mutex> logger_guard(logger_mutex_);                                             \
        if (!logger_) {                                                                                                \
            std::stringstream ss;                                                                                      \
            ss << "kvcm logger not inited, try print to stderr:";                                                      \
            ss << "msg=" << msg;                                                                                       \
            std::cerr << ss.str() << std::endl;                                                                        \
            return;                                                                                                    \
        }                                                                                                              \
    }

void LoggerBroker::Log(int level, const char *file, int line, const char *func, const char *format, ...) {
    CHECK_LOGGER_INITED();
    va_list args;
    va_start(args, format);
    logger_->Log(level, file, line, func, format, args);
    va_end(args);
}

void LoggerBroker::AccessLog(int level, const std::string &msg) {
    CHECK_LOGGER_INITED_2();
    access_logger_->Log(level, msg);
}

void LoggerBroker::MetricsLog(int level, const std::string &msg) {
    CHECK_LOGGER_INITED_2();
    metrics_logger_->Log(level, msg);
}

void LoggerBroker::PublisherLog(int level, const std::string &msg) {
    CHECK_LOGGER_INITED_2();
    publisher_logger_->Log(level, msg);
}

#undef CHECK_LOGGER_INITED
#undef CHECK_LOGGER_INITED_2

struct Logger::Impl {
    alog::Logger *logger_ = nullptr;
};

Logger::Logger(void *real_logger) : impl_(new Logger::Impl) { impl_->logger_ = (alog::Logger *)(real_logger); }

Logger::~Logger() {}

void Logger::SetLogLevel(uint32_t level) { impl_->logger_->setLevel((uint32_t)level); }

void Logger::Log(uint32_t level, const char *file, int line, const char *func, const char *format, va_list args) {
    va_list args_copy;
    va_copy(args_copy, args);
    int needed = std::vsnprintf(nullptr, 0, format, args_copy);
    va_end(args_copy);
    std::string format_str;
    format_str.resize(static_cast<size_t>(needed));
    va_copy(args_copy, args);
    std::vsnprintf(&format_str[0], static_cast<size_t>(needed) + 1, format, args_copy);
    va_end(args_copy);
    impl_->logger_->log(level, "%s:%d %s() - %s", file, line, func, format_str.c_str());
    if (level >= impl_->logger_->getLevel() || level <= alog::LOG_LEVEL_ERROR) {
        impl_->logger_->flush();
    }
}

void Logger::Log(uint32_t level, const std::string &msg) {
    impl_->logger_->logBinaryMessage(level, msg);
    if (impl_->logger_->getLevel() <= alog::LOG_LEVEL_ERROR) {
        impl_->logger_->flush();
    }
}

const char *Logger::LevelToString(uint32_t level) {
    switch (level) {
    case LEVEL_DEBUG:
        return "DEBUG";
    case LEVEL_INFO:
        return "INFO";
    case LEVEL_WARN:
        return "WARN";
    case LEVEL_ERROR:
        return "ERROR";
    default:
        return "UNKNOWN";
    }
}

const uint32_t Logger::StringToLevel(const std::string &level_str) {
    std::string upper_level_str = level_str;
    auto to_upper = [](unsigned char c) { return std::toupper(c); };
    std::transform(level_str.begin(), level_str.end(), upper_level_str.begin(), to_upper);
    if (upper_level_str == "DEBUG") {
        return LEVEL_DEBUG;
    } else if (upper_level_str == "INFO") {
        return LEVEL_INFO;
    } else if (upper_level_str == "WARN") {
        return LEVEL_WARN;
    } else if (upper_level_str == "ERROR") {
        return LEVEL_ERROR;
    }
    std::cerr << "Invalid log level string: " << level_str << std::endl;
    return LEVEL_UNSET;
}

} // namespace kv_cache_manager
