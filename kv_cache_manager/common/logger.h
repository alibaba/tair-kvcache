#pragma once

#include <cstdarg>
#include <memory>
#include <mutex>
#include <string>

#ifndef KVCM_DECLARE_AND_SETUP_LOGGER
// TODO
#define KVCM_DECLARE_AND_SETUP_LOGGER(namespace, class) ;
#endif

#define KVCM_LOG_DEBUG(format, ...) KVCM_LOG(kv_cache_manager::Logger::LEVEL_DEBUG, format, ##__VA_ARGS__)
#define KVCM_LOG_INFO(format, ...) KVCM_LOG(kv_cache_manager::Logger::LEVEL_INFO, format, ##__VA_ARGS__)
#define KVCM_LOG_WARN(format, ...) KVCM_LOG(kv_cache_manager::Logger::LEVEL_WARN, format, ##__VA_ARGS__)
#define KVCM_LOG_ERROR(format, ...) KVCM_LOG(kv_cache_manager::Logger::LEVEL_ERROR, format, ##__VA_ARGS__)
#define KVCM_LOG(level, format, ...)                                                                                   \
    do {                                                                                                               \
        if (kv_cache_manager::LoggerBroker::IsLevelEnable(level)) {                                                    \
            kv_cache_manager::LoggerBroker::Log(level, __FILE__, __LINE__, __func__, format, ##__VA_ARGS__);           \
        }                                                                                                              \
    } while (0)

#define KVCM_INTERVAL_LOG_DEBUG(interval, format, ...)                                                                 \
    KVCM_INTERVAL_LOG(interval, kv_cache_manager::Logger::LEVEL_DEBUG, format, ##__VA_ARGS__)
#define KVCM_INTERVAL_LOG_INFO(interval, format, ...)                                                                  \
    KVCM_INTERVAL_LOG(interval, kv_cache_manager::Logger::LEVEL_INFO, format, ##__VA_ARGS__)
#define KVCM_INTERVAL_LOG_WARN(interval, format, ...)                                                                  \
    KVCM_INTERVAL_LOG(interval, kv_cache_manager::Logger::LEVEL_WARN, format, ##__VA_ARGS__)
#define KVCM_INTERVAL_LOG_ERROR(interval, format, ...)                                                                 \
    KVCM_INTERVAL_LOG(interval, kv_cache_manager::Logger::LEVEL_ERROR, format, ##__VA_ARGS__)

#define KVCM_PERIOD_LOG_DEBUG(period_s, format, ...)                                                                   \
    KVCM_PERIOD_LOG(period_s, kv_cache_manager::Logger::LEVEL_DEBUG, format, ##__VA_ARGS__)
#define KVCM_PERIOD_LOG_INFO(period_s, format, ...)                                                                    \
    KVCM_PERIOD_LOG(period_s, kv_cache_manager::Logger::LEVEL_INFO, format, ##__VA_ARGS__)
#define KVCM_PERIOD_LOG_WARN(period_s, format, ...)                                                                    \
    KVCM_PERIOD_LOG(period_s, kv_cache_manager::Logger::LEVEL_WARN, format, ##__VA_ARGS__)
#define KVCM_PERIOD_LOG_ERROR(period_s, format, ...)                                                                   \
    KVCM_PERIOD_LOG(period_s, kv_cache_manager::Logger::LEVEL_ERROR, format, ##__VA_ARGS__)

#define KVCM_INTERVAL_LOG(interval, level, format, ...)                                                                \
    do {                                                                                                               \
        if (kv_cache_manager::LoggerBroker::IsLevelEnable(level)) {                                                    \
            static int log_counter = 0;                                                                                \
            if (log_counter <= 0) {                                                                                    \
                kv_cache_manager::LoggerBroker::Log(level, __FILE__, __LINE__, __func__, format, ##__VA_ARGS__);       \
                log_counter = interval;                                                                                \
            }                                                                                                          \
            log_counter--;                                                                                             \
        }                                                                                                              \
    } while (0)

#define KVCM_PERIOD_LOG(period_s, level, format, ...)                                                                  \
    do {                                                                                                               \
        if (kv_cache_manager::LoggerBroker::IsLevelEnable(level)) {                                                    \
            static int32_t last_log_timestamp_s = 0;                                                                   \
            int32_t now = std::time(nullptr);                                                                          \
            if (now - last_log_timestamp_s > period_s) {                                                               \
                kv_cache_manager::LoggerBroker::Log(level, __FILE__, __LINE__, __func__, format, ##__VA_ARGS__);       \
                last_log_timestamp_s = now;                                                                            \
            }                                                                                                          \
        }                                                                                                              \
    } while (0)

#define KVCM_ACCESS_LOG(msg) kv_cache_manager::LoggerBroker::AccessLog(kv_cache_manager::Logger::LEVEL_INFO, msg)

#define KVCM_METRICS_LOG(msg) kv_cache_manager::LoggerBroker::MetricsLog(kv_cache_manager::Logger::LEVEL_INFO, msg)

#define KVCM_PUBLISHER_LOG(msg) kv_cache_manager::LoggerBroker::PublisherLog(kv_cache_manager::Logger::LEVEL_INFO, msg)

namespace kv_cache_manager {

class Logger {
public:
    static constexpr uint32_t LEVEL_UNSET = 0;
    // static constexpr uint32_t LEVEL_FATAL = 1;
    static constexpr uint32_t LEVEL_ERROR = 2;
    static constexpr uint32_t LEVEL_WARN = 3;
    static constexpr uint32_t LEVEL_INFO = 4;
    static constexpr uint32_t LEVEL_DEBUG = 5;

public:
    explicit Logger(void *real_logger);
    ~Logger();
    void SetLogLevel(uint32_t level);
    void Log(uint32_t level, const char *file, int line, const char *func, const char *format, va_list args);
    void Log(uint32_t level, const std::string &msg);

    static const char *LevelToString(uint32_t level);
    static const uint32_t StringToLevel(const std::string &level_str);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class LoggerBroker {
public:
    LoggerBroker() = delete;
    static bool InitLogger(const std::string &log_config_file);
    static bool InitLoggerForClient();
    static void InitLoggerForClientOnce();
    static void DestroyLogger();
    static void SetLogLevel(uint32_t level);
    static void Log(int level, const char *file, int line, const char *func, const char *format, ...)
        __attribute__((format(printf, 5, 6)));
    static void AccessLog(int level, const std::string &msg);
    static void MetricsLog(int level, const std::string &msg);
    static void PublisherLog(int level, const std::string &msg);
    inline static bool IsLevelEnable(uint32_t level) { return (level <= base_log_level_) ? true : false; }

private:
    static void InitLogLevelFromEnv();

    static uint32_t base_log_level_;
    static std::recursive_mutex logger_mutex_;
    static std::unique_ptr<Logger> logger_;
    static std::unique_ptr<Logger> access_logger_;
    static std::unique_ptr<Logger> metrics_logger_;
    static std::unique_ptr<Logger> publisher_logger_;
};

} // namespace kv_cache_manager
