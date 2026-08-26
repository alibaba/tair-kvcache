#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/common/jsonizable.h"

namespace kv_cache_manager {

// 复制完成后源端 location 的处理方式，与 proto admin::MigrationRetention 对齐
enum class MigrationRetention {
    MIGRATION_RETENTION_UNSPECIFIED = 0,
    MIGRATION_RETENTION_DELETE_SOURCE = 1, // 复制完成后立即删除源端 location，尽量只保留单副本
    MIGRATION_RETENTION_KEEP_BOTH = 2,     // 保留双副本，由 Reclaimer 后续回收
};

// Mark 被 StartWriteCache 消费后的清理策略。
enum class MigrationMarkClearPolicy {
    CLEAR_ON_NEXT_WRITE_SUCCESS = 0, // 本次 target CacheLocation 成功 SERVING 后清标
    CLEAR_ON_FULL_BLOCK_COVERED = 1, // target CacheLocation 覆盖完整 block 后清标
};

enum class MigrationCopyExecutionMode {
    SYNC = 0,
    ASYNC_REQUIRED = 1,
};

// Copy 执行方式：通过 DataStorageBackend::Copy 把数据复制到目标 storage。
class MigrationCopyMethod : public Jsonizable {
public:
    MigrationCopyMethod() = default;
    explicit MigrationCopyMethod(bool enabled)
        : enabled_(enabled) {}
    ~MigrationCopyMethod() override;

    bool enabled() const { return enabled_; }
    void set_enabled(bool enabled) { enabled_ = enabled; }

    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

private:
    bool enabled_ = false;
};

// Mark 执行方式：给 block 打标，下次 StartWriteCache 时引导推理引擎多写一份到目标 storage
class MigrationMarkMethod : public Jsonizable {
public:
    static constexpr int64_t kDefaultTimeoutMs = 24 * 60 * 60 * 1000;

    MigrationMarkMethod() = default;
    explicit MigrationMarkMethod(bool enabled)
        : enabled_(enabled) {}
    ~MigrationMarkMethod() override;

    bool enabled() const { return enabled_; }
    int64_t timeout_ms() const { return timeout_ms_; }
    void set_enabled(bool enabled) { enabled_ = enabled; }
    void set_timeout_ms(int64_t timeout_ms) { timeout_ms_ = timeout_ms; }

    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

private:
    bool enabled_ = false;
    int64_t timeout_ms_ = kDefaultTimeoutMs;
};

// 执行方式集合，Copy / Mark 可同时开启（与触发模式正交）
class MigrationMethods : public Jsonizable {
public:
    MigrationMethods() = default;
    ~MigrationMethods() override;

    const MigrationCopyMethod &copy() const { return copy_; }
    const MigrationMarkMethod &mark() const { return mark_; }
    MigrationCopyMethod &mutable_copy() { return copy_; }
    MigrationMarkMethod &mutable_mark() { return mark_; }
    void set_copy(const MigrationCopyMethod &copy) { copy_ = copy; }
    void set_mark(const MigrationMarkMethod &mark) { mark_ = mark; }

    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

private:
    MigrationCopyMethod copy_;
    MigrationMarkMethod mark_;
};

/**
 * 单条多层存储迁移规则，绑定一个源 storage（热层），描述何时、以何种方式迁移到目标 storage（冷层）。
 * 与 proto admin::MigrationStrategy 一一对应。
 */
class MigrationStrategy : public Jsonizable {
public:
    MigrationStrategy() = default;
    ~MigrationStrategy() override;

    // Source/target getters use symmetric names.
    const std::string &source_storage_name() const { return source_storage_name_; }
    const std::string &target_storage_name() const { return target_storage_name_; }
    double trigger_threshold() const { return trigger_threshold_; }
    const MigrationMethods &methods() const { return methods_; }
    MigrationMethods &mutable_methods() { return methods_; }
    MigrationRetention retention() const { return retention_; }

    // Setters
    void set_source_storage_name(const std::string &name) { source_storage_name_ = name; }
    void set_target_storage_name(const std::string &name) { target_storage_name_ = name; }
    void set_trigger_threshold(double trigger_threshold) { trigger_threshold_ = trigger_threshold; }
    void set_methods(const MigrationMethods &methods) { methods_ = methods; }
    void set_retention(MigrationRetention retention) { retention_ = retention; }

    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;
    bool ValidateRequiredFields(std::string &invalid_fields) const;

private:
    std::string source_storage_name_;
    std::string target_storage_name_;
    double trigger_threshold_ = 0.0;
    MigrationMethods methods_;
    MigrationRetention retention_ = MigrationRetention::MIGRATION_RETENTION_UNSPECIFIED;
};

class MigrationConfig : public Jsonizable {
public:
    static constexpr int64_t kDefaultCopyMaxConcurrency = 1;
    static constexpr int64_t kDefaultCopyOperationDeadlineMs = 10 * 60 * 1000;
    static constexpr int64_t kDefaultCopyPollInitialIntervalMs = 20;
    static constexpr int64_t kDefaultCopyPollMaxIntervalMs = 1000;

    MigrationConfig() = default;
    ~MigrationConfig() override;

    // A source/target pair identifies one migration route and must be unique within this config.
    const std::vector<std::shared_ptr<MigrationStrategy>> &strategies() const { return strategies_; }
    int64_t copy_max_concurrency() const { return copy_max_concurrency_; }
    MigrationMarkClearPolicy mark_clear_policy() const { return mark_clear_policy_; }
    MigrationCopyExecutionMode copy_execution_mode() const { return copy_execution_mode_; }
    uint64_t copy_max_inflight_bytes() const { return copy_max_inflight_bytes_; }
    int64_t copy_max_quarantine_operations() const { return copy_max_quarantine_operations_; }
    uint64_t copy_max_quarantine_bytes() const { return copy_max_quarantine_bytes_; }
    int64_t copy_operation_deadline_ms() const { return copy_operation_deadline_ms_; }
    int64_t copy_poll_initial_interval_ms() const { return copy_poll_initial_interval_ms_; }
    int64_t copy_poll_max_interval_ms() const { return copy_poll_max_interval_ms_; }

    void set_strategies(const std::vector<std::shared_ptr<MigrationStrategy>> &strategies) {
        strategies_ = strategies;
    }
    void set_copy_max_concurrency(int64_t copy_max_concurrency) {
        copy_max_concurrency_ = copy_max_concurrency;
    }
    void set_mark_clear_policy(MigrationMarkClearPolicy mark_clear_policy) {
        mark_clear_policy_ = mark_clear_policy;
    }
    void set_copy_execution_mode(MigrationCopyExecutionMode value) { copy_execution_mode_ = value; }
    void set_copy_max_inflight_bytes(uint64_t value) { copy_max_inflight_bytes_ = value; }
    void set_copy_max_quarantine_operations(int64_t value) { copy_max_quarantine_operations_ = value; }
    void set_copy_max_quarantine_bytes(uint64_t value) { copy_max_quarantine_bytes_ = value; }
    void set_copy_operation_deadline_ms(int64_t value) { copy_operation_deadline_ms_ = value; }
    void set_copy_poll_initial_interval_ms(int64_t value) { copy_poll_initial_interval_ms_ = value; }
    void set_copy_poll_max_interval_ms(int64_t value) { copy_poll_max_interval_ms_ = value; }

    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;
    bool ValidateRequiredFields(std::string &invalid_fields) const;

private:
    std::vector<std::shared_ptr<MigrationStrategy>> strategies_;
    int64_t copy_max_concurrency_ = kDefaultCopyMaxConcurrency;
    MigrationMarkClearPolicy mark_clear_policy_ = MigrationMarkClearPolicy::CLEAR_ON_NEXT_WRITE_SUCCESS;
    MigrationCopyExecutionMode copy_execution_mode_ = MigrationCopyExecutionMode::SYNC;
    uint64_t copy_max_inflight_bytes_ = 0;
    int64_t copy_max_quarantine_operations_ = 0;
    uint64_t copy_max_quarantine_bytes_ = 0;
    int64_t copy_operation_deadline_ms_ = kDefaultCopyOperationDeadlineMs;
    int64_t copy_poll_initial_interval_ms_ = kDefaultCopyPollInitialIntervalMs;
    int64_t copy_poll_max_interval_ms_ = kDefaultCopyPollMaxIntervalMs;
};

} // namespace kv_cache_manager
