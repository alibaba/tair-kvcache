#pragma once

#include <cstdint>
#include <string>

#include "kv_cache_manager/common/jsonizable.h"

namespace kv_cache_manager {

// 复制完成后源端 location 的处理方式，与 proto admin::MigrationRetention 对齐
enum class MigrationRetention {
    MIGRATION_RETENTION_UNSPECIFIED = 0,
    MIGRATION_RETENTION_DELETE_SOURCE = 1, // 复制完成后立即删除源端 location，尽量只保留单副本
    MIGRATION_RETENTION_KEEP_BOTH = 2,     // 保留双副本，由 Reclaimer 后续回收
};

// Copy 执行方式：通过 DataStorageBackend::Copy 把数据复制到目标 storage
class MigrationCopyMethod : public Jsonizable {
public:
    static constexpr int64_t kDefaultMaxConcurrency = 1;

    MigrationCopyMethod() = default;
    explicit MigrationCopyMethod(bool enabled, int64_t max_concurrency = kDefaultMaxConcurrency)
        : enabled_(enabled)
        , max_concurrency_(max_concurrency) {}
    ~MigrationCopyMethod() override;

    bool enabled() const { return enabled_; }
    int64_t max_concurrency() const { return max_concurrency_; }
    void set_enabled(bool enabled) { enabled_ = enabled; }
    void set_max_concurrency(int64_t max_concurrency) { max_concurrency_ = max_concurrency; }

    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

private:
    bool enabled_ = false;
    int64_t max_concurrency_ = kDefaultMaxConcurrency;
};

// Mark 执行方式：给 block 打标，下次 StartWriteCache 时引导推理引擎多写一份到目标 storage
class MigrationMarkMethod : public Jsonizable {
public:
    MigrationMarkMethod() = default;
    MigrationMarkMethod(bool enabled, int64_t mark_timeout_ms)
        : enabled_(enabled)
        , mark_timeout_ms_(mark_timeout_ms) {}
    ~MigrationMarkMethod() override;

    bool enabled() const { return enabled_; }
    int64_t mark_timeout_ms() const { return mark_timeout_ms_; }
    void set_enabled(bool enabled) { enabled_ = enabled; }
    void set_mark_timeout_ms(int64_t mark_timeout_ms) { mark_timeout_ms_ = mark_timeout_ms; }

    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

private:
    bool enabled_ = false;
    int64_t mark_timeout_ms_ = 0;
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

    // Getters
    const std::string &storage_unique_name() const { return storage_unique_name_; }
    const std::string &target_storage() const { return target_storage_; }
    double trigger_threshold() const { return trigger_threshold_; }
    const MigrationMethods &methods() const { return methods_; }
    MigrationMethods &mutable_methods() { return methods_; }
    MigrationRetention retention() const { return retention_; }

    // Setters
    void set_storage_unique_name(const std::string &storage_unique_name) {
        storage_unique_name_ = storage_unique_name;
    }
    void set_target_storage(const std::string &target_storage) { target_storage_ = target_storage; }
    void set_trigger_threshold(double trigger_threshold) { trigger_threshold_ = trigger_threshold; }
    void set_methods(const MigrationMethods &methods) { methods_ = methods; }
    void set_retention(MigrationRetention retention) { retention_ = retention; }

    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;
    bool ValidateRequiredFields(std::string &invalid_fields) const;

private:
    std::string storage_unique_name_;                                            // 源 storage（热层）
    std::string target_storage_;                                                 // 目标 storage（冷层）
    double trigger_threshold_ = 0.0;                                             // 水位触发阈值（迁移区间下界）
    MigrationMethods methods_;                                                   // Copy / Mark 执行方式
    MigrationRetention retention_ = MigrationRetention::MIGRATION_RETENTION_UNSPECIFIED; // 源端保留策略
};

} // namespace kv_cache_manager
