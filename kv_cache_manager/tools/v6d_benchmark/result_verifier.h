#pragma once

#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {
namespace v6d_benchmark {

class ResultVerifier {
public:
    ResultVerifier() = default;

    // 记录预期的操作
    void RecordAdd(int64_t block_key, const std::string &uri, const std::string &medium);
    void RecordDelete(int64_t block_key, const std::string &medium);

    // 验证查询结果
    struct VerificationResult {
        bool success;
        std::string error_msg;
        int expected_count;
        int actual_count;
        std::vector<int64_t> missing_keys;    // 期望存在但查询未返回的key
        std::vector<int64_t> unexpected_keys; // 不应该存在但查询返回的key (已删除的)
    };

    VerificationResult VerifyQuery(const std::vector<int64_t> &query_keys, const rapidjson::Document &query_response);

    // 获取统计信息
    struct VerificationStats {
        int64_t total_verifications = 0;
        int64_t passed_verifications = 0;
        int64_t failed_verifications = 0;
        int64_t total_missing_keys = 0;
        int64_t total_unexpected_keys = 0;
    };

    VerificationStats GetStats() const {
        std::shared_lock<std::shared_mutex> lock(const_cast<std::shared_mutex &>(mutex_));
        return stats_;
    }

private:
    std::shared_mutex mutex_;
    // block_key -> set of (uri, medium)，记录期望存在的block
    std::unordered_map<int64_t, std::unordered_set<std::string>> expected_blocks_;
    // 记录已删除的block_key
    std::unordered_set<int64_t> deleted_blocks_;

    VerificationStats stats_;
};

inline void ResultVerifier::RecordAdd(int64_t block_key, const std::string &uri, const std::string &medium) {
    std::unique_lock lock(mutex_);
    expected_blocks_[block_key].insert(uri + "#" + medium);
    // 如果之前被删除过，从deleted_blocks_中移除
    deleted_blocks_.erase(block_key);
}

inline void ResultVerifier::RecordDelete(int64_t block_key, const std::string &medium) {
    std::unique_lock lock(mutex_);
    // 从期望状态中移除该medium的记录
    auto it = expected_blocks_.find(block_key);
    if (it != expected_blocks_.end()) {
        // 简单处理：删除该key的所有记录
        // 更精细的实现可以只删除指定medium的记录
        expected_blocks_.erase(it);
    }
    deleted_blocks_.insert(block_key);
}

inline ResultVerifier::VerificationResult ResultVerifier::VerifyQuery(const std::vector<int64_t> &query_keys,
                                                                      const rapidjson::Document &query_response) {

    VerificationResult result;
    result.success = true;
    result.expected_count = 0;
    result.actual_count = 0;

    std::shared_lock lock(mutex_);
    stats_.total_verifications++;

    // 解析查询响应，提取实际返回的keys
    // locations 是 repeated CacheLocation，JSON中为数组
    std::unordered_set<int64_t> actual_keys;
    if (query_response.HasMember("locations")) {
        const auto &locations = query_response["locations"];
        if (locations.IsArray()) {
            // 数组格式：每个元素是一个CacheLocation，按位置匹配query_keys
            for (rapidjson::SizeType i = 0; i < locations.Size(); ++i) {
                const auto &loc = locations[i];
                if (loc.HasMember("location_specs") && loc["location_specs"].IsArray()) {
                    const auto &specs = loc["location_specs"];
                    for (rapidjson::SizeType j = 0; j < specs.Size(); ++j) {
                        if (specs[j].HasMember("uri") && specs[j]["uri"].IsString()) {
                            // 从URI中解析block_key: vineyard://host:port/medium?block_key=123
                            std::string uri = specs[j]["uri"].GetString();
                            auto pos = uri.find("block_key=");
                            if (pos != std::string::npos) {
                                try {
                                    int64_t key = std::stoll(uri.substr(pos + 10));
                                    actual_keys.insert(key);
                                    result.actual_count++;
                                } catch (...) {}
                            }
                        }
                    }
                }
            }
        } else if (locations.IsObject()) {
            // 兼容对象格式（旧版）
            for (auto it = locations.MemberBegin(); it != locations.MemberEnd(); ++it) {
                try {
                    int64_t key = std::stoll(it->name.GetString());
                    actual_keys.insert(key);
                    result.actual_count++;
                } catch (...) {}
            }
        }
    }

    // 检查每个查询的key
    for (int64_t key : query_keys) {
        bool expected_exist = (expected_blocks_.find(key) != expected_blocks_.end());
        bool was_deleted = (deleted_blocks_.find(key) != deleted_blocks_.end());
        bool actual_exist = (actual_keys.find(key) != actual_keys.end());

        if (expected_exist && !was_deleted) {
            result.expected_count++;
            // 期望存在但实际未返回
            if (!actual_exist) {
                result.missing_keys.push_back(key);
                result.success = false;
            }
        } else if (was_deleted) {
            // 期望已删除但实际仍返回
            if (actual_exist) {
                result.unexpected_keys.push_back(key);
                result.success = false;
            }
        }
    }

    // 更新统计
    if (result.success) {
        stats_.passed_verifications++;
    } else {
        stats_.failed_verifications++;
        stats_.total_missing_keys += result.missing_keys.size();
        stats_.total_unexpected_keys += result.unexpected_keys.size();
        // 详细的失败诊断（含 first 5 keys）由调用方 (V6DBenchmark::QueryLocation) 在拿到
        // verify_result 后统一打，这里不再重复打日志。
    }

    return result;
}

} // namespace v6d_benchmark
} // namespace kv_cache_manager
