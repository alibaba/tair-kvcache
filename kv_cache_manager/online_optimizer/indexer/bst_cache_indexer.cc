#include "kv_cache_manager/online_optimizer/indexer/bst_cache_indexer.h"

#include <algorithm>
#include <cmath>
#include <climits>

namespace kv_cache_manager {

// ==================== AugmentedTreap ====================

AugmentedTreap::AugmentedTreap() : root_(nullptr), rng_(42) {}

AugmentedTreap::~AugmentedTreap() { DestroyTree(root_); }

int64_t AugmentedTreap::GetSize(TreapNode *node) {
    return node ? node->size : 0;
}

void AugmentedTreap::UpdateSize(TreapNode *node) {
    if (node) {
        node->size = 1 + GetSize(node->left) + GetSize(node->right);
    }
}

void AugmentedTreap::SplitByKey(TreapNode *node, int64_t key, TreapNode *&left, TreapNode *&right) {
    if (!node) {
        left = right = nullptr;
        return;
    }
    if (node->key <= key) {
        SplitByKey(node->right, key, node->right, right);
        left = node;
    } else {
        SplitByKey(node->left, key, left, node->left);
        right = node;
    }
    UpdateSize(node);
}

TreapNode *AugmentedTreap::Merge(TreapNode *left, TreapNode *right) {
    if (!left || !right) return left ? left : right;

    if (left->priority > right->priority) {
        left->right = Merge(left->right, right);
        UpdateSize(left);
        return left;
    } else {
        right->left = Merge(left, right->left);
        UpdateSize(right);
        return right;
    }
}

void AugmentedTreap::DestroyTree(TreapNode *node) {
    if (!node) return;
    DestroyTree(node->left);
    DestroyTree(node->right);
    delete node;
}

void AugmentedTreap::Insert(int64_t key) {
    auto *node = new TreapNode(key, static_cast<int64_t>(rng_()));
    TreapNode *left = nullptr;
    TreapNode *right = nullptr;
    SplitByKey(root_, key, left, right);
    root_ = Merge(Merge(left, node), right);
}

void AugmentedTreap::Erase(int64_t key) {
    TreapNode *left = nullptr;
    TreapNode *mid = nullptr;
    TreapNode *right = nullptr;

    SplitByKey(root_, key, left, right);
    SplitByKey(left, key - 1, left, mid);

    if (mid) {
        TreapNode *to_delete = mid;
        mid = Merge(mid->left, mid->right);
        to_delete->left = nullptr;
        to_delete->right = nullptr;
        delete to_delete;
    }

    root_ = Merge(Merge(left, mid), right);
}

int64_t AugmentedTreap::CountGreater(int64_t key) const {
    TreapNode *cur = root_;
    int64_t count = 0;
    while (cur) {
        if (key < cur->key) {
            count += 1 + GetSize(cur->right);
            cur = cur->left;
        } else if (key > cur->key) {
            cur = cur->right;
        } else {
            count += GetSize(cur->right);
            break;
        }
    }
    return count;
}

int64_t AugmentedTreap::Size() const {
    return GetSize(root_);
}

int64_t AugmentedTreap::Min() const {
    TreapNode *cur = root_;
    if (!cur) return 0;
    while (cur->left) cur = cur->left;
    return cur->key;
}

// ==================== BSTCacheIndexer ====================

BSTCacheIndexer::BSTCacheIndexer(int64_t max_key_count) : max_key_count_(max_key_count) {}

void BSTCacheIndexer::Init(const std::vector<double> &capacity_gb,
                            int64_t size_full_only,
                            int64_t size_full_linear,
                            int32_t linear_step) {
    linear_step = std::max(linear_step, int32_t(1));
    int64_t avg_bytes_per_block;
    if (linear_step <= 1) {
        avg_bytes_per_block = size_full_linear;
    } else {
        avg_bytes_per_block =
            ((linear_step - 1) * size_full_only + size_full_linear) / linear_step;
    }

    avg_bytes_per_block_ = avg_bytes_per_block;

    capacity_blocks_.resize(capacity_gb.size());
    for (size_t i = 0; i < capacity_gb.size(); i++) {
        int64_t bytes = static_cast<int64_t>(capacity_gb[i] * 1024.0 * 1024.0 * 1024.0);
        capacity_blocks_[i] = (avg_bytes_per_block_ > 0) ? bytes / avg_bytes_per_block_ : 0;
    }
}

int64_t BSTCacheIndexer::kv_cache_usage_bytes() const {
    return unique_count() * avg_bytes_per_block_;
}

int64_t BSTCacheIndexer::ComputeStackDistance(int64_t key) {
    int64_t sd = INT64_MAX;
    auto it = last_access_.find(key);
    if (it != last_access_.end()) {
        int64_t t_prev = it->second;
        sd = treap_.CountGreater(t_prev);
        treap_.Erase(t_prev);
        reverse_map_.erase(t_prev);
    }

    treap_.Insert(logical_time_);
    last_access_[key] = logical_time_;
    reverse_map_[logical_time_] = key;
    logical_time_++;

    return sd;
}

void BSTCacheIndexer::ProcessKeys(const std::vector<int64_t> &keys,
                                  std::vector<int64_t> &hit_count,
                                  int64_t &max_hit_count) {
    const size_t num_caps = capacity_blocks_.size();
    const int64_t total_keys = static_cast<int64_t>(keys.size());
    hit_count.assign(num_caps, total_keys);
    max_hit_count = (max_key_count_ <= 0) ? total_keys : -1;

    for (int64_t i = 0; i < total_keys; i++) {
        int64_t sd = ComputeStackDistance(keys[i]);
        for (size_t j = 0; j < num_caps; j++) {
            bool is_hit = (sd != INT64_MAX && sd < capacity_blocks_[j]);
            if (i < hit_count[j] && !is_hit) {
                hit_count[j] = i;
            }
        }
        if (max_hit_count > 0 && i < max_hit_count && sd == INT64_MAX) {
            max_hit_count = i;
        }
    }
}

int64_t BSTCacheIndexer::unique_count() const {
    return treap_.Size();
}

void BSTCacheIndexer::PostQueryMaintenance() {
    EvictIfExceedsCapacity();
}

void BSTCacheIndexer::EvictIfExceedsCapacity() {
    if (max_key_count_ <= 0) return;
    while (treap_.Size() > max_key_count_) {
        DoEvictOne();
    }
}

void BSTCacheIndexer::DoEvictOne() {
    int64_t t_min = treap_.Min();
    auto rev_it = reverse_map_.find(t_min);
    if (rev_it == reverse_map_.end()) return;
    int64_t evicted_key = rev_it->second;
    treap_.Erase(t_min);
    reverse_map_.erase(rev_it);
    last_access_.erase(evicted_key);
    eviction_count_++;
}

bool BSTCacheIndexer::RemoveKey(int64_t key) {
    auto it = last_access_.find(key);
    if (it == last_access_.end()) return false;
    int64_t t = it->second;
    treap_.Erase(t);
    reverse_map_.erase(t);
    last_access_.erase(it);
    eviction_count_++;
    return true;
}

int64_t BSTCacheIndexer::memory_usage_bytes() const {
    // TreapNode: ~48 bytes each (key, priority, size, left, right pointers + allocation overhead)
    constexpr int64_t kTreapNodeBytes = 48;
    int64_t treap_bytes = treap_.Size() * kTreapNodeBytes;
    // unordered_map overhead: ~56 bytes per entry
    constexpr int64_t kMapEntryBytes = 56;
    int64_t map_bytes = static_cast<int64_t>(last_access_.size() + reverse_map_.size()) * kMapEntryBytes;
    return treap_bytes + map_bytes;
}

} // namespace kv_cache_manager
