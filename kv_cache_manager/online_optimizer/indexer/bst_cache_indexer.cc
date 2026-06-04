#include "kv_cache_manager/online_optimizer/indexer/bst_cache_indexer.h"

#include <algorithm>
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

// ==================== BSTCacheIndexer ====================

BSTCacheIndexer::BSTCacheIndexer(int64_t max_key_count) : max_key_count_(max_key_count) {}

int64_t BSTCacheIndexer::ProcessKey(int64_t key) {
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

    int64_t current_unique = treap_.Size();
    if (current_unique > peak_unique_count_) {
        peak_unique_count_ = current_unique;
    }

    return sd;
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
    while (reverse_map_.find(min_time_) == reverse_map_.end()) {
        min_time_++;
    }
    int64_t evicted_key = reverse_map_[min_time_];
    treap_.Erase(min_time_);
    reverse_map_.erase(min_time_);
    last_access_.erase(evicted_key);
    min_time_++;
}

} // namespace kv_cache_manager
