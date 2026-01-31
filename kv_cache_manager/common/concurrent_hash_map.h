#pragma once

#include <mutex>
#include <shared_mutex>
#include <unordered_map>

namespace kv_cache_manager {

template <typename Key, typename Value, typename Hash = std::hash<Key>, typename KeyEqual = std::equal_to<Key>>
class ConcurrentHashMap {
public:
    using KeyType = Key;
    using ValueType = Value;
    using ValueTypePair = std::pair<const Key, Value>;
    using SizeType = typename std::unordered_map<Key, Value, Hash, KeyEqual>::size_type;
    using Hasher = Hash;
    using KeyEqualFunc = KeyEqual;
    using Iterator = typename std::unordered_map<Key, Value, Hash, KeyEqual>::iterator;
    using ConstIterator = typename std::unordered_map<Key, Value, Hash, KeyEqual>::const_iterator;

    // Default constructor
    ConcurrentHashMap() = default;

    // Constructor with bucket count
    explicit ConcurrentHashMap(SizeType bucket_count,
                               const Hasher &hash = Hasher(),
                               const KeyEqualFunc &equal = KeyEqualFunc())
        : map_(bucket_count, hash, equal) {}

    // Copy constructor
    ConcurrentHashMap(const ConcurrentHashMap &other) {
        std::shared_lock<std::shared_mutex> lock(other.mutex_);
        map_ = other.map_;
    }

    // Move constructor
    ConcurrentHashMap(ConcurrentHashMap &&other) {
        std::unique_lock<std::shared_mutex> lock(other.mutex_);
        map_ = std::move(other.map_);
    }

    // Copy assignment operator
    ConcurrentHashMap &operator=(const ConcurrentHashMap &other) {
        if (this != &other) {
            std::unique_lock<std::shared_mutex> lock_this(mutex_, std::defer_lock);
            std::shared_lock<std::shared_mutex> lock_other(other.mutex_, std::defer_lock);
            std::lock(lock_this, lock_other);
            map_ = other.map_;
        }
        return *this;
    }

    // Move assignment operator
    ConcurrentHashMap &operator=(ConcurrentHashMap &&other) {
        if (this != &other) {
            std::unique_lock<std::shared_mutex> lock_this(mutex_, std::defer_lock);
            std::unique_lock<std::shared_mutex> lock_other(other.mutex_, std::defer_lock);
            std::lock(lock_this, lock_other);
            map_ = std::move(other.map_);
        }
        return *this;
    }

    // Destructor
    ~ConcurrentHashMap() = default;

    // Element access
    ValueType &At(const KeyType &key) {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return map_.at(key);
    }

    const ValueType &At(const KeyType &key) const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return map_.at(key);
    }

    ValueType &operator[](const KeyType &key) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        return map_[key];
    }

    // Note: These iterators should only be used while holding the lock
    ConstIterator Begin() const { return map_.begin(); }

    ConstIterator End() const { return map_.end(); }

    template <typename Func>
    void ForEach(Func func) {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        for (const auto &pair : map_) {
            if (!func(pair)) {
                break;
            }
        }
    }

    template <typename Func>
    void ForEachKV(Func func) {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        for (const auto &[key, value] : map_) {
            if (!func(key, value)) {
                break;
            }
        }
    }

    // Capacity
    bool Empty() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return map_.empty();
    }

    SizeType Size() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return map_.size();
    }

    // Modifiers
    void Clear() {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        map_.clear();
    }

    std::pair<Iterator, bool> Insert(const ValueTypePair &value) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        return map_.insert(value);
    }

    std::pair<Iterator, bool> Insert(ValueTypePair &&value) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        return map_.insert(std::move(value));
    }

    void Insert(std::initializer_list<ValueTypePair> ilist) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        map_.insert(ilist);
    }

    template <class... Args>
    std::pair<Iterator, bool> Emplace(Args &&...args) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        return map_.emplace(std::forward<Args>(args)...);
    }

    template <class... Args>
    std::pair<Iterator, bool> TryEmplace(const KeyType &k, Args &&...args) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        return map_.try_emplace(k, std::forward<Args>(args)...);
    }

    template <class... Args>
    std::pair<Iterator, bool> TryEmplace(KeyType &&k, Args &&...args) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        return map_.try_emplace(std::move(k), std::forward<Args>(args)...);
    }

    // Erase
    SizeType Erase(const KeyType &key) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        return map_.erase(key);
    }

    void Erase(ConstIterator pos) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        map_.erase(pos);
    }

    void Erase(ConstIterator first, ConstIterator last) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        map_.erase(first, last);
    }

    // Lookup
    SizeType Count(const KeyType &key) const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return map_.count(key);
    }

    Iterator Find(const KeyType &key) {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return map_.find(key);
    }

    ConstIterator Find(const KeyType &key) const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return map_.find(key);
    }

    bool Contains(const KeyType &key) const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        // return map_.contains(key);
        return map_.find(key) != map_.end();
    }

    // Bucket interface
    SizeType GetBucketCount() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return map_.bucket_count();
    }

    SizeType GetMaxBucketCount() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return map_.max_bucket_count();
    }

    void Rehash(SizeType count) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        map_.rehash(count);
    }

    void Reserve(SizeType count) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        map_.reserve(count);
    }

private:
    mutable std::shared_mutex mutex_;
    std::unordered_map<Key, Value, Hash, KeyEqual> map_;
};

} // namespace kv_cache_manager
