#pragma once

#include <cstdint>
#include <map>
#include <shared_mutex>

namespace kv_cache_manager {

class Hf3fsMempool final {
public:
    Hf3fsMempool(void *buffer, size_t total_size, size_t align_size)
        : buffer_(static_cast<uint8_t *>(buffer)), total_size_(total_size), align_size_(align_size) {}
    ~Hf3fsMempool();

public:
    bool Init();
    void *Alloc(size_t size);
    void Free(void *ptr);

    size_t AllocatedSize() const;
    size_t FreeSize() const;
    int AllocatedBlockCount() const;

    // for debug
    void PrintStatus() const;

private:
    int FreeBlockCount() const;
    std::string ToString() const;

private:
    uint8_t *buffer_;
    size_t total_size_;
    size_t align_size_;

    // 空闲块管理: 起始地址 -> 块大小
    std::map<uint8_t *, size_t> free_blocks_;
    mutable std::shared_mutex free_blocks_mutex_;

    // 已分配块管理: 起始地址 -> 块大小
    std::map<uint8_t *, size_t> allocated_blocks_;
    mutable std::shared_mutex allocated_blocks_mutex_;
};

} // namespace kv_cache_manager