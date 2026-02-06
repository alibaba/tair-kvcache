#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>
#include <vector>

#include "kv_cache_manager/client/include/common.h"
#include "kv_cache_manager/client/src/internal/sdk/cuda_util.h"

namespace kv_cache_manager {

struct IovDevice {
    const void *base;
    size_t size;
};

class SdkBufferCheckUtil {
public:
    static std::vector<int64_t> GetBlocksHash(const BlockBuffers &block_buffers);
    static std::vector<int64_t> GetBlocksHash(const BlockBuffers &block_buffers,
                                              IovDevice *iovs_d,
                                              uint32_t *crcs_d,
                                              size_t max_iov_num,
                                              cudaStream_t stream);
    static std::vector<int64_t> GetBlocksHash(const BlockBuffers &block_buffers,
                                              IovDevice *iovs_d,
                                              uint32_t *crcs_d,
                                              IovDevice *iovs_h_to_save,
                                              size_t max_iov_num,
                                              cudaStream_t stream);

    static std::vector<uint32_t> GetIovsCrc(const std::vector<IovDevice> &iovs_h);
    static std::vector<uint32_t>
    GetIovsCrc(const std::vector<IovDevice> &iovs_h, IovDevice *iovs_d, uint32_t *crcs_d, cudaStream_t stream);
    static std::vector<uint32_t>
    GetIovsCrc(const IovDevice *iovs_h_ptr, size_t iovs_size, IovDevice *iovs_d, uint32_t *crcs_d, cudaStream_t stream);

private:
    static size_t min_cal_byte_size_;
};

class SdkBufferCheckPool {
    static constexpr size_t kDefaultCellNum = 4;

public:
    explicit SdkBufferCheckPool(size_t cell_num = kDefaultCellNum);
    ~SdkBufferCheckPool();

    struct Cell {
        IovDevice *h_iovs = nullptr;
        IovDevice *d_iovs = nullptr;
        uint32_t *d_crcs = nullptr;
        cudaStream_t cuda_stream = nullptr;
    };

    class CellHandle {
    public:
        CellHandle(SdkBufferCheckPool *pool, Cell *cell) : pool_(pool), cell_(cell) {}
        CellHandle(const CellHandle &) = delete;
        CellHandle(CellHandle &&other) : pool_(std::move(other.pool_)), cell_(std::move(other.cell_)) {
            other.pool_ = nullptr;
            other.cell_ = nullptr;
        }
        ~CellHandle();
        Cell *operator->() { return cell_; }
        Cell &operator*() { return *cell_; }
        explicit operator bool() const { return cell_ != nullptr; }

    private:
        SdkBufferCheckPool *pool_;
        Cell *cell_;
    };

    bool Init(size_t max_check_iov_num);
    CellHandle GetCell();

private:
    friend class CellHandle;
    void PutCell(Cell *cell);

    std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<Cell *> cell_queue_;
    std::vector<Cell> cells_;
};

}; // namespace kv_cache_manager