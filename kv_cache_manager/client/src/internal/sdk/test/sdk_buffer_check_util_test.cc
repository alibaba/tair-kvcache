#include <atomic>
#include <string>
#include <thread>
#ifdef USING_CUDA
#include "kv_cache_manager/client/src/internal/sdk/sdk_buffer_check_util.h"
#endif
#include "kv_cache_manager/common/unittest.h"

using namespace kv_cache_manager;

class SdkBufferCheckUtilTest : public TESTBASE {};

TEST_F(SdkBufferCheckUtilTest, TestGetIovsCrc_1) {
#ifdef USING_CUDA
    SdkBufferCheckUtil::min_cal_byte_size_ = 8;
    std::string s("1234xxxxxxxx5678");
    // crc32("1234xxxxxxxx5678") == DE28307C
    auto byte_size = s.size();
    char *buffer = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&buffer, byte_size));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(buffer, s.data(), byte_size, cudaMemcpyHostToDevice));
    std::vector<IovDevice> iovs_h{{buffer, byte_size}};
    auto crcs = SdkBufferCheckUtil::GetIovsCrc(iovs_h);
    ASSERT_EQ(std::vector<uint32_t>({0xDE28307C}), crcs);
    ASSERT_EQ(cudaSuccess, cudaFree(buffer));
#endif
}

TEST_F(SdkBufferCheckUtilTest, TestGetIovsCrc_2) {
#ifdef USING_CUDA
    SdkBufferCheckUtil::min_cal_byte_size_ = 4;
    std::string s("1234xxxxxxxx5678");
    // crc32("12345678") == 9AE0DAAF
    auto byte_size = s.size();
    char *buffer = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&buffer, byte_size));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(buffer, s.data(), byte_size, cudaMemcpyHostToDevice));
    std::vector<IovDevice> iovs_h{{buffer, byte_size}};
    auto crcs = SdkBufferCheckUtil::GetIovsCrc(iovs_h);
    ASSERT_EQ(std::vector<uint32_t>({0x9AE0DAAF}), crcs);
    ASSERT_EQ(cudaSuccess, cudaFree(buffer));
#endif
}

TEST_F(SdkBufferCheckUtilTest, TestGetIovsCrcTooSmall_1) {
#ifdef USING_CUDA
    ScopedEnv env("KVCM_CHECK_IOV_BYTE_SIZE", "4");
    std::string s("1278");
    // crc32("1278") == F639698C
    auto byte_size = s.size();
    char *buffer = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&buffer, byte_size));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(buffer, s.data(), byte_size, cudaMemcpyHostToDevice));
    std::vector<IovDevice> iovs_h{{buffer, byte_size}};
    auto crcs = SdkBufferCheckUtil::GetIovsCrc(iovs_h);
    ASSERT_EQ(std::vector<uint32_t>({0xF639698C}), crcs);
    ASSERT_EQ(cudaSuccess, cudaFree(buffer));
#endif
}

TEST_F(SdkBufferCheckUtilTest, TestGetIovsCrcTooSmall_2) {
#ifdef USING_CUDA
    ScopedEnv env("KVCM_CHECK_IOV_BYTE_SIZE", "4");
    std::string s("1");
    auto byte_size = s.size();
    char *buffer = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&buffer, byte_size));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(buffer, s.data(), byte_size, cudaMemcpyHostToDevice));
    std::vector<IovDevice> iovs_h{{buffer, byte_size}};
    auto crcs = SdkBufferCheckUtil::GetIovsCrc(iovs_h);
    ASSERT_TRUE(crcs.empty());
    ASSERT_EQ(cudaSuccess, cudaFree(buffer));
#endif
}

TEST_F(SdkBufferCheckUtilTest, TestGetIovsCrcAutoMalloc) {
#ifdef USING_CUDA
    char *buffer = nullptr;
    auto byte_size = 4 * 1024;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&buffer, byte_size));
    std::string s(byte_size, 'a');
    ASSERT_EQ(cudaSuccess, cudaMemcpy(buffer, s.data(), byte_size, cudaMemcpyHostToDevice));
    std::vector<IovDevice> iovs_h;
    iovs_h.push_back({buffer, 1024});
    iovs_h.push_back({buffer + 1024, 1024});
    iovs_h.push_back({buffer + 1024 * 2, 1024});
    iovs_h.push_back({buffer + 1024 * 3, 1024});
    auto crcs = SdkBufferCheckUtil::GetIovsCrc(iovs_h);
    ASSERT_EQ(4, crcs.size());
    ASSERT_EQ(cudaSuccess, cudaFree(buffer));
#endif
}

TEST_F(SdkBufferCheckUtilTest, TestGetIovsCrcWithStream) {
#ifdef USING_CUDA
    char *buffer = nullptr;
    auto byte_size = 4 * 1024;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&buffer, byte_size));
    std::string s(byte_size, 'a');
    ASSERT_EQ(cudaSuccess, cudaMemcpy(buffer, s.data(), byte_size, cudaMemcpyHostToDevice));
    std::vector<IovDevice> iovs_h;
    iovs_h.push_back({buffer, 1024});
    iovs_h.push_back({buffer + 1024, 1024});
    iovs_h.push_back({buffer + 1024 * 2, 1024});
    iovs_h.push_back({buffer + 1024 * 3, 1024});
    IovDevice *iovs_d = nullptr;
    uint32_t *crcs_d = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&iovs_d, sizeof(IovDevice) * iovs_h.size()));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&crcs_d, sizeof(uint32_t) * iovs_h.size()));
    cudaStream_t stream = nullptr;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    auto crcs = SdkBufferCheckUtil::GetIovsCrc(iovs_h, iovs_d, crcs_d, stream);
    ASSERT_EQ(4, crcs.size());
    ASSERT_EQ(cudaSuccess, cudaFree(buffer));
    ASSERT_EQ(cudaSuccess, cudaFree(iovs_d));
    ASSERT_EQ(cudaSuccess, cudaFree(crcs_d));
#endif
}

TEST_F(SdkBufferCheckUtilTest, TestGetIovsCrcWithStreamAndPinnedMem) {
#ifdef USING_CUDA
    char *buffer = nullptr;
    auto byte_size = 4 * 1024;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&buffer, byte_size));
    std::string s(byte_size, 'a');
    ASSERT_EQ(cudaSuccess, cudaMemcpy(buffer, s.data(), byte_size, cudaMemcpyHostToDevice));
    IovDevice *pinned_iovs_ptr = nullptr;
    size_t iovs_size = 4;
    ASSERT_EQ(cudaSuccess, cudaMallocHost(&pinned_iovs_ptr, iovs_size * sizeof(IovDevice)));
    pinned_iovs_ptr[0] = {buffer, 1024};
    pinned_iovs_ptr[1] = {buffer + 1024, 1024};
    pinned_iovs_ptr[2] = {buffer + 1024 * 2, 1024};
    pinned_iovs_ptr[3] = {buffer + 1024 * 3, 1024};
    IovDevice *iovs_d = nullptr;
    uint32_t *crcs_d = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&iovs_d, sizeof(IovDevice) * iovs_size));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&crcs_d, sizeof(uint32_t) * iovs_size));
    cudaStream_t stream = nullptr;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    auto crcs = SdkBufferCheckUtil::GetIovsCrc(pinned_iovs_ptr, iovs_size, iovs_d, crcs_d, stream);
    ASSERT_EQ(4, crcs.size());
    ASSERT_EQ(cudaSuccess, cudaFree(buffer));
    ASSERT_EQ(cudaSuccess, cudaFree(iovs_d));
    ASSERT_EQ(cudaSuccess, cudaFree(crcs_d));
    ASSERT_EQ(cudaSuccess, cudaFreeHost(pinned_iovs_ptr));
#endif
}

TEST_F(SdkBufferCheckUtilTest, TestGetBlocksHashAutoMalloc) {
#ifdef USING_CUDA
    auto byte_size = 4 * 1024;
    BlockBuffers block_buffers;
    auto push_buffer = [&block_buffers, byte_size]() {
        char *buffer = nullptr;
        ASSERT_EQ(cudaSuccess, cudaMalloc(&buffer, byte_size));
        block_buffers.push_back({});
        auto &iovs = block_buffers.back().iovs;
        iovs.push_back({MemoryType::GPU, buffer, 1024, false});
        iovs.push_back({MemoryType::GPU, buffer + 1024, 1024, false});
        iovs.push_back({MemoryType::GPU, buffer + 2 * 1024, 1024, false});
        iovs.push_back({MemoryType::GPU, buffer + 3 * 1024, 1024, false});
    };
    push_buffer();
    push_buffer();
    auto hashs = SdkBufferCheckUtil::GetBlocksHash(block_buffers);
    ASSERT_EQ(2, hashs.size());
    ASSERT_EQ(cudaSuccess, cudaFree(block_buffers[0].iovs[0].base));
    ASSERT_EQ(cudaSuccess, cudaFree(block_buffers[1].iovs[0].base));
#endif
}

TEST_F(SdkBufferCheckUtilTest, TestGetBlocksHashManuallyMalloc) {
#ifdef USING_CUDA
    auto byte_size = 4 * 1024;
    BlockBuffers block_buffers;
    auto push_buffer = [&block_buffers, byte_size]() {
        char *buffer = nullptr;
        ASSERT_EQ(cudaSuccess, cudaMalloc(&buffer, byte_size));
        block_buffers.push_back({});
        auto &iovs = block_buffers.back().iovs;
        iovs.push_back({MemoryType::GPU, buffer, 1024, false});
        iovs.push_back({MemoryType::GPU, buffer + 1024, 1024, false});
        iovs.push_back({MemoryType::GPU, buffer + 2 * 1024, 1024, false});
        iovs.push_back({MemoryType::GPU, buffer + 3 * 1024, 1024, false});
    };
    push_buffer();
    push_buffer();
    IovDevice *iovs_d = nullptr;
    uint32_t *crcs_d = nullptr;
    constexpr auto max_iov_num = 6;
    constexpr auto iovs_d_byte_size = sizeof(IovDevice) * max_iov_num;
    constexpr auto crcs_d_byte_size = sizeof(uint32_t) * max_iov_num;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&iovs_d, iovs_d_byte_size));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&crcs_d, crcs_d_byte_size));
    std::string init(1024, 'a');
    ASSERT_EQ(cudaSuccess, cudaMemcpy(iovs_d, init.data(), iovs_d_byte_size, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(crcs_d, init.data(), crcs_d_byte_size, cudaMemcpyHostToDevice));

    auto hashs = SdkBufferCheckUtil::GetBlocksHash(block_buffers, iovs_d, crcs_d, max_iov_num, nullptr);
    ASSERT_EQ(1, hashs.size());
    std::string after_hash_iovs_d(1024, ' ');
    std::string after_hash_crcs_d(1024, ' ');
    ASSERT_EQ(cudaSuccess, cudaMemcpy(after_hash_iovs_d.data(), iovs_d, iovs_d_byte_size, cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(after_hash_crcs_d.data(), crcs_d, crcs_d_byte_size, cudaMemcpyDeviceToHost));
    constexpr auto real_iov_num = 4;
    constexpr auto change_iovs_d_byte_size = sizeof(IovDevice) * real_iov_num;
    constexpr auto change_crcs_d_byte_size = sizeof(uint32_t) * real_iov_num;
    ASSERT_NE(std::string(change_iovs_d_byte_size, 'a'), after_hash_iovs_d.substr(0, change_iovs_d_byte_size));
    ASSERT_NE(std::string(change_crcs_d_byte_size, 'a'), after_hash_crcs_d.substr(0, change_crcs_d_byte_size));
    constexpr auto not_change_iovs_d_byte_size = iovs_d_byte_size - change_iovs_d_byte_size;
    constexpr auto not_change_crcs_d_byte_size = crcs_d_byte_size - change_crcs_d_byte_size;
    ASSERT_EQ(std::string(not_change_iovs_d_byte_size, 'a'),
              after_hash_iovs_d.substr(change_iovs_d_byte_size, not_change_iovs_d_byte_size));
    ASSERT_EQ(std::string(not_change_crcs_d_byte_size, 'a'),
              after_hash_crcs_d.substr(change_crcs_d_byte_size, not_change_crcs_d_byte_size));

    ASSERT_EQ(cudaSuccess, cudaFree(block_buffers[0].iovs[0].base));
    ASSERT_EQ(cudaSuccess, cudaFree(block_buffers[1].iovs[0].base));
    ASSERT_EQ(cudaSuccess, cudaFree(iovs_d));
    ASSERT_EQ(cudaSuccess, cudaFree(crcs_d));
#endif
}

TEST_F(SdkBufferCheckUtilTest, TestGetBlocksHashManuallyMallocAndPinnedMem) {
#ifdef USING_CUDA
    auto byte_size = 4 * 1024;
    BlockBuffers block_buffers;
    auto push_buffer = [&block_buffers, byte_size]() {
        char *buffer = nullptr;
        ASSERT_EQ(cudaSuccess, cudaMalloc(&buffer, byte_size));
        block_buffers.push_back({});
        auto &iovs = block_buffers.back().iovs;
        iovs.push_back({MemoryType::GPU, buffer, 1024, false});
        iovs.push_back({MemoryType::GPU, buffer + 1024, 1024, false});
        iovs.push_back({MemoryType::GPU, buffer + 2 * 1024, 1024, false});
        iovs.push_back({MemoryType::GPU, buffer + 3 * 1024, 1024, false});
    };
    push_buffer();
    push_buffer();
    IovDevice *iovs_d = nullptr;
    uint32_t *crcs_d = nullptr;
    constexpr auto max_iov_num = 6;
    constexpr auto iovs_d_byte_size = sizeof(IovDevice) * max_iov_num;
    constexpr auto crcs_d_byte_size = sizeof(uint32_t) * max_iov_num;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&iovs_d, iovs_d_byte_size));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&crcs_d, crcs_d_byte_size));
    std::string init(1024, 'a');
    ASSERT_EQ(cudaSuccess, cudaMemcpy(iovs_d, init.data(), iovs_d_byte_size, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(crcs_d, init.data(), crcs_d_byte_size, cudaMemcpyHostToDevice));
    IovDevice *pinned_iovs_ptr = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMallocHost(&pinned_iovs_ptr, max_iov_num * sizeof(IovDevice)));
    cudaStream_t stream = nullptr;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    auto hashs = SdkBufferCheckUtil::GetBlocksHash(block_buffers, iovs_d, crcs_d, pinned_iovs_ptr, max_iov_num, stream);
    ASSERT_EQ(1, hashs.size());
    std::string after_hash_iovs_d(1024, ' ');
    std::string after_hash_crcs_d(1024, ' ');
    ASSERT_EQ(cudaSuccess, cudaMemcpy(after_hash_iovs_d.data(), iovs_d, iovs_d_byte_size, cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(after_hash_crcs_d.data(), crcs_d, crcs_d_byte_size, cudaMemcpyDeviceToHost));
    constexpr auto real_iov_num = 4;
    constexpr auto change_iovs_d_byte_size = sizeof(IovDevice) * real_iov_num;
    constexpr auto change_crcs_d_byte_size = sizeof(uint32_t) * real_iov_num;
    ASSERT_NE(std::string(change_iovs_d_byte_size, 'a'), after_hash_iovs_d.substr(0, change_iovs_d_byte_size));
    ASSERT_NE(std::string(change_crcs_d_byte_size, 'a'), after_hash_crcs_d.substr(0, change_crcs_d_byte_size));
    constexpr auto not_change_iovs_d_byte_size = iovs_d_byte_size - change_iovs_d_byte_size;
    constexpr auto not_change_crcs_d_byte_size = crcs_d_byte_size - change_crcs_d_byte_size;
    ASSERT_EQ(std::string(not_change_iovs_d_byte_size, 'a'),
              after_hash_iovs_d.substr(change_iovs_d_byte_size, not_change_iovs_d_byte_size));
    ASSERT_EQ(std::string(not_change_crcs_d_byte_size, 'a'),
              after_hash_crcs_d.substr(change_crcs_d_byte_size, not_change_crcs_d_byte_size));

    ASSERT_EQ(cudaSuccess, cudaFree(block_buffers[0].iovs[0].base));
    ASSERT_EQ(cudaSuccess, cudaFree(block_buffers[1].iovs[0].base));
    ASSERT_EQ(cudaSuccess, cudaFree(iovs_d));
    ASSERT_EQ(cudaSuccess, cudaFree(crcs_d));
    ASSERT_EQ(cudaSuccess, cudaFreeHost(pinned_iovs_ptr));
#endif
}

TEST_F(SdkBufferCheckUtilTest, TestSdkBufferCheckPool) {
#ifdef USING_CUDA
    SdkBufferCheckPool pool;
    size_t max_check_iov_num = 10;
    ASSERT_TRUE(pool.Init(max_check_iov_num));
    auto byte_size = 4 * 1024;
    BlockBuffers block_buffers;
    auto push_buffer = [&block_buffers, byte_size]() {
        char *buffer = nullptr;
        ASSERT_EQ(cudaSuccess, cudaMalloc(&buffer, byte_size));
        std::string init(byte_size, 'a');
        ASSERT_EQ(cudaSuccess, cudaMemcpy(buffer, init.data(), byte_size, cudaMemcpyHostToDevice));
        block_buffers.push_back({});
        auto &iovs = block_buffers.back().iovs;
        iovs.push_back({MemoryType::GPU, buffer, 1024, false});
        iovs.push_back({MemoryType::GPU, buffer + 1024, 1024, false});
        iovs.push_back({MemoryType::GPU, buffer + 2 * 1024, 1024, false});
        iovs.push_back({MemoryType::GPU, buffer + 3 * 1024, 1024, false});
    };
    push_buffer();
    push_buffer();
    auto handle = pool.GetCell();
    auto block_hashs = SdkBufferCheckUtil::GetBlocksHash(
        block_buffers, handle->d_iovs, handle->d_crcs, handle->h_iovs, max_check_iov_num, handle->cuda_stream);
    ASSERT_EQ(2, block_hashs.size());
    ASSERT_EQ(block_hashs[0], block_hashs[1]);
#endif
}

TEST_F(SdkBufferCheckUtilTest, TestSdkBufferCheckPoolGetCell) {
#ifdef USING_CUDA
    SdkBufferCheckPool pool(2);
    size_t max_check_iov_num = 10;
    ASSERT_TRUE(pool.Init(max_check_iov_num));
    auto handle1 = pool.GetCell();
    ASSERT_EQ(1, pool.cell_queue_.size());
    auto handle2 = pool.GetCell();
    ASSERT_EQ(0, pool.cell_queue_.size());
#endif
}

TEST_F(SdkBufferCheckUtilTest, TestSdkBufferCheckPoolMultiThread) {
#ifdef USING_CUDA
    SdkBufferCheckPool pool;
    size_t max_check_iov_num = 10;
    ASSERT_TRUE(pool.Init(max_check_iov_num));
    auto byte_size = 4 * 1024;
    auto push_buffer = [byte_size](BlockBuffers &block_buffers) {
        char *buffer = nullptr;
        ASSERT_EQ(cudaSuccess, cudaMalloc(&buffer, byte_size));
        std::string init(byte_size, 'a');
        ASSERT_EQ(cudaSuccess, cudaMemcpy(buffer, init.data(), byte_size, cudaMemcpyHostToDevice));
        block_buffers.push_back({});
        auto &iovs = block_buffers.back().iovs;
        iovs.push_back({MemoryType::GPU, buffer, 1024, false});
        iovs.push_back({MemoryType::GPU, buffer + 1024, 1024, false});
        iovs.push_back({MemoryType::GPU, buffer + 2 * 1024, 1024, false});
        iovs.push_back({MemoryType::GPU, buffer + 3 * 1024, 1024, false});
    };
    std::vector<BlockBuffers> block_buffers_vec(2);
    push_buffer(block_buffers_vec[0]);
    push_buffer(block_buffers_vec[1]);
    size_t thread_num = 8;
    std::atomic<bool> go = false;
    int64_t expect;
    {
        auto handle = pool.GetCell();
        auto block_hashs = SdkBufferCheckUtil::GetBlocksHash(block_buffers_vec[0],
                                                             handle->d_iovs,
                                                             handle->d_crcs,
                                                             handle->h_iovs,
                                                             max_check_iov_num,
                                                             handle->cuda_stream);
        ASSERT_EQ(1, block_hashs.size());
        expect = block_hashs[0];
    }
    auto thread_fcn = [this, &pool, &go, &block_buffers_vec, max_check_iov_num, expect](int i) {
        while (!go.load(std::memory_order_relaxed)) {}
        auto handle = pool.GetCell();
        auto block_hashs = SdkBufferCheckUtil::GetBlocksHash(block_buffers_vec[i % 2],
                                                             handle->d_iovs,
                                                             handle->d_crcs,
                                                             handle->h_iovs,
                                                             max_check_iov_num,
                                                             handle->cuda_stream);
        ASSERT_EQ(std::vector<int64_t>({expect}), block_hashs);
    };
    for (int i = 0; i < 20; ++i) {
        std::vector<std::thread> threads;
        for (int j = 0; j < thread_num; ++j) {
            threads.push_back(std::thread(thread_fcn, i));
        }
        go.store(true, std::memory_order_relaxed);
        for (auto &thread : threads) {
            thread.join();
        }
        go.store(false, std::memory_order_relaxed);
    }
#endif
}