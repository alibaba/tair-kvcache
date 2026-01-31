#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

#include "kv_cache_manager/client/src/internal/sdk/hf3fs_cuda_util.h"
#include "kv_cache_manager/client/src/internal/sdk/hf3fs_mempool.h"
#include "kv_cache_manager/client/src/internal/sdk/hf3fs_usrbio_api.h"
#include "kv_cache_manager/client/src/internal/sdk/hf3fs_usrbio_client.h"
#include "kv_cache_manager/client/src/internal/sdk/test/mock/mock_hf3fs_usrbio_api.h"
#include "kv_cache_manager/common/unittest.h"

using namespace ::testing;
using namespace kv_cache_manager;

class Hf3fsUsrbioClientTest : public TESTBASE {
public:
    void SetUp() override {
        const auto test_path = GetTestTempRootPath();
        auto cfg = Hf3fsFileConfig{test_path + "/test_file", "/"};
        auto read_iov_handle = BuildIovHandle(1 << 20);
        auto write_iov_handle = BuildIovHandle(1 << 20);
        auto mock_usrbio_api = std::make_shared<MockHf3fsUsrbioApi>();
        client_ = std::make_shared<Hf3fsUsrbioClient>(cfg, read_iov_handle, write_iov_handle, mock_usrbio_api);
        ior_ = new ::hf3fs_ior();
    }
    void TearDown() override {
        delete ior_;
        read_mempool_.reset();
        write_mempool_.reset();
        client_.reset();
    }

protected:
    Hf3fsIovHandle BuildIovHandle(size_t size, char c = 'a') const {
        Hf3fsIovHandle handle;
        handle.iov_base = std::shared_ptr<uint8_t>((uint8_t *)malloc(size), [](void *ptr) { free(ptr); });
        std::memset(handle.iov_base.get(), c, size);
        handle.iov_size = size;
        handle.cuda_util = std::make_shared<Hf3fsCudaUtil>();
        auto mempool = std::make_shared<Hf3fsMempool>(handle.iov_base.get(), handle.iov_size, 0);
        EXPECT_TRUE(mempool->Init());
        handle.iov_mempool = mempool;
        return handle;
    }
    Hf3fsIorHandle BuildIorHandle() const {
        Hf3fsIorHandle handle;
        handle.ior = ior_;
        return handle;
    }

protected:
    std::shared_ptr<Hf3fsUsrbioClient> client_;
    std::shared_ptr<Hf3fsMempool> read_mempool_;
    std::shared_ptr<Hf3fsMempool> write_mempool_;
    ::hf3fs_ior *ior_{nullptr};
};

// ---------- ReadFrom3FS ----------
TEST_F(Hf3fsUsrbioClientTest, ReadFrom3FS_ReturnFalse_PrepIoFail) {
    auto api = static_cast<MockHf3fsUsrbioApi *>(client_->usrbio_api_.get());

    Hf3fsIovHandle iov = BuildIovHandle(1 << 20);
    iov.iov_block_size = 256;
    const int64_t len = 512;

    EXPECT_CALL(*api, Hf3fsIorCreate(testing::_, testing::_, len / iov.iov_block_size + 1, true, 0, 0, -1, 0))
        .WillOnce(testing::Return(0));
    auto handle = client_->InitIovIor(true, iov, len, iov.iov_block_size, 1);
    ASSERT_NE(handle, nullptr);

    EXPECT_CALL(*api,
                Hf3fsPrepIo(testing::_, testing::_, true, testing::_, testing::_, testing::_, testing::_, testing::_))
        .WillOnce(testing::Return(-5));

    std::vector<Hf3fsUsrbioClient::Segment> segs{{0, len}};
    EXPECT_FALSE(client_->ReadFrom3FS(handle, segs));

    EXPECT_CALL(*api, Hf3fsIorDestroy(::testing::NotNull())).Times(1);
    client_->ReleaseIovIor(handle);
}

// ---------- Read ----------
TEST_F(Hf3fsUsrbioClientTest, Read_ReturnTrue_ReadLenZero) {
    std::vector<Iov> iovs; // empty
    EXPECT_TRUE(client_->Read(iovs));

    std::vector<Iov> ignored{{MemoryType::CPU, nullptr, 10, true}};
    EXPECT_TRUE(client_->Read(ignored));
}

TEST_F(Hf3fsUsrbioClientTest, Read_ReturnFalse_ZeroSizeNonIgnored) {
    std::vector<Iov> iovs{{MemoryType::CPU, nullptr, 0, false}};
    EXPECT_FALSE(client_->Read(iovs));
}

TEST_F(Hf3fsUsrbioClientTest, Read_ReturnFalse_FileLengthUnknown) {
    // file not exist, FileLength() returns nullopt
    std::vector<Iov> iovs{{MemoryType::CPU, nullptr, 8, false}};
    EXPECT_FALSE(client_->Read(iovs));
}

TEST_F(Hf3fsUsrbioClientTest, Read_ReturnFalse_TotalLenExceedFile) {
    // simpler: just ensure file exists and small
    {
        std::ofstream f(client_->filepath_, std::ios::binary | std::ios::trunc);
        std::string blob(4, '\x01');
        f.write(blob.data(), blob.size());
    }
    std::vector<Iov> iovs{{MemoryType::CPU, nullptr, 8, false}}; // total_len 8 > file_len 4
    EXPECT_FALSE(client_->Read(iovs));
}

TEST_F(Hf3fsUsrbioClientTest, Read_ReturnFalse_OpenFail) {
    // path doesn't exist with read-only Open()
    std::vector<Iov> iovs{{MemoryType::CPU, nullptr, 8, false}};
    EXPECT_FALSE(client_->Read(iovs));
}

TEST_F(Hf3fsUsrbioClientTest, Read_ReturnTrue_Success) {
    // prepare a file with enough size
    {
        std::ofstream f(client_->filepath_, std::ios::binary | std::ios::trunc);
        std::string blob(32, '\xAB');
        f.write(blob.data(), blob.size());
    }

    auto api = static_cast<MockHf3fsUsrbioApi *>(client_->usrbio_api_.get());

    EXPECT_CALL(*api, Hf3fsRegFd(testing::_, testing::_)).WillOnce(testing::Return(0));
    EXPECT_CALL(
        *api, Hf3fsIorCreate(testing::_, testing::_, testing::_, true, testing::_, testing::_, testing::_, testing::_))
        .WillOnce(testing::Return(0));
    EXPECT_CALL(*api,
                Hf3fsPrepIo(testing::_, testing::_, true, testing::_, testing::_, testing::_, testing::_, testing::_))
        .WillRepeatedly(testing::Return(0));
    EXPECT_CALL(*api, Hf3fsSubmitIos(testing::NotNull())).WillRepeatedly(testing::Return(0));
    EXPECT_CALL(*api, Hf3fsWaitForIos(testing::_, testing::_, testing::_, testing::_, testing::_))
        .WillRepeatedly(testing::Invoke(
            [](const ::hf3fs_ior *, ::hf3fs_cqe *cqes, int cqec, int min_results, const struct timespec *) {
                for (int i = 0; i < min_results; ++i)
                    cqes[i].result = 1;
                return min_results;
            }));
    EXPECT_CALL(*api, Hf3fsDeregFd(testing::_)).Times(1);
    EXPECT_CALL(*api, Hf3fsIorDestroy(testing::NotNull())).Times(testing::AtLeast(1));

    std::vector<uint8_t> buf(16, 0);
    std::vector<Iov> iovs{{MemoryType::CPU, buf.data(), 16, false}};
    EXPECT_TRUE(client_->Read(iovs));
}

// ---------- DoRead ----------
TEST_F(Hf3fsUsrbioClientTest, DoRead_ReturnFalse_SegmentsEmpty) {
    std::vector<Iov> iovs{{MemoryType::CPU, nullptr, 10, true}};
    EXPECT_FALSE(client_->DoRead(iovs));
}

TEST_F(Hf3fsUsrbioClientTest, DoRead_ReturnFalse_InitIovIorFail) {
    // exhaust mempool to force alloc fail
    auto mempool = client_->read_iov_handle_.iov_mempool;
    ASSERT_NE(mempool, nullptr);
    void *hold = mempool->Alloc(mempool->FreeSize());
    ASSERT_NE(hold, nullptr);
    std::vector<Iov> iovs{{MemoryType::CPU, nullptr, 256, false}};
    EXPECT_FALSE(client_->DoRead(iovs));
    mempool->Free(hold);
}

TEST_F(Hf3fsUsrbioClientTest, DoRead_ReturnFalse_ReadFrom3FSFail) {
    auto api = static_cast<MockHf3fsUsrbioApi *>(client_->usrbio_api_.get());
    EXPECT_CALL(
        *api, Hf3fsIorCreate(testing::_, testing::_, testing::_, true, testing::_, testing::_, testing::_, testing::_))
        .WillOnce(testing::Return(0));
    EXPECT_CALL(*api,
                Hf3fsPrepIo(testing::_, testing::_, true, testing::_, testing::_, testing::_, testing::_, testing::_))
        .WillRepeatedly(testing::Return(0));
    EXPECT_CALL(*api, Hf3fsSubmitIos(testing::NotNull())).WillRepeatedly(testing::Return(0));
    EXPECT_CALL(*api, Hf3fsWaitForIos(testing::_, testing::_, testing::_, testing::_, testing::_))
        .WillRepeatedly(testing::Return(0));
    EXPECT_CALL(*api, Hf3fsIorDestroy(::testing::NotNull())).Times(1);

    std::vector<Iov> iovs{{MemoryType::CPU, nullptr, 128, false}};
    EXPECT_FALSE(client_->DoRead(iovs));
}

TEST_F(Hf3fsUsrbioClientTest, DoRead_ReturnTrue_Success) {
    auto api = static_cast<MockHf3fsUsrbioApi *>(client_->usrbio_api_.get());
    EXPECT_CALL(
        *api, Hf3fsIorCreate(testing::_, testing::_, testing::_, true, testing::_, testing::_, testing::_, testing::_))
        .WillOnce(testing::Return(0));
    EXPECT_CALL(*api,
                Hf3fsPrepIo(testing::_, testing::_, true, testing::_, testing::_, testing::_, testing::_, testing::_))
        .WillRepeatedly(testing::Return(0));
    EXPECT_CALL(*api, Hf3fsSubmitIos(testing::NotNull())).WillRepeatedly(testing::Return(0));
    EXPECT_CALL(*api, Hf3fsWaitForIos(testing::_, testing::_, testing::_, testing::_, testing::_))
        .WillRepeatedly(testing::Invoke(
            [](const ::hf3fs_ior *, ::hf3fs_cqe *cqes, int cqec, int min_results, const struct timespec *) {
                for (int i = 0; i < min_results; ++i)
                    cqes[i].result = 1;
                return min_results;
            }));

    std::vector<uint8_t> buf(64, 0);
    std::vector<Iov> iovs{{MemoryType::CPU, buf.data(), 64, false}};
    EXPECT_TRUE(client_->DoRead(iovs));
}

// ---------- ReadFrom3FS ----------
TEST_F(Hf3fsUsrbioClientTest, ReadFrom3FS_ReturnFalse_SubmitFail) {
    auto api = static_cast<MockHf3fsUsrbioApi *>(client_->usrbio_api_.get());

    Hf3fsIovHandle iov = BuildIovHandle(1 << 20);
    iov.iov_block_size = 256;
    const int64_t len = 512;

    EXPECT_CALL(*api, Hf3fsIorCreate(testing::_, testing::_, len / iov.iov_block_size + 1, true, 0, 0, -1, 0))
        .WillOnce(testing::Return(0));
    auto handle = client_->InitIovIor(true, iov, len, iov.iov_block_size, 1);
    ASSERT_NE(handle, nullptr);

    {
        testing::InSequence s;
        EXPECT_CALL(
            *api, Hf3fsPrepIo(testing::_, testing::_, true, testing::_, testing::_, testing::_, testing::_, testing::_))
            .WillOnce(testing::Return(0));
        EXPECT_CALL(
            *api, Hf3fsPrepIo(testing::_, testing::_, true, testing::_, testing::_, testing::_, testing::_, testing::_))
            .WillOnce(testing::Return(0));
        EXPECT_CALL(*api, Hf3fsSubmitIos(testing::NotNull())).WillOnce(testing::Return(-1));
    }

    std::vector<Hf3fsUsrbioClient::Segment> segs{{0, len}};
    EXPECT_FALSE(client_->ReadFrom3FS(handle, segs));

    EXPECT_CALL(*api, Hf3fsIorDestroy(::testing::NotNull())).Times(1);
    client_->ReleaseIovIor(handle);
}

TEST_F(Hf3fsUsrbioClientTest, ReadFrom3FS_ReturnFalse_WaitFail) {
    auto api = static_cast<MockHf3fsUsrbioApi *>(client_->usrbio_api_.get());

    Hf3fsIovHandle iov = BuildIovHandle(1 << 20);
    iov.iov_block_size = 128;
    const int64_t len = 384; // 3 preps then a submit

    EXPECT_CALL(*api, Hf3fsIorCreate(testing::_, testing::_, len / iov.iov_block_size + 1, true, 0, 0, -1, 0))
        .WillOnce(testing::Return(0));
    auto handle = client_->InitIovIor(true, iov, len, iov.iov_block_size, 1);
    ASSERT_NE(handle, nullptr);

    {
        testing::InSequence s;
        EXPECT_CALL(
            *api, Hf3fsPrepIo(testing::_, testing::_, true, testing::_, testing::_, testing::_, testing::_, testing::_))
            .WillOnce(testing::Return(0));
        EXPECT_CALL(
            *api, Hf3fsPrepIo(testing::_, testing::_, true, testing::_, testing::_, testing::_, testing::_, testing::_))
            .WillOnce(testing::Return(0));
        EXPECT_CALL(
            *api, Hf3fsPrepIo(testing::_, testing::_, true, testing::_, testing::_, testing::_, testing::_, testing::_))
            .WillOnce(testing::Return(0));
        EXPECT_CALL(*api, Hf3fsSubmitIos(testing::NotNull())).WillOnce(testing::Return(0));
        EXPECT_CALL(*api, Hf3fsWaitForIos(testing::_, testing::_, testing::_, testing::_, testing::_))
            .WillOnce(testing::Return(0)); // not enough completed
    }

    std::vector<Hf3fsUsrbioClient::Segment> segs{{0, len}};
    EXPECT_FALSE(client_->ReadFrom3FS(handle, segs));

    EXPECT_CALL(*api, Hf3fsIorDestroy(::testing::NotNull())).Times(1);
    client_->ReleaseIovIor(handle);
}

TEST_F(Hf3fsUsrbioClientTest, ReadFrom3FS_ReturnTrue_Success) {
    auto api = static_cast<MockHf3fsUsrbioApi *>(client_->usrbio_api_.get());

    Hf3fsIovHandle iov = BuildIovHandle(1 << 20);
    iov.iov_block_size = 128;
    const int64_t len = 384; // 3 preps

    EXPECT_CALL(*api, Hf3fsIorCreate(testing::_, testing::_, len / iov.iov_block_size + 1, true, 0, 0, -1, 0))
        .WillOnce(testing::Return(0));
    auto handle = client_->InitIovIor(true, iov, len, iov.iov_block_size, 1);
    ASSERT_NE(handle, nullptr);

    {
        testing::InSequence s;
        EXPECT_CALL(
            *api, Hf3fsPrepIo(testing::_, testing::_, true, testing::_, testing::_, testing::_, testing::_, testing::_))
            .WillOnce(testing::Return(0));
        EXPECT_CALL(
            *api, Hf3fsPrepIo(testing::_, testing::_, true, testing::_, testing::_, testing::_, testing::_, testing::_))
            .WillOnce(testing::Return(0));
        EXPECT_CALL(
            *api, Hf3fsPrepIo(testing::_, testing::_, true, testing::_, testing::_, testing::_, testing::_, testing::_))
            .WillOnce(testing::Return(0));
        EXPECT_CALL(*api, Hf3fsSubmitIos(testing::NotNull())).WillOnce(testing::Return(0));
        EXPECT_CALL(*api, Hf3fsWaitForIos(testing::_, testing::_, testing::_, testing::_, testing::_))
            .WillOnce(testing::Invoke(
                [](const ::hf3fs_ior *, ::hf3fs_cqe *cqes, int cqec, int min_results, const struct timespec *) {
                    for (int i = 0; i < min_results; ++i)
                        cqes[i].result = 1;
                    return min_results;
                }));
    }

    std::vector<Hf3fsUsrbioClient::Segment> segs{{0, len}};
    EXPECT_TRUE(client_->ReadFrom3FS(handle, segs));

    EXPECT_CALL(*api, Hf3fsIorDestroy(::testing::NotNull())).Times(1);
    client_->ReleaseIovIor(handle);
}

// ---------- Write ----------
TEST_F(Hf3fsUsrbioClientTest, Write_ReturnTrue_WriteLenZero) {
    std::vector<Iov> iovs; // empty
    EXPECT_TRUE(client_->Write(iovs));
    // all ignored also returns true
    std::vector<Iov> ignored{{MemoryType::CPU, nullptr, 10, true}, {MemoryType::GPU, nullptr, 20, true}};
    EXPECT_TRUE(client_->Write(ignored));
}

TEST_F(Hf3fsUsrbioClientTest, Write_ReturnFalse_OpenFail) {
    auto api = static_cast<MockHf3fsUsrbioApi *>(client_->usrbio_api_.get());
    EXPECT_CALL(*api, Hf3fsRegFd(testing::_, testing::_)).WillOnce(testing::Return(1));
    std::vector<Iov> iovs{{MemoryType::CPU, nullptr, 16, false}};
    EXPECT_FALSE(client_->Write(iovs));
}

TEST_F(Hf3fsUsrbioClientTest, Write_ReturnFalse_DoWriteFail) {
    auto api = static_cast<MockHf3fsUsrbioApi *>(client_->usrbio_api_.get());

    {
        testing::InSequence s;
        EXPECT_CALL(*api, Hf3fsRegFd(testing::_, testing::_)).WillOnce(testing::Return(0));
        EXPECT_CALL(*api,
                    Hf3fsIorCreate(
                        testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
            .WillOnce(testing::Return(0));
        EXPECT_CALL(
            *api,
            Hf3fsPrepIo(testing::_, testing::_, false, testing::_, testing::_, testing::_, testing::_, testing::_))
            .WillRepeatedly(testing::Return(0));
        EXPECT_CALL(*api, Hf3fsSubmitIos(testing::NotNull())).WillRepeatedly(testing::Return(0));
        EXPECT_CALL(*api, Hf3fsWaitForIos(testing::_, testing::_, testing::_, testing::_, testing::_))
            .WillRepeatedly(testing::Return(0)); // WaitIos -> false
    }

    // ReleaseIovIor() happens before Close(), so IorDestroy is expected before DeregFd; allow any order
    EXPECT_CALL(*api, Hf3fsIorDestroy(testing::NotNull())).Times(testing::AtLeast(1));
    EXPECT_CALL(*api, Hf3fsDeregFd(testing::_)).Times(1);

    auto buffer1 = std::shared_ptr<uint8_t>((uint8_t *)malloc(64), [](void *ptr) { free(ptr); });
    auto buffer2 = std::shared_ptr<uint8_t>((uint8_t *)malloc(64), [](void *ptr) { free(ptr); });
    std::vector<Iov> iovs{{MemoryType::CPU, buffer1.get(), 64, false}, {MemoryType::CPU, buffer2.get(), 64, false}};
    EXPECT_FALSE(client_->Write(iovs));
}

TEST_F(Hf3fsUsrbioClientTest, Write_ReturnTrue_Success) {
    auto api = static_cast<MockHf3fsUsrbioApi *>(client_->usrbio_api_.get());

    {
        testing::InSequence s;
        EXPECT_CALL(*api, Hf3fsRegFd(testing::_, testing::_)).WillOnce(testing::Return(0));
        EXPECT_CALL(*api,
                    Hf3fsIorCreate(
                        testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
            .WillOnce(testing::Return(0));
        EXPECT_CALL(
            *api,
            Hf3fsPrepIo(testing::_, testing::_, false, testing::_, testing::_, testing::_, testing::_, testing::_))
            .WillRepeatedly(testing::Return(0));
        EXPECT_CALL(*api, Hf3fsSubmitIos(testing::NotNull())).WillRepeatedly(testing::Return(0));
        EXPECT_CALL(*api, Hf3fsWaitForIos(testing::_, testing::_, testing::_, testing::_, testing::_))
            .WillRepeatedly(testing::Invoke(
                [](const ::hf3fs_ior *, ::hf3fs_cqe *cqes, int cqec, int min_results, const struct timespec *) {
                    for (int i = 0; i < min_results; ++i)
                        cqes[i].result = 1;
                    return min_results;
                }));
    }

    EXPECT_CALL(*api, Hf3fsIorDestroy(testing::NotNull())).Times(testing::AtLeast(1));
    EXPECT_CALL(*api, Hf3fsDeregFd(testing::_)).Times(1);

    auto buffer1 = std::shared_ptr<uint8_t>((uint8_t *)malloc(32), [](void *ptr) { free(ptr); });
    auto buffer2 = std::shared_ptr<uint8_t>((uint8_t *)malloc(16), [](void *ptr) { free(ptr); });
    std::vector<Iov> iovs{{MemoryType::CPU, buffer1.get(), 32, false}, {MemoryType::CPU, buffer2.get(), 16, false}};
    EXPECT_TRUE(client_->Write(iovs));
}

// ---------- DoWrite ----------
TEST_F(Hf3fsUsrbioClientTest, DoWrite_ReturnFalse_SegmentsEmpty) {
    // all ignored -> segments empty
    std::vector<Iov> iovs{
        {MemoryType::CPU, nullptr, 10, true},
        {MemoryType::GPU, nullptr, 20, true},
    };
    EXPECT_FALSE(client_->DoWrite(iovs));
}

TEST_F(Hf3fsUsrbioClientTest, DoWrite_ReturnFalse_InitIovIorFail_MempoolAlloc) {
    // exhaust the same mempool used by client's write_iov_handle_
    auto mempool = client_->write_iov_handle_.iov_mempool;
    ASSERT_NE(mempool, nullptr);
    void *hold = mempool->Alloc(mempool->FreeSize());
    ASSERT_NE(hold, nullptr);

    auto buf = std::shared_ptr<uint8_t>((uint8_t *)malloc(128), [](void *p) { free(p); });
    std::vector<Iov> iovs{{MemoryType::CPU, buf.get(), 128, false}};
    EXPECT_FALSE(client_->DoWrite(iovs));

    mempool->Free(hold);
}

TEST_F(Hf3fsUsrbioClientTest, DoWrite_ReturnFalse_WriteTo3FSFail_Wait) {
    auto api = static_cast<MockHf3fsUsrbioApi *>(client_->usrbio_api_.get());

    EXPECT_CALL(
        *api, Hf3fsIorCreate(testing::_, testing::_, testing::_, false, testing::_, testing::_, testing::_, testing::_))
        .WillOnce(testing::Return(0));
    EXPECT_CALL(*api,
                Hf3fsPrepIo(testing::_, testing::_, false, testing::_, testing::_, testing::_, testing::_, testing::_))
        .WillRepeatedly(testing::Return(0));
    EXPECT_CALL(*api, Hf3fsSubmitIos(testing::NotNull())).WillRepeatedly(testing::Return(0));
    EXPECT_CALL(*api, Hf3fsWaitForIos(testing::_, testing::_, testing::_, testing::_, testing::_))
        .WillRepeatedly(testing::Return(0)); // triggers WaitIos false
    EXPECT_CALL(*api, Hf3fsIorDestroy(::testing::NotNull())).Times(1);

    auto buffer1 = std::shared_ptr<uint8_t>((uint8_t *)malloc(64), [](void *ptr) { free(ptr); });
    auto buffer2 = std::shared_ptr<uint8_t>((uint8_t *)malloc(64), [](void *ptr) { free(ptr); });
    std::vector<Iov> iovs{
        {MemoryType::CPU, buffer1.get(), 64, false},
        {MemoryType::CPU, buffer2.get(), 64, false},
    };
    EXPECT_FALSE(client_->DoWrite(iovs));
}

TEST_F(Hf3fsUsrbioClientTest, DoWrite_ReturnTrue_Success) {
    auto api = static_cast<MockHf3fsUsrbioApi *>(client_->usrbio_api_.get());

    EXPECT_CALL(
        *api, Hf3fsIorCreate(testing::_, testing::_, testing::_, false, testing::_, testing::_, testing::_, testing::_))
        .WillOnce(testing::Return(0));
    EXPECT_CALL(*api,
                Hf3fsPrepIo(testing::_, testing::_, false, testing::_, testing::_, testing::_, testing::_, testing::_))
        .WillRepeatedly(testing::Return(0));
    EXPECT_CALL(*api, Hf3fsSubmitIos(testing::NotNull())).WillRepeatedly(testing::Return(0));
    EXPECT_CALL(*api, Hf3fsWaitForIos(testing::_, testing::_, testing::_, testing::_, testing::_))
        .WillRepeatedly(testing::Invoke(
            [](const ::hf3fs_ior *, ::hf3fs_cqe *cqes, int cqec, int min_results, const struct timespec *) {
                for (int i = 0; i < min_results; ++i)
                    cqes[i].result = 1;
                return min_results;
            }));
    EXPECT_CALL(*api, Hf3fsIorDestroy(::testing::NotNull())).Times(1);

    auto buffer1 = std::shared_ptr<uint8_t>((uint8_t *)malloc(32), [](void *ptr) { free(ptr); });
    auto buffer2 = std::shared_ptr<uint8_t>((uint8_t *)malloc(32), [](void *ptr) { free(ptr); });

#ifdef USING_CUDA
    void *buffer3 = nullptr;
    ASSERT_EQ(cudaMalloc(&buffer3, 16), cudaSuccess);
    ASSERT_NE(buffer3, nullptr);
    ASSERT_EQ(cudaMemset(buffer3, 0, 16), cudaSuccess);
    std::shared_ptr<uint8_t> buffer3_ptr(reinterpret_cast<uint8_t *>(buffer3), [](void *ptr) { cudaFree(ptr); });
#endif

    std::vector<Iov> iovs{
        {MemoryType::CPU, buffer1.get(), 32, false},
        {MemoryType::CPU, buffer2.get(), 32, false},
#ifdef USING_CUDA
        {MemoryType::GPU, buffer3, 16, false}, // type change -> multiple segments
#endif
    };
    EXPECT_TRUE(client_->DoWrite(iovs));
    EXPECT_EQ(client_->write_iov_handle_.iov_mempool->AllocatedSize(), 0); // released
}

// ---------- WriteTo3FS ----------
TEST_F(Hf3fsUsrbioClientTest, WriteTo3FS_ReturnFalse_PrepIoFail) {
    auto api = static_cast<MockHf3fsUsrbioApi *>(client_->usrbio_api_.get());

    Hf3fsIovHandle iov_handle = BuildIovHandle(1 << 20);
    iov_handle.iov_block_size = 256;
    const int64_t len = 512; // 2 steps with block size 256

    EXPECT_CALL(*api, Hf3fsIorCreate(testing::_, testing::_, len / iov_handle.iov_block_size + 1, false, 0, 0, -1, 0))
        .WillOnce(testing::Return(0));

    auto handle = client_->InitIovIor(false, iov_handle, len, iov_handle.iov_block_size, 1);
    ASSERT_NE(handle, nullptr);

    EXPECT_CALL(*api,
                Hf3fsPrepIo(testing::_, testing::_, false, testing::_, testing::_, testing::_, testing::_, testing::_))
        .WillOnce(testing::Return(-6));

    std::vector<Hf3fsUsrbioClient::Segment> segs{{0, len}};
    EXPECT_FALSE(client_->WriteTo3FS(handle, segs));

    EXPECT_CALL(*api, Hf3fsIorDestroy(::testing::NotNull())).Times(1);
    client_->ReleaseIovIor(handle);
}

TEST_F(Hf3fsUsrbioClientTest, WriteTo3FS_ReturnFalse_SubmitFail) {
    auto api = static_cast<MockHf3fsUsrbioApi *>(client_->usrbio_api_.get());

    Hf3fsIovHandle iov_handle = BuildIovHandle(1 << 20);
    iov_handle.iov_block_size = 256;
    const int64_t len = 512; // 2 prep then submit

    EXPECT_CALL(*api, Hf3fsIorCreate(testing::_, testing::_, len / iov_handle.iov_block_size + 1, false, 0, 0, -1, 0))
        .WillOnce(testing::Return(0));
    auto handle = client_->InitIovIor(false, iov_handle, len, iov_handle.iov_block_size, 1);
    ASSERT_NE(handle, nullptr);

    {
        testing::InSequence s;
        EXPECT_CALL(
            *api,
            Hf3fsPrepIo(testing::_, testing::_, false, testing::_, testing::_, testing::_, testing::_, testing::_))
            .WillOnce(testing::Return(0));
        EXPECT_CALL(
            *api,
            Hf3fsPrepIo(testing::_, testing::_, false, testing::_, testing::_, testing::_, testing::_, testing::_))
            .WillOnce(testing::Return(0));
        EXPECT_CALL(*api, Hf3fsSubmitIos(testing::NotNull())).WillOnce(testing::Return(-1));
    }

    std::vector<Hf3fsUsrbioClient::Segment> segs{{0, len}};
    EXPECT_FALSE(client_->WriteTo3FS(handle, segs));

    EXPECT_CALL(*api, Hf3fsIorDestroy(::testing::NotNull())).Times(1);
    client_->ReleaseIovIor(handle);
}

TEST_F(Hf3fsUsrbioClientTest, WriteTo3FS_ReturnFalse_WaitIoFail) {
    auto api = static_cast<MockHf3fsUsrbioApi *>(client_->usrbio_api_.get());

    Hf3fsIovHandle iov_handle = BuildIovHandle(1 << 20);
    iov_handle.iov_block_size = 256;
    const int64_t len = 512; // 2 prep then submit

    EXPECT_CALL(*api, Hf3fsIorCreate(testing::_, testing::_, len / iov_handle.iov_block_size + 1, false, 0, 0, -1, 0))
        .WillOnce(testing::Return(0));
    auto handle = client_->InitIovIor(false, iov_handle, len, iov_handle.iov_block_size, 1);
    ASSERT_NE(handle, nullptr);

    {
        testing::InSequence s;
        EXPECT_CALL(
            *api,
            Hf3fsPrepIo(testing::_, testing::_, false, testing::_, testing::_, testing::_, testing::_, testing::_))
            .WillOnce(testing::Return(0));
        EXPECT_CALL(
            *api,
            Hf3fsPrepIo(testing::_, testing::_, false, testing::_, testing::_, testing::_, testing::_, testing::_))
            .WillOnce(testing::Return(0));
        EXPECT_CALL(*api, Hf3fsSubmitIos(testing::NotNull())).WillOnce(testing::Return(0));
        EXPECT_CALL(*api, Hf3fsWaitForIos(testing::_, testing::_, testing::_, testing::_, testing::_))
            .WillOnce(testing::Return(0)); // partial or zero -> WaitIos false
    }

    std::vector<Hf3fsUsrbioClient::Segment> segs{{0, len}};
    EXPECT_FALSE(client_->WriteTo3FS(handle, segs));

    EXPECT_CALL(*api, Hf3fsIorDestroy(::testing::NotNull())).Times(1);
    client_->ReleaseIovIor(handle);
}

TEST_F(Hf3fsUsrbioClientTest, WriteTo3FS_ReturnTrue_Success_WithBatching) {
    auto api = static_cast<MockHf3fsUsrbioApi *>(client_->usrbio_api_.get());

    Hf3fsIovHandle iov_handle = BuildIovHandle(1 << 20);
    // Force frequent submit: ior_entries computed as len/block + base(=1) -> small
    iov_handle.iov_block_size = 128;
    const int64_t len = 384; // 3 steps with block size 128

    EXPECT_CALL(*api, Hf3fsIorCreate(testing::_, testing::_, len / iov_handle.iov_block_size + 1, false, 0, 0, -1, 0))
        .WillOnce(testing::Return(0));
    auto handle = client_->InitIovIor(false, iov_handle, len, iov_handle.iov_block_size, 1);
    ASSERT_NE(handle, nullptr);

    {
        testing::InSequence s;
        EXPECT_CALL(
            *api,
            Hf3fsPrepIo(testing::_, testing::_, false, testing::_, testing::_, testing::_, testing::_, testing::_))
            .WillOnce(testing::Return(0));
        EXPECT_CALL(
            *api,
            Hf3fsPrepIo(testing::_, testing::_, false, testing::_, testing::_, testing::_, testing::_, testing::_))
            .WillOnce(testing::Return(0));
        EXPECT_CALL(
            *api,
            Hf3fsPrepIo(testing::_, testing::_, false, testing::_, testing::_, testing::_, testing::_, testing::_))
            .WillOnce(testing::Return(0));
        EXPECT_CALL(*api, Hf3fsSubmitIos(testing::NotNull())).WillOnce(testing::Return(0));
        EXPECT_CALL(*api, Hf3fsWaitForIos(testing::_, testing::_, testing::_, testing::_, testing::_))
            .WillOnce(testing::Invoke(
                [](const ::hf3fs_ior *, ::hf3fs_cqe *cqes, int cqec, int min_results, const struct timespec *) {
                    for (int i = 0; i < min_results; ++i)
                        cqes[i].result = 1;
                    return min_results; // completed = submitted
                }));
    }

    std::vector<Hf3fsUsrbioClient::Segment> segs{{0, len}};
    EXPECT_TRUE(client_->WriteTo3FS(handle, segs));

    EXPECT_CALL(*api, Hf3fsIorDestroy(::testing::NotNull())).Times(1);
    client_->ReleaseIovIor(handle);
}

// ---------- WaitIos ----------
TEST_F(Hf3fsUsrbioClientTest, WaitIos_ReturnFalse_NullIor) {
    Hf3fsIorHandle ior_handle;
    EXPECT_FALSE(client_->WaitIos(ior_handle, 2));
}

TEST_F(Hf3fsUsrbioClientTest, WaitIos_ReturnFalse_WaitNegative) {
    auto api = static_cast<MockHf3fsUsrbioApi *>(client_->usrbio_api_.get());
    EXPECT_CALL(*api, Hf3fsWaitForIos(testing::_, testing::_, testing::_, testing::_, testing::_))
        .WillOnce(testing::Return(-1));
    Hf3fsIorHandle ior_handle = BuildIorHandle();
    EXPECT_FALSE(client_->WaitIos(ior_handle, 2));
}

TEST_F(Hf3fsUsrbioClientTest, WaitIos_ReturnFalse_CqeResultNegative) {
    auto api = static_cast<MockHf3fsUsrbioApi *>(client_->usrbio_api_.get());

    EXPECT_CALL(*api, Hf3fsWaitForIos(testing::_, testing::_, testing::_, testing::_, testing::_))
        .WillOnce(testing::Invoke(
            [](const ::hf3fs_ior *, ::hf3fs_cqe *cqes, int cqec, int min_results, const struct timespec *) {
                EXPECT_EQ(cqec, min_results);
                if (cqec > 0)
                    cqes[0].result = 0;
                if (cqec > 1)
                    cqes[1].result = -1; // inject error
                return min_results;      // equals submit count
            }));
    Hf3fsIorHandle ior_handle = BuildIorHandle();
    EXPECT_FALSE(client_->WaitIos(ior_handle, 2));
}

TEST_F(Hf3fsUsrbioClientTest, WaitIos_ReturnFalse_PartialDone) {
    auto api = static_cast<MockHf3fsUsrbioApi *>(client_->usrbio_api_.get());
    EXPECT_CALL(*api, Hf3fsWaitForIos(testing::_, testing::_, testing::_, testing::_, testing::_))
        .WillOnce(testing::Invoke(
            [](const ::hf3fs_ior *, ::hf3fs_cqe *cqes, int cqec, int min_results, const struct timespec *) {
                EXPECT_EQ(cqec, min_results);
                for (int i = 0; i < cqec; ++i) {
                    cqes[i].result = 0;
                }
                return cqec - 1; // less than submit count
            }));
    Hf3fsIorHandle ior_handle = BuildIorHandle();
    EXPECT_FALSE(client_->WaitIos(ior_handle, 2));
}

TEST_F(Hf3fsUsrbioClientTest, WaitIos_ReturnTrue_AllDoneAndNonNegative) {
    auto api = static_cast<MockHf3fsUsrbioApi *>(client_->usrbio_api_.get());
    EXPECT_CALL(*api, Hf3fsWaitForIos(testing::_, testing::_, testing::_, testing::_, testing::_))
        .WillOnce(testing::Invoke(
            [](const ::hf3fs_ior *, ::hf3fs_cqe *cqes, int cqec, int min_results, const struct timespec *) {
                for (int i = 0; i < cqec; ++i) {
                    cqes[i].result = 0;
                }
                return cqec; // equals submit count
            }));
    Hf3fsIorHandle ior_handle = BuildIorHandle();
    EXPECT_TRUE(client_->WaitIos(ior_handle, 3));
}

// ---------- BuildContiguousSegments ----------
TEST_F(Hf3fsUsrbioClientTest, BuildContiguousSegments_ReturnEmpty_AllIgnoredOrZero) {
    std::vector<Iov> iovs;
    iovs.push_back(Iov{MemoryType::CPU, nullptr, 0, true});
    iovs.push_back(Iov{MemoryType::CPU, nullptr, 100, true});
    iovs.push_back(Iov{MemoryType::CPU, nullptr, 200, true});
    auto segs = client_->BuildContiguousSegments(iovs);
    EXPECT_TRUE(segs.empty());
}

TEST_F(Hf3fsUsrbioClientTest, BuildContiguousSegments_ReturnMerged_WithIgnores) {
    std::vector<Iov> iovs{
        {MemoryType::CPU, nullptr, 10, true},
        {MemoryType::CPU, nullptr, 10, true},
        {MemoryType::CPU, nullptr, 5, false},
        {MemoryType::CPU, nullptr, 7, false},
        {MemoryType::CPU, nullptr, 2, true},
        {MemoryType::CPU, nullptr, 3, false},
    };
    auto segs = client_->BuildContiguousSegments(iovs);
    ASSERT_EQ(segs.size(), 2u);
    EXPECT_EQ(segs[0].offset, 20);
    EXPECT_EQ(segs[0].len, 12);
    EXPECT_EQ(segs[1].offset, 34);
    EXPECT_EQ(segs[1].len, 3);
}

TEST_F(Hf3fsUsrbioClientTest, BuildContiguousSegments_ReturnSingle_AllData) {
    std::vector<Iov> iovs{
        {MemoryType::CPU, nullptr, 5, false},
        {MemoryType::CPU, nullptr, 7, false},
        {MemoryType::CPU, nullptr, 3, false},
    };
    auto segs = client_->BuildContiguousSegments(iovs);
    ASSERT_EQ(segs.size(), 1u);
    EXPECT_EQ(segs[0].offset, 0);
    EXPECT_EQ(segs[0].len, 15);
}

TEST_F(Hf3fsUsrbioClientTest, BuildContiguousSegments_ReturnSplit_OnTypeChange) {
    std::vector<Iov> iovs{
        {MemoryType::CPU, nullptr, 4, false},
        {MemoryType::CPU, nullptr, 6, false},
        {MemoryType::GPU, nullptr, 3, false},
        {MemoryType::GPU, nullptr, 5, false},
        {MemoryType::CPU, nullptr, 2, false},
    };
    auto segs = client_->BuildContiguousSegments(iovs);
    ASSERT_EQ(segs.size(), 3u);
    EXPECT_EQ(segs[0].offset, 0);
    EXPECT_EQ(segs[0].len, 10);
    EXPECT_EQ(segs[1].offset, 10);
    EXPECT_EQ(segs[1].len, 8);
    EXPECT_EQ(segs[2].offset, 18);
    EXPECT_EQ(segs[2].len, 2);
}

TEST_F(Hf3fsUsrbioClientTest, BuildContiguousSegments_ReturnCorrect_LeadingTrailingIgnores) {
    std::vector<Iov> iovs{
        {MemoryType::CPU, nullptr, 7, true}, // leading ignore
        {MemoryType::CPU, nullptr, 5, false},
        {MemoryType::CPU, nullptr, 5, false},
        {MemoryType::CPU, nullptr, 3, true}, // middle ignore
        {MemoryType::CPU, nullptr, 2, false},
        {MemoryType::CPU, nullptr, 4, true}, // trailing ignore
    };
    auto segs = client_->BuildContiguousSegments(iovs);
    ASSERT_EQ(segs.size(), 2u);
    EXPECT_EQ(segs[0].offset, 7);
    EXPECT_EQ(segs[0].len, 10);
    EXPECT_EQ(segs[1].offset, 20);
    EXPECT_EQ(segs[1].len, 2);
}

TEST_F(Hf3fsUsrbioClientTest, BuildContiguousSegments_ReturnSkipZero_NonIgnoredZeroSize) {
    std::vector<Iov> iovs{
        {MemoryType::CPU, nullptr, 0, false}, // zero-size data
        {MemoryType::CPU, nullptr, 4, false},
        {MemoryType::CPU, nullptr, 0, false}, // zero-size data
        {MemoryType::GPU, nullptr, 0, false}, // zero-size data with type change
        {MemoryType::GPU, nullptr, 3, false},
        {MemoryType::CPU, nullptr, 0, false}, // zero-size data with type change
        {MemoryType::CPU, nullptr, 0, true},  // ignored zero-size (skipped)
        {MemoryType::CPU, nullptr, 2, true},  // ignored positive-size (advances offset & flush)
        {MemoryType::CPU, nullptr, 1, false},
    };
    auto segs = client_->BuildContiguousSegments(iovs);
    ASSERT_EQ(segs.size(), 3u);
    EXPECT_EQ(segs[0].offset, 0);
    EXPECT_EQ(segs[0].len, 4);
    EXPECT_EQ(segs[1].offset, 4);
    EXPECT_EQ(segs[1].len, 3);
    EXPECT_EQ(segs[2].offset, 9);
    EXPECT_EQ(segs[2].len, 1);
}

TEST_F(Hf3fsUsrbioClientTest, BuildContiguousSegments_ReturnSplit_MixedTypeAndIgnores) {
    std::vector<Iov> iovs{
        {MemoryType::CPU, nullptr, 3, false},
        {MemoryType::CPU, nullptr, 2, true}, // flush and advance
        {MemoryType::GPU, nullptr, 2, false},
        {MemoryType::GPU, nullptr, 0, true},  // ignored zero-size (skip)
        {MemoryType::GPU, nullptr, 1, false}, // extend GPU segment
        {MemoryType::CPU, nullptr, 4, false}, // type change -> new segment
    };
    auto segs = client_->BuildContiguousSegments(iovs);
    ASSERT_EQ(segs.size(), 3u);
    EXPECT_EQ(segs[0].offset, 0);
    EXPECT_EQ(segs[0].len, 3);
    EXPECT_EQ(segs[1].offset, 5);
    EXPECT_EQ(segs[1].len, 3);
    EXPECT_EQ(segs[2].offset, 8);
    EXPECT_EQ(segs[2].len, 4);
}

// ---------- InitIovIor ----------
TEST_F(Hf3fsUsrbioClientTest, InitIovIor_ReturnNull_MempoolNull) {
    auto iov_handle = BuildIovHandle(1 << 20);
    iov_handle.iov_mempool = nullptr;
    int64_t len = 1 << 10;
    int32_t size_per_io = 1024;
    int32_t base_ior_entries = 1;
    auto handle = client_->InitIovIor(true, iov_handle, len, size_per_io, base_ior_entries);
    EXPECT_EQ(handle, nullptr);
}

TEST_F(Hf3fsUsrbioClientTest, InitIovIor_ReturnNull_MempoolAllocFail) {
    auto iov_handle = BuildIovHandle(1 << 10);
    int64_t len = 1 << 20; // large enough to force alloc fail
    int32_t size_per_io = 1024;
    int32_t base_ior_entries = 1;

    auto handle = client_->InitIovIor(true, iov_handle, len, size_per_io, base_ior_entries);
    EXPECT_EQ(handle, nullptr);
}

TEST_F(Hf3fsUsrbioClientTest, InitIovIor_ReturnNull_CreateIorFail) {
    auto api = static_cast<MockHf3fsUsrbioApi *>(client_->usrbio_api_.get());

    bool for_read = true;
    auto iov_handle = BuildIovHandle(1 << 20);
    int64_t len = 1 << 10;
    int32_t size_per_io = 1024;
    int32_t base_ior_entries = 1;

    EXPECT_CALL(*api,
                Hf3fsIorCreate(testing::_, testing::_, len / size_per_io + base_ior_entries, for_read, 0, 0, -1, 0))
        .WillOnce(testing::Return(-5));
    auto handle = client_->InitIovIor(for_read, iov_handle, len, size_per_io, base_ior_entries);
    EXPECT_EQ(handle, nullptr);
    EXPECT_EQ(iov_handle.iov_mempool->AllocatedSize(), 0);
}

TEST_F(Hf3fsUsrbioClientTest, InitIovIor_Success) {
    auto api = static_cast<MockHf3fsUsrbioApi *>(client_->usrbio_api_.get());

    bool for_read = true;
    auto iov_handle = BuildIovHandle(1 << 20);
    int64_t len = 1 << 10;
    int32_t size_per_io = 1024;
    int32_t base_ior_entries = 1;

    EXPECT_CALL(*api,
                Hf3fsIorCreate(testing::_, testing::_, len / size_per_io + base_ior_entries, for_read, 0, 0, -1, 0))
        .WillOnce(testing::Return(0));
    auto handle = client_->InitIovIor(for_read, iov_handle, len, size_per_io, base_ior_entries);

    EXPECT_NE(handle, nullptr);
    EXPECT_EQ(iov_handle.iov_mempool->AllocatedSize(), len);
    EXPECT_EQ(iov_handle.iov_mempool->allocated_blocks_[handle->iov_handle.iov_base.get()], len);
    EXPECT_TRUE(iov_handle.iov_mempool->allocated_blocks_.count(handle->iov_handle.iov_base.get()) != 0);
    EXPECT_EQ(handle->iov_handle.iov_size, len);
    EXPECT_NE(handle->ior_handle.ior, nullptr);

    EXPECT_CALL(*api, Hf3fsIorDestroy(::testing::NotNull())).Times(1);
    client_->ReleaseIovIor(handle);
}

// ---------- ReleaseIovIor ----------
TEST_F(Hf3fsUsrbioClientTest, ReleaseIovIor_ReturnVoid_Nullptr) {
    client_->ReleaseIovIor(nullptr);
    SUCCEED();
}

TEST_F(Hf3fsUsrbioClientTest, ReleaseIovIor_ReturnVoid_ValidHandleWithIovBase) {
    auto api = static_cast<MockHf3fsUsrbioApi *>(client_->usrbio_api_.get());

    auto handle = std::make_shared<Hf3fsHandle>();
    handle->iov_handle = BuildIovHandle(1 << 20);

    EXPECT_CALL(
        *api,
        Hf3fsIorCreate(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
        .WillOnce(testing::Return(0));
    struct hf3fs_ior *ior = nullptr;
    EXPECT_TRUE(client_->CreateIor(ior, true, 1024, 1, 100));
    EXPECT_NE(ior, nullptr);
    handle->ior_handle = Hf3fsIorHandle{ior, 1024, 1, 100};

    EXPECT_CALL(*api, Hf3fsIorDestroy(ior)).Times(1);
    client_->ReleaseIovIor(handle);
    EXPECT_EQ(handle->iov_handle.iov_base, nullptr);
    EXPECT_EQ(handle->ior_handle.ior, nullptr);
}

// ---------- CreateIor ----------
TEST_F(Hf3fsUsrbioClientTest, CreateIor_ReturnFalse_EmptyMountpoint) {
    const auto read_iov_handle = BuildIovHandle(1 << 20);
    const auto write_iov_handle = BuildIovHandle(1 << 20);
    Hf3fsFileConfig cfg{"/tmp/x", ""}; // 空 mountpoint
    auto client = std::make_shared<Hf3fsUsrbioClient>(cfg, read_iov_handle, write_iov_handle);

    ::hf3fs_ior *ior = nullptr;
    EXPECT_FALSE(client->CreateIor(ior, true, 4, 1, 100));
    EXPECT_EQ(ior, nullptr);
}

TEST_F(Hf3fsUsrbioClientTest, CreateIor_ReturnFalse_CreateIorFail) {
    auto mock_api = static_cast<MockHf3fsUsrbioApi *>(client_->usrbio_api_.get());

    struct hf3fs_ior *ior = nullptr;
    bool read = true;
    int ior_entries = 4;
    int ior_io_depth = 1;
    int ior_timeout_ms = 100;

    EXPECT_CALL(
        *mock_api,
        Hf3fsIorCreate(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
        .WillOnce(Invoke([&ior, read, ior_entries, ior_io_depth, ior_timeout_ms](struct hf3fs_ior *actual_ior,
                                                                                 const char *mount,
                                                                                 int entries,
                                                                                 bool for_read,
                                                                                 int io_depth,
                                                                                 int timeout,
                                                                                 int numa,
                                                                                 uint64_t flags) {
            EXPECT_EQ(actual_ior, ior);
            EXPECT_EQ(std::strcmp(mount, "/"), 0);
            EXPECT_EQ(entries, ior_entries);
            EXPECT_EQ(for_read, read);
            EXPECT_EQ(io_depth, ior_io_depth);
            EXPECT_EQ(timeout, ior_timeout_ms);
            EXPECT_EQ(numa, -1);
            EXPECT_EQ(flags, 0);
            return -1;
        }));
    EXPECT_FALSE(client_->CreateIor(ior, read, ior_entries, ior_io_depth, ior_timeout_ms));
    EXPECT_EQ(ior, nullptr);
}

TEST_F(Hf3fsUsrbioClientTest, CreateIor_ReturnTrue_CreateIorSuccess) {
    auto mock_api = static_cast<MockHf3fsUsrbioApi *>(client_->usrbio_api_.get());

    struct hf3fs_ior *ior = nullptr;
    bool read = true;
    int ior_entries = 4;
    int ior_io_depth = 1;
    int ior_timeout_ms = 100;

    EXPECT_CALL(
        *mock_api,
        Hf3fsIorCreate(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
        .WillOnce(Invoke([&ior, read, ior_entries, ior_io_depth, ior_timeout_ms](struct hf3fs_ior *actual_ior,
                                                                                 const char *mount,
                                                                                 int entries,
                                                                                 bool for_read,
                                                                                 int io_depth,
                                                                                 int timeout,
                                                                                 int numa,
                                                                                 uint64_t flags) {
            EXPECT_EQ(actual_ior, ior);
            EXPECT_EQ(std::strcmp(mount, "/"), 0);
            EXPECT_EQ(entries, ior_entries);
            EXPECT_EQ(for_read, read);
            EXPECT_EQ(io_depth, ior_io_depth);
            EXPECT_EQ(timeout, ior_timeout_ms);
            EXPECT_EQ(numa, -1);
            EXPECT_EQ(flags, 0);
            return 0;
        }));
    EXPECT_TRUE(client_->CreateIor(ior, read, ior_entries, ior_io_depth, ior_timeout_ms));
    EXPECT_NE(ior, nullptr);

    EXPECT_CALL(*mock_api, Hf3fsIorDestroy(ior)).Times(1);
    client_->DestroyIor(ior);
    SUCCEED();
}

// ---------- DestroyIor ----------
TEST_F(Hf3fsUsrbioClientTest, DestroyIor_ReturnVoid_Nullptr) {
    client_->DestroyIor(nullptr);
    SUCCEED();
}

TEST_F(Hf3fsUsrbioClientTest, DestroyIor_ReturnVoid_ValidIor) {
    auto api = static_cast<MockHf3fsUsrbioApi *>(client_->usrbio_api_.get());
    EXPECT_CALL(*api, Hf3fsIorDestroy(testing::NotNull())).Times(1);

    auto ior = new struct hf3fs_ior();
    client_->DestroyIor(ior);
    SUCCEED();
}

// ---------- CopyIovs ----------
TEST_F(Hf3fsUsrbioClientTest, CopyIovs_ReturnCopied_MemoryTypeCPU_LoadTrue) {
    const size_t size = 1024;
    auto handle = BuildIovHandle(size, 'a');
    ASSERT_NE(handle.iov_base, nullptr);
    ASSERT_EQ(handle.iov_size, size);

    std::vector<uint8_t> data1(512, 0);
    std::vector<uint8_t> data2(512, 0);

    std::vector<Iov> iovs;
    iovs.push_back(Iov{MemoryType::CPU, data1.data(), 512, false});
    iovs.push_back(Iov{MemoryType::CPU, data2.data(), 512, false});

    client_->CopyIovs(iovs, handle, true);
    EXPECT_EQ(std::memcmp(data1.data(), handle.iov_base.get(), 512), 0);
    EXPECT_EQ(std::memcmp(data2.data(), handle.iov_base.get() + 512, 512), 0);
}

TEST_F(Hf3fsUsrbioClientTest, CopyIovs_ReturnCopied_MemoryTypeCPU_LoadFalse) {
    const size_t size = 1024;
    auto handle = BuildIovHandle(size, 0);
    ASSERT_NE(handle.iov_base, nullptr);
    ASSERT_EQ(handle.iov_size, size);

    std::vector<uint8_t> data1(512, 'a');
    std::vector<uint8_t> data2(512, 'b');

    std::vector<Iov> iovs;
    iovs.push_back(Iov{MemoryType::CPU, data1.data(), 512, false});
    iovs.push_back(Iov{MemoryType::CPU, data2.data(), 512, false});

    client_->CopyIovs(iovs, handle, false);
    EXPECT_EQ(std::memcmp(data1.data(), handle.iov_base.get(), 512), 0);
    EXPECT_EQ(std::memcmp(data2.data(), handle.iov_base.get() + 512, 512), 0);
}

#ifdef USING_CUDA
TEST_F(Hf3fsUsrbioClientTest, CopyIovs_ReturnCopied_MemoryTypeGPU_LoadTrue) {
    const size_t size = 1024;
    auto handle = BuildIovHandle(size, 'a');
    ASSERT_NE(handle.iov_base, nullptr);
    ASSERT_EQ(handle.iov_size, size);

    void *data1 = nullptr;
    ASSERT_EQ(cudaMalloc(&data1, 512), cudaSuccess);
    ASSERT_NE(data1, nullptr);
    ASSERT_EQ(cudaMemset(data1, 0, 512), cudaSuccess);

    void *data2 = nullptr;
    ASSERT_EQ(cudaMalloc(&data2, 512), cudaSuccess);
    ASSERT_NE(data2, nullptr);
    ASSERT_EQ(cudaMemset(data2, 0, 512), cudaSuccess);

    std::vector<Iov> iovs;
    iovs.push_back(Iov{MemoryType::GPU, data1, 512, false});
    iovs.push_back(Iov{MemoryType::GPU, data2, 512, false});

    client_->CopyIovs(iovs, handle, true);

    std::shared_ptr<uint8_t> data1_cpu(new uint8_t[512], [](uint8_t *ptr) { delete[] ptr; });
    std::shared_ptr<uint8_t> data2_cpu(new uint8_t[512], [](uint8_t *ptr) { delete[] ptr; });
    ASSERT_EQ(cudaMemcpy(data1_cpu.get(), data1, 512, cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(data2_cpu.get(), data2, 512, cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(std::memcmp(data1_cpu.get(), handle.iov_base.get(), 512), 0);
    EXPECT_EQ(std::memcmp(data2_cpu.get(), handle.iov_base.get() + 512, 512), 0);

    ASSERT_EQ(cudaFree(data1), cudaSuccess);
    ASSERT_EQ(cudaFree(data2), cudaSuccess);
}
#endif

#ifdef USING_CUDA
TEST_F(Hf3fsUsrbioClientTest, CopyIovs_ReturnCopied_MemoryTypeGPU_LoadFalse) {
    const size_t size = 1024;
    auto handle = BuildIovHandle(size, 0);
    ASSERT_NE(handle.iov_base, nullptr);
    ASSERT_EQ(handle.iov_size, size);

    void *data1 = nullptr;
    ASSERT_EQ(cudaMalloc(&data1, 512), cudaSuccess);
    ASSERT_NE(data1, nullptr);
    ASSERT_EQ(cudaMemset(data1, 'a', 512), cudaSuccess);

    void *data2 = nullptr;
    ASSERT_EQ(cudaMalloc(&data2, 512), cudaSuccess);
    ASSERT_NE(data2, nullptr);
    ASSERT_EQ(cudaMemset(data2, 'b', 512), cudaSuccess);

    std::vector<Iov> iovs;
    iovs.push_back(Iov{MemoryType::GPU, data1, 512, false});
    iovs.push_back(Iov{MemoryType::GPU, data2, 512, false});

    client_->CopyIovs(iovs, handle, false);

    std::shared_ptr<uint8_t> data1_cpu(new uint8_t[512], [](uint8_t *ptr) { delete[] ptr; });
    std::shared_ptr<uint8_t> data2_cpu(new uint8_t[512], [](uint8_t *ptr) { delete[] ptr; });
    ASSERT_EQ(cudaMemcpy(data1_cpu.get(), data1, 512, cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(data2_cpu.get(), data2, 512, cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(std::memcmp(data1_cpu.get(), handle.iov_base.get(), 512), 0);
    EXPECT_EQ(std::memcmp(data2_cpu.get(), handle.iov_base.get() + 512, 512), 0);

    ASSERT_EQ(cudaFree(data1), cudaSuccess);
    ASSERT_EQ(cudaFree(data2), cudaSuccess);
}
#endif

// ---------- CalcLeftSizeInBlock ----------
TEST_F(Hf3fsUsrbioClientTest, CalcLeftSizeInBlock_ReturnCorrect_Simple) {
    EXPECT_EQ(client_->CalcLeftSizeInBlock(16, 0), 16);
    EXPECT_EQ(client_->CalcLeftSizeInBlock(16, 1), 15);
    EXPECT_EQ(client_->CalcLeftSizeInBlock(16, 15), 1);
}

TEST_F(Hf3fsUsrbioClientTest, CalcLeftSizeInBlock_ReturnCorrect_LargeOffset) {
    EXPECT_EQ(client_->CalcLeftSizeInBlock(4096, 4100), 4092);
}

// ---------- Open ----------
TEST_F(Hf3fsUsrbioClientTest, Open_ReturnFalse_ReadMode_FileNotExist) {
    if (std::filesystem::exists(client_->filepath_)) {
        std::filesystem::remove(client_->filepath_);
    }
    EXPECT_FALSE(client_->Open(false));
}

TEST_F(Hf3fsUsrbioClientTest, Open_ReturnFalse_WriteNode_RegFdFail) {
    auto mock_usrbio_api = static_cast<MockHf3fsUsrbioApi *>(client_->usrbio_api_.get());
    EXPECT_CALL(*mock_usrbio_api, Hf3fsRegFd(testing::_, testing::_)).WillOnce(testing::Return(1)); // reg fd failed
    EXPECT_FALSE(client_->Open(true));
}

TEST_F(Hf3fsUsrbioClientTest, Open_ReturnTrue_WriteMode_CreateAndCleanup) {
    auto mock_usrbio_api = static_cast<MockHf3fsUsrbioApi *>(client_->usrbio_api_.get());
    EXPECT_CALL(*mock_usrbio_api, Hf3fsRegFd(testing::_, testing::_)).WillOnce(testing::Return(0));
    EXPECT_CALL(*mock_usrbio_api, Hf3fsDeregFd(testing::_)).Times(1);
    ASSERT_TRUE(client_->Open(true));
    ASSERT_TRUE(client_->Fsync());
    ASSERT_TRUE(client_->Close());
    ASSERT_TRUE(client_->FileLength().has_value());
    client_->Del();
    SUCCEED();
}

// ---------- Close ----------
TEST_F(Hf3fsUsrbioClientTest, Close_ReturnTrue_FdNotOpen) { EXPECT_TRUE(client_->Close()); }

TEST_F(Hf3fsUsrbioClientTest, Close_ReturnTrue_DeregFdIsFalse) {
    auto mock_usrbio_api = static_cast<MockHf3fsUsrbioApi *>(client_->usrbio_api_.get());
    EXPECT_CALL(*mock_usrbio_api, Hf3fsDeregFd(testing::_)).Times(0);
    EXPECT_TRUE(client_->Close(false));
}

TEST_F(Hf3fsUsrbioClientTest, Close_ReturnTrue_DeregFdIsTrue) {
    auto mock_usrbio_api = static_cast<MockHf3fsUsrbioApi *>(client_->usrbio_api_.get());

    EXPECT_CALL(*mock_usrbio_api, Hf3fsRegFd(testing::_, testing::_)).WillOnce(testing::Return(0));
    EXPECT_TRUE(client_->Open(true));

    EXPECT_CALL(*mock_usrbio_api, Hf3fsDeregFd(testing::_)).Times(1);
    EXPECT_TRUE(client_->Close(true));
}

// ---------- Del ----------
TEST_F(Hf3fsUsrbioClientTest, Del_ReturnVoid) {
    client_->Del();
    SUCCEED();
}

// ---------- Fsync ----------
TEST_F(Hf3fsUsrbioClientTest, Fsync_ReturnFalse_FdNotOpen) { EXPECT_FALSE(client_->Fsync()); }

// ---------- FileLength ----------
TEST_F(Hf3fsUsrbioClientTest, FileLength_ReturnNullopt_NotExist) {
    auto len = client_->FileLength();
    EXPECT_FALSE(len.has_value());
}