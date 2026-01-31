// Basic unit tests for Hf3fsSdk internal helpers

#include <fstream>
#include <gtest/gtest.h>

#include "kv_cache_manager/client/src/internal/sdk/hf3fs_cuda_util.h"
#include "kv_cache_manager/client/src/internal/sdk/hf3fs_mempool.h"
#include "kv_cache_manager/client/src/internal/sdk/hf3fs_sdk.h"
#include "kv_cache_manager/client/src/internal/sdk/test/mock/mock_hf3fs_usrbio_api.h"
#include "kv_cache_manager/common/unittest.h"

using namespace ::testing;
using namespace kv_cache_manager;

class Hf3fsSdkTest : public TESTBASE {
protected:
    void SetUp() override {
        mount_point_ = GetTestTempRootPath();
        sdk_ = std::make_shared<Hf3fsSdk>();

        auto sdk_config = std::make_shared<Hf3fsSdkConfig>();
        sdk_config->set_type(DataStorageType::DATA_STORAGE_TYPE_HF3FS);
        sdk_config->set_sdk_log_file_path(mount_point_ + "/sdk.log");
        sdk_config->set_sdk_log_level("INFO");
        sdk_config->set_byte_size_per_block(1024 * 1024);
        sdk_config->set_mountpoint(mount_point_);
        sdk_config->set_root_dir("/");

        auto storage_config = std::make_shared<StorageConfig>();
        storage_config->set_type(DataStorageType::DATA_STORAGE_TYPE_HF3FS);

        sdk_->config_ = sdk_config;
        sdk_->usrbio_api_ = std::make_shared<MockHf3fsUsrbioApi>();
        sdk_->read_iov_handle_ = BuildIovHandle(1 << 20);
        sdk_->write_iov_handle_ = BuildIovHandle(1 << 20);
    }

    void TearDown() override { sdk_.reset(); }

private:
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

protected:
    std::string mount_point_;
    std::shared_ptr<Hf3fsSdk> sdk_;
};

// ------------- Init -------------
TEST_F(Hf3fsSdkTest, Init_ReturnInvalidConfig_WrongBackendType) {
    auto wrong = std::make_shared<SdkBackendConfig>();
    wrong->set_type(DataStorageType::DATA_STORAGE_TYPE_NFS);
    auto storage = std::make_shared<StorageConfig>();
    storage->set_type(DataStorageType::DATA_STORAGE_TYPE_HF3FS);
    auto rc = sdk_->Init(wrong, storage);
    EXPECT_EQ(rc, ER_INVALID_SDKBACKEND_CONFIG);
}

TEST_F(Hf3fsSdkTest, Init_ReturnInvalidConfig_CheckConfigFail) {
    auto cfg = std::make_shared<Hf3fsSdkConfig>();
    cfg->set_type(DataStorageType::DATA_STORAGE_TYPE_HF3FS);
    // invalid mountpoint/root -> CheckConfig false
    cfg->set_mountpoint(mount_point_ + "/no_such");
    cfg->set_root_dir("/no_such_root");
    auto storage = std::make_shared<StorageConfig>();
    storage->set_type(DataStorageType::DATA_STORAGE_TYPE_HF3FS);
    auto rc = sdk_->Init(cfg, storage);
    EXPECT_EQ(rc, ER_INVALID_SDKBACKEND_CONFIG);
}

TEST_F(Hf3fsSdkTest, Init_ReturnOk_SuccessOrSkipInitIovHandleFail) {
    auto cfg = std::make_shared<Hf3fsSdkConfig>();
    cfg->set_type(DataStorageType::DATA_STORAGE_TYPE_HF3FS);
    cfg->set_mountpoint(mount_point_);
    std::string rd = "init_ok_root_" + std::to_string(::getpid());
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(mount_point_) / rd, ec);
    cfg->set_root_dir(rd);
    cfg->set_read_iov_block_size(0);
    cfg->set_read_iov_size(1 << 20);
    cfg->set_write_iov_block_size(1 << 20);
    cfg->set_write_iov_size(1 << 20);

    auto storage = std::make_shared<StorageConfig>();
    storage->set_type(DataStorageType::DATA_STORAGE_TYPE_HF3FS);
    auto rc = sdk_->Init(cfg, storage);
    if (rc == ER_SDKINIT_ERROR) {
        GTEST_SKIP() << "init iov handle failed; skip";
    }
    EXPECT_EQ(rc, ER_OK);
}

// ------------- Get (batch) -------------
TEST_F(Hf3fsSdkTest, GetBatch_ReturnInvalidParams_SizeMismatch) {
    std::vector<DataStorageUri> uris(2);
    BlockBuffers bufs(1);
    auto rc = sdk_->Get(uris, bufs);
    EXPECT_EQ(rc, ER_INVALID_PARAMS);
}

TEST_F(Hf3fsSdkTest, GetBatch_ReturnReadError_FirstFailShortCircuit) {
    // prepare two files; first will fail via RegFd, second should not be attempted
    auto path1 = (std::filesystem::path(mount_point_) / "get_batch_short/a/x.bin");
    auto path2 = (std::filesystem::path(mount_point_) / "get_batch_short/b/y.bin");
    std::error_code ec;
    std::filesystem::create_directories(path1.parent_path(), ec);
    std::filesystem::create_directories(path2.parent_path(), ec);
    {
        std::ofstream f1(path1, std::ios::out | std::ios::binary);
        f1.seekp(8192 - 1);
        f1.write("\0", 1);
    }
    {
        std::ofstream f2(path2, std::ios::out | std::ios::binary);
        f2.seekp(8192 - 1);
        f2.write("\0", 1);
    }

    DataStorageUri u1, u2;
    u1.SetPath(path1.string());
    u1.SetParam("blkid", "0");
    u1.SetParam("size", "4096");
    u2.SetPath(path2.string());
    u2.SetParam("blkid", "0");
    u2.SetParam("size", "4096");

    std::vector<DataStorageUri> uris{u1, u2};
    char out1[16] = {0};
    char out2[16] = {0};
    BlockBuffer b1, b2;
    b1.iovs.push_back(Iov{MemoryType::CPU, out1, sizeof(out1), false});
    b2.iovs.push_back(Iov{MemoryType::CPU, out2, sizeof(out2), false});
    BlockBuffers bufs{b1, b2};

    auto mock = std::dynamic_pointer_cast<MockHf3fsUsrbioApi>(sdk_->usrbio_api_);
    ASSERT_TRUE(mock != nullptr);
    // first read fails at RegFd
    {
        ::testing::InSequence seq;
        EXPECT_CALL(*mock, Hf3fsRegFd(::testing::_, ::testing::_)).WillOnce(::testing::Return(1));
    }
    // The SDK should short-circuit and return ER_SDKREAD_ERROR
    auto rc = sdk_->Get(uris, bufs);
    EXPECT_EQ(rc, ER_SDKREAD_ERROR);
}

TEST_F(Hf3fsSdkTest, GetBatch_ReturnOk_BothSuccess) {
    // prepare two files
    std::error_code ec;
    auto path1 = (std::filesystem::path(mount_point_) / "get_batch_ok/a/x.bin");
    std::filesystem::create_directories(path1.parent_path(), ec);
    EXPECT_EQ(ec.value(), 0);
    auto path2 = (std::filesystem::path(mount_point_) / "get_batch_ok/b/y.bin");
    std::filesystem::create_directories(path2.parent_path(), ec);
    EXPECT_EQ(ec.value(), 0);

    size_t file_size = 8192;
    {
        std::ofstream f1(path1, std::ios::out | std::ios::binary);
        for (size_t i = 0; i < file_size; ++i) {
            f1.write("a", 1);
        }
    }
    {
        std::ofstream f2(path2, std::ios::out | std::ios::binary);
        for (size_t i = 0; i < file_size; ++i) {
            f2.write("b", 1);
        }
    }

    DataStorageUri u1, u2;
    u1.SetPath(path1.string());
    u1.SetParam("blkid", "0");
    u1.SetParam("size", "4096");
    u2.SetPath(path2.string());
    u2.SetParam("blkid", "1");
    u2.SetParam("size", "4096");

    std::vector<DataStorageUri> uris{u1, u2};
    char out1[16] = {0};
    char out2[32] = {0};
    BlockBuffer b1, b2;
    b1.iovs.push_back(Iov{MemoryType::CPU, out1, sizeof(out1), false});
    b2.iovs.push_back(Iov{MemoryType::CPU, out2, sizeof(out2), false});
    BlockBuffers bufs{b1, b2};

    auto mock = std::dynamic_pointer_cast<MockHf3fsUsrbioApi>(sdk_->usrbio_api_);
    ASSERT_TRUE(mock != nullptr);
    EXPECT_CALL(*mock, Hf3fsRegFd(::testing::_, ::testing::_)).WillRepeatedly(::testing::Return(0));
    EXPECT_CALL(*mock, Hf3fsDeregFd(::testing::_)).Times(::testing::AtLeast(0));
    EXPECT_CALL(*mock,
                Hf3fsIorCreate(::testing::NotNull(),
                               ::testing::_,
                               ::testing::_,
                               ::testing::_,
                               ::testing::_,
                               ::testing::_,
                               ::testing::_,
                               ::testing::_))
        .WillRepeatedly(::testing::Return(0));
    EXPECT_CALL(*mock, Hf3fsIorDestroy(::testing::NotNull())).Times(::testing::AtLeast(0));
    EXPECT_CALL(*mock,
                Hf3fsPrepIo(::testing::_,
                            ::testing::_,
                            ::testing::_,
                            ::testing::_,
                            ::testing::_,
                            ::testing::_,
                            ::testing::_,
                            ::testing::_))
        .WillRepeatedly(::testing::Invoke([](const struct hf3fs_ior *ior,
                                             const struct hf3fs_iov *iov,
                                             bool read,
                                             void *ptr,
                                             int fd,
                                             size_t off,
                                             uint64_t len,
                                             const void *userdata) {
            lseek(fd, off, SEEK_SET);
            return ::read(fd, ptr, len) == len ? 0 : -1;
        }));
    EXPECT_CALL(*mock, Hf3fsSubmitIos(::testing::_)).WillRepeatedly(::testing::Return(0));
    EXPECT_CALL(*mock, Hf3fsWaitForIos(::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Invoke([](const ::hf3fs_ior *, ::hf3fs_cqe *cqes, int cqec, int, const timespec *) {
            for (int i = 0; i < cqec; ++i) {
                cqes[i].result = 1;
            }
            return cqec;
        }));

    auto rc = sdk_->Get(uris, bufs);
    EXPECT_EQ(rc, ER_OK);
    EXPECT_EQ(std::string(out1, sizeof(out1)), std::string(16, 'a'));
    EXPECT_EQ(std::string(out2, sizeof(out2)), std::string(32, 'b'));
}

// ------------- Get (single) -------------
TEST_F(Hf3fsSdkTest, Get_ReturnOk_EmptyIovs) {
    DataStorageUri uri;
    uri.SetPath((std::filesystem::path(mount_point_) / "get/empty.dat").string());
    BlockBuffer buf; // empty iovs
    auto rc = sdk_->Get(uri, buf);
    EXPECT_EQ(rc, ER_OK);
}

TEST_F(Hf3fsSdkTest, Get_ReturnInvalid_ParamsEmptyPath) {
    DataStorageUri uri;
    uri.SetPath("");
    BlockBuffer buf;
    buf.iovs.push_back(Iov{MemoryType::CPU, (void *)0x1, 10, false});
    auto rc = sdk_->Get(uri, buf);
    EXPECT_EQ(rc, ER_INVALID_PARAMS);
}

TEST_F(Hf3fsSdkTest, Get_ReturnInvalid_ParamsNoSize) {
    DataStorageUri uri;
    uri.SetPath((std::filesystem::path(mount_point_) / "get/no_size.dat").string());
    // no size/blkid param set
    BlockBuffer buf;
    buf.iovs.push_back(Iov{MemoryType::CPU, (void *)0x1, 10, false});
    auto rc = sdk_->Get(uri, buf);
    EXPECT_EQ(rc, ER_INVALID_PARAMS);
}

TEST_F(Hf3fsSdkTest, Get_ReturnReadError_RegFdFail) {
    // prepare file to satisfy FileLength check
    auto path = (std::filesystem::path(mount_point_) / "get/regfd_fail.dat");
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    {
        std::ofstream fs(path, std::ios::out | std::ios::binary);
        fs.seekp(4096 - 1);
        fs.write("\0", 1);
    }

    DataStorageUri uri;
    uri.SetPath(path.string());
    uri.SetParam("blkid", "0");
    uri.SetParam("size", "4096");

    char bufmem[16] = {0};
    BlockBuffer buf;
    buf.iovs.push_back(Iov{MemoryType::CPU, bufmem, sizeof(bufmem), false});

    auto mock = std::dynamic_pointer_cast<MockHf3fsUsrbioApi>(sdk_->usrbio_api_);
    ASSERT_TRUE(mock != nullptr);
    // make RegFd fail (>0)
    EXPECT_CALL(*mock, Hf3fsRegFd(::testing::_, ::testing::_)).WillRepeatedly(::testing::Return(1));

    auto rc = sdk_->Get(uri, buf);
    EXPECT_EQ(rc, ER_SDKREAD_ERROR);
}

TEST_F(Hf3fsSdkTest, Get_ReturnOk_Success) {
    // create a file large enough
    auto path = (std::filesystem::path(mount_point_) / "get/success.dat");
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    size_t file_size = 8192;
    {
        std::ofstream fs(path, std::ios::out | std::ios::binary);
        for (size_t i = 0; i < file_size; ++i) {
            fs.write("a", 1);
        }
    }

    DataStorageUri uri;
    uri.SetPath(path.string());
    uri.SetParam("blkid", "0");
    uri.SetParam("size", "4096");

    char outbuf1[16] = {0};
    char outbuf2[32] = {0};
    BlockBuffer buf;
    buf.iovs.push_back(Iov{MemoryType::CPU, outbuf1, sizeof(outbuf1), false});
    buf.iovs.push_back(Iov{MemoryType::CPU, outbuf2, sizeof(outbuf2), false});

    auto mock = std::dynamic_pointer_cast<MockHf3fsUsrbioApi>(sdk_->usrbio_api_);
    ASSERT_TRUE(mock != nullptr);
    EXPECT_CALL(*mock, Hf3fsRegFd(::testing::_, ::testing::_)).WillRepeatedly(::testing::Return(0));
    EXPECT_CALL(*mock, Hf3fsDeregFd(::testing::_)).Times(::testing::AtLeast(0));
    EXPECT_CALL(*mock,
                Hf3fsIorCreate(::testing::NotNull(),
                               ::testing::_,
                               ::testing::_,
                               ::testing::_,
                               ::testing::_,
                               ::testing::_,
                               ::testing::_,
                               ::testing::_))
        .WillRepeatedly(::testing::Return(0));
    EXPECT_CALL(*mock, Hf3fsIorDestroy(::testing::NotNull())).Times(::testing::AtLeast(0));
    EXPECT_CALL(*mock,
                Hf3fsPrepIo(::testing::_,
                            ::testing::_,
                            ::testing::_,
                            ::testing::_,
                            ::testing::_,
                            ::testing::_,
                            ::testing::_,
                            ::testing::_))
        .WillRepeatedly(::testing::Invoke([](const struct hf3fs_ior *ior,
                                             const struct hf3fs_iov *iov,
                                             bool read,
                                             void *ptr,
                                             int fd,
                                             size_t off,
                                             uint64_t len,
                                             const void *userdata) {
            lseek(fd, off, SEEK_SET);
            return ::read(fd, ptr, len) == len ? 0 : -1;
        }));
    EXPECT_CALL(*mock, Hf3fsSubmitIos(::testing::_)).WillRepeatedly(::testing::Return(0));
    EXPECT_CALL(*mock, Hf3fsWaitForIos(::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Invoke([](const ::hf3fs_ior *, ::hf3fs_cqe *cqes, int cqec, int, const timespec *) {
            for (int i = 0; i < cqec; ++i) {
                cqes[i].result = 1;
            }
            return cqec;
        }));

    auto rc = sdk_->Get(uri, buf);
    EXPECT_EQ(rc, ER_OK);
    EXPECT_EQ(std::string(outbuf1, sizeof(outbuf1)), std::string(16, 'a'));
    EXPECT_EQ(std::string(outbuf2, sizeof(outbuf2)), std::string(32, 'a'));
}

// ------------- Put (batch) -------------
TEST_F(Hf3fsSdkTest, PutBatch_ReturnInvalidParams_SizeMismatch) {
    std::vector<DataStorageUri> uris(2);
    BlockBuffers bufs(1);
    auto out = std::make_shared<std::vector<DataStorageUri>>();
    auto rc = sdk_->Put(uris, bufs, out);
    EXPECT_EQ(rc, ER_INVALID_PARAMS);
}

TEST_F(Hf3fsSdkTest, PutBatch_ReturnAllocError_CreateDirFail) {
    std::vector<DataStorageUri> uris;
    DataStorageUri u;
    u.SetPath("/proc/kvcm_put_alloc_" + std::to_string(::getpid()) + "/f.bin");
    u.SetParam("blkid", "0");
    u.SetParam("size", "4096");
    uris.push_back(u);

    BlockBuffer b;
    char data[8] = {0};
    b.iovs.push_back(Iov{MemoryType::CPU, data, sizeof(data), false});
    BlockBuffers bufs{b};
    auto out = std::make_shared<std::vector<DataStorageUri>>();
    auto rc = sdk_->Put(uris, bufs, out);
    EXPECT_EQ(rc, ER_SDKALLOC_ERROR);
}

TEST_F(Hf3fsSdkTest, PutBatch_ReturnOk_AllSuccess) {
    DataStorageUri u1;
    u1.SetPath(
        (std::filesystem::path(mount_point_) / ("put_batch_ok_" + std::to_string(::getpid()) + "/a/x.bin")).string());
    u1.SetParam("blkid", "1");
    u1.SetParam("size", "4096");

    DataStorageUri u2;
    u2.SetPath(
        (std::filesystem::path(mount_point_) / ("put_batch_ok_" + std::to_string(::getpid()) + "/b/y.bin")).string());
    u2.SetParam("blkid", "2");
    u2.SetParam("size", "4096");

    std::vector<DataStorageUri> uris;
    uris.push_back(u1);
    uris.push_back(u2);

    size_t size1 = 16;
    size_t size2 = 32;
    BlockBuffer b1, b2;
    auto d1 = std::shared_ptr<uint8_t>((uint8_t *)malloc(size1), [](void *ptr) { free(ptr); });
    std::memset(d1.get(), 'a', size1);
    auto d2 = std::shared_ptr<uint8_t>((uint8_t *)malloc(size2), [](void *ptr) { free(ptr); });
    std::memset(d2.get(), 'b', size2);
    b1.iovs.push_back(Iov{MemoryType::CPU, d1.get(), size1, false});
    b2.iovs.push_back(Iov{MemoryType::CPU, d2.get(), size2, false});
    BlockBuffers bufs{b1, b2};

    auto mock = std::dynamic_pointer_cast<MockHf3fsUsrbioApi>(sdk_->usrbio_api_);
    ASSERT_TRUE(mock != nullptr);
    EXPECT_CALL(*mock, Hf3fsRegFd(::testing::_, ::testing::_)).WillRepeatedly(::testing::Return(0));
    EXPECT_CALL(*mock, Hf3fsDeregFd(::testing::_)).Times(::testing::AtLeast(0));
    EXPECT_CALL(*mock,
                Hf3fsIorCreate(::testing::NotNull(),
                               ::testing::_,
                               ::testing::_,
                               ::testing::_,
                               ::testing::_,
                               ::testing::_,
                               ::testing::_,
                               ::testing::_))
        .WillRepeatedly(::testing::Return(0));
    EXPECT_CALL(*mock, Hf3fsIorDestroy(::testing::NotNull())).Times(::testing::AtLeast(0));
    EXPECT_CALL(*mock,
                Hf3fsPrepIo(::testing::_,
                            ::testing::_,
                            ::testing::_,
                            ::testing::_,
                            ::testing::_,
                            ::testing::_,
                            ::testing::_,
                            ::testing::_))
        .WillRepeatedly(::testing::Invoke([](const struct hf3fs_ior *ior,
                                             const struct hf3fs_iov *iov,
                                             bool read,
                                             void *ptr,
                                             int fd,
                                             size_t off,
                                             uint64_t len,
                                             const void *userdata) {
            lseek(fd, off, SEEK_SET);
            return ::write(fd, ptr, len) == len ? 0 : -1;
        }));
    EXPECT_CALL(*mock, Hf3fsSubmitIos(::testing::_)).WillRepeatedly(::testing::Return(0));
    EXPECT_CALL(*mock, Hf3fsWaitForIos(::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Invoke([](const ::hf3fs_ior *, ::hf3fs_cqe *cqes, int cqec, int, const timespec *) {
            for (int i = 0; i < cqec; ++i) {
                cqes[i].result = 1;
            }
            return cqec;
        }));

    auto out = std::make_shared<std::vector<DataStorageUri>>();
    auto rc = sdk_->Put(uris, bufs, out);
    EXPECT_EQ(rc, ER_OK);
    ASSERT_EQ(out->size(), uris.size());

    EXPECT_EQ(out->at(0).GetPath(), uris[0].GetPath());
    EXPECT_TRUE(std::filesystem::exists(uris[0].GetPath()));
    EXPECT_EQ(std::filesystem::file_size(uris[0].GetPath()), 1 * 4096 + size1);

    EXPECT_EQ(out->at(1).GetPath(), uris[1].GetPath());
    EXPECT_TRUE(std::filesystem::exists(uris[1].GetPath()));
    EXPECT_EQ(std::filesystem::file_size(uris[1].GetPath()), 2 * 4096 + size2);
}

// ------------- Put (single) -------------
TEST_F(Hf3fsSdkTest, Put_ReturnOk_EmptyIovs) {
    DataStorageUri uri;
    uri.SetPath((std::filesystem::path(mount_point_) / "put/empty.dat").string());
    BlockBuffer buf; // empty iovs
    auto rc = sdk_->Put(uri, buf);
    EXPECT_EQ(rc, ER_OK);
}

TEST_F(Hf3fsSdkTest, Put_ReturnInvalid_ParamsEmptyPath) {
    DataStorageUri uri;
    uri.SetPath("");
    BlockBuffer buf;
    buf.iovs.push_back(Iov{MemoryType::CPU, (void *)0x1, 10, false});
    auto rc = sdk_->Put(uri, buf);
    EXPECT_EQ(rc, ER_INVALID_PARAMS);
}

TEST_F(Hf3fsSdkTest, Put_ReturnInvalid_ParamsNoSize) {
    DataStorageUri uri;
    uri.SetPath((std::filesystem::path(mount_point_) / "put/no_size.dat").string());
    // no size param set
    BlockBuffer buf;
    buf.iovs.push_back(Iov{MemoryType::CPU, (void *)0x1, 10, false});
    auto rc = sdk_->Put(uri, buf);
    EXPECT_EQ(rc, ER_INVALID_PARAMS);
}

TEST_F(Hf3fsSdkTest, Put_ReturnWriteError_OpenFail) {
    DataStorageUri uri;
    // parent directories deliberately do not exist so Open(true) fails
    uri.SetPath((std::filesystem::path(mount_point_) / "put_should_fail/open_fail_dir/sub/file.bin").string());
    uri.SetParam("blkid", "1");
    uri.SetParam("size", "4096");

    char payload[16] = {0};
    BlockBuffer buf;
    buf.iovs.push_back(Iov{MemoryType::CPU, payload, sizeof(payload), false});

    auto rc = sdk_->Put(uri, buf);
    EXPECT_EQ(rc, ER_SDKWRITE_ERROR);
}

TEST_F(Hf3fsSdkTest, Put_ReturnOk_WriteSuccess) {
    DataStorageUri uri;
    uri.SetPath((std::filesystem::path(mount_point_) / "put/write_success.dat").string());
    uri.SetParam("blkid", "1");
    uri.SetParam("size", "4096");

    std::vector<DataStorageUri> out;
    EXPECT_EQ(sdk_->Alloc({uri}, out), ER_OK);
    EXPECT_EQ(out.size(), 1);
    EXPECT_EQ(out[0].GetPath(), uri.GetPath());

    size_t payload_size = 1024;
    auto payload = std::shared_ptr<uint8_t>((uint8_t *)malloc(payload_size), [](void *ptr) { free(ptr); });
    std::memset(payload.get(), 'a', payload_size);

    BlockBuffer buf;
    buf.iovs.push_back(Iov{MemoryType::CPU, payload.get(), payload_size, false});
    buf.iovs.push_back(Iov{MemoryType::CPU, payload.get(), payload_size, false});

    auto mock = std::dynamic_pointer_cast<MockHf3fsUsrbioApi>(sdk_->usrbio_api_);
    ASSERT_TRUE(mock != nullptr);

    EXPECT_CALL(*mock, Hf3fsRegFd(::testing::_, ::testing::_)).WillRepeatedly(::testing::Return(0));
    EXPECT_CALL(*mock, Hf3fsDeregFd(::testing::_)).Times(::testing::AtLeast(0));
    EXPECT_CALL(*mock,
                Hf3fsIorCreate(::testing::NotNull(),
                               ::testing::_,
                               ::testing::_,
                               ::testing::_,
                               ::testing::_,
                               ::testing::_,
                               ::testing::_,
                               ::testing::_))
        .WillRepeatedly(::testing::Return(0));
    EXPECT_CALL(*mock, Hf3fsIorDestroy(::testing::NotNull())).Times(::testing::AtLeast(0));
    EXPECT_CALL(*mock,
                Hf3fsPrepIo(::testing::_,
                            ::testing::_,
                            ::testing::_,
                            ::testing::_,
                            ::testing::_,
                            ::testing::_,
                            ::testing::_,
                            ::testing::_))
        .WillRepeatedly(::testing::Invoke([](const struct hf3fs_ior *ior,
                                             const struct hf3fs_iov *iov,
                                             bool read,
                                             void *ptr,
                                             int fd,
                                             size_t off,
                                             uint64_t len,
                                             const void *userdata) {
            lseek(fd, off, SEEK_SET);
            return ::write(fd, ptr, len) == len ? 0 : -1;
        }));
    EXPECT_CALL(*mock, Hf3fsSubmitIos(::testing::_)).WillRepeatedly(::testing::Return(0));
    EXPECT_CALL(*mock, Hf3fsWaitForIos(::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Invoke([](const ::hf3fs_ior *, ::hf3fs_cqe *cqes, int cqec, int, const timespec *) {
            for (int i = 0; i < cqec; ++i) {
                cqes[i].result = 1;
            }
            return cqec;
        }));

    auto rc = sdk_->Put(uri, buf);
    EXPECT_EQ(rc, ER_OK);
    auto expected_file = std::filesystem::path(uri.GetPath());
    EXPECT_TRUE(std::filesystem::exists(expected_file));
    auto file_size = std::filesystem::file_size(expected_file);
    EXPECT_EQ(file_size, 1 * 4096 + payload_size * buf.iovs.size());
}

// ------------- Alloc -------------
TEST_F(Hf3fsSdkTest, Alloc_ReturnError_EmptyPath) {
    std::vector<DataStorageUri> uris;
    uris.emplace_back(); // empty path
    std::vector<DataStorageUri> out;
    auto rc = sdk_->Alloc(uris, out);
    EXPECT_EQ(rc, ER_SDKALLOC_ERROR);
}

TEST_F(Hf3fsSdkTest, Alloc_ReturnError_CreateDirFail) {
    std::vector<DataStorageUri> uris;
    DataStorageUri u;
    u.SetPath("/proc/kvcm_alloc_" + std::to_string(::getpid()) + "/file.dat"); // read-only fs
    uris.emplace_back(u);
    std::vector<DataStorageUri> out;
    auto rc = sdk_->Alloc(uris, out);
    EXPECT_EQ(rc, ER_SDKALLOC_ERROR);
}

TEST_F(Hf3fsSdkTest, Alloc_ReturnOk_CreateParents) {
    std::vector<DataStorageUri> uris;
    DataStorageUri u;
    std::string rel = std::string("kvcm_alloc_ok_") + std::to_string(::getpid()) + "/a/b/file.dat";
    u.SetPath((std::filesystem::path(mount_point_) / rel).string());
    uris.emplace_back(u);
    std::vector<DataStorageUri> out;
    auto rc = sdk_->Alloc(uris, out);
    EXPECT_EQ(rc, ER_OK);
    EXPECT_EQ(out.size(), uris.size());
    auto expected_dir = std::filesystem::path(uris[0].GetPath()).parent_path();
    EXPECT_TRUE(std::filesystem::exists(expected_dir));
}

TEST_F(Hf3fsSdkTest, Alloc_ReturnOk_MultiUris_CopyUris) {
    std::vector<DataStorageUri> uris;
    DataStorageUri u1, u2;
    u1.SetPath(
        (std::filesystem::path(mount_point_) / ("kvcm_alloc_multi_" + std::to_string(::getpid()) + "/x/y.f")).string());
    u2.SetPath(
        (std::filesystem::path(mount_point_) / ("kvcm_alloc_multi_" + std::to_string(::getpid()) + "/p/q.f")).string());
    uris.push_back(u1);
    uris.push_back(u2);
    std::vector<DataStorageUri> out;
    auto rc = sdk_->Alloc(uris, out);
    EXPECT_EQ(rc, ER_OK);
    ASSERT_EQ(out.size(), uris.size());
    EXPECT_EQ(out[0].GetPath(), uris[0].GetPath());
    EXPECT_EQ(out[1].GetPath(), uris[1].GetPath());
}

// ------------- CheckConfig -------------
TEST_F(Hf3fsSdkTest, CheckConfig_ReturnFalse_EmptyMountpoint) {
    Hf3fsSdkConfig cfg;
    cfg.set_mountpoint("");
    cfg.set_root_dir("/");
    EXPECT_FALSE(sdk_->CheckConfig(cfg));
}

TEST_F(Hf3fsSdkTest, CheckConfig_ReturnFalse_MountpointNotExists) {
    Hf3fsSdkConfig cfg;
    cfg.set_mountpoint(mount_point_ + "/no_such_mount");
    cfg.set_root_dir("/");
    EXPECT_FALSE(sdk_->CheckConfig(cfg));
}

TEST_F(Hf3fsSdkTest, CheckConfig_ReturnFalse_EmptyRootDir) {
    Hf3fsSdkConfig cfg;
    cfg.set_mountpoint(mount_point_);
    cfg.set_root_dir("");
    EXPECT_FALSE(sdk_->CheckConfig(cfg));
}

TEST_F(Hf3fsSdkTest, CheckConfig_ReturnFalse_RootDirNotExists) {
    Hf3fsSdkConfig cfg;
    cfg.set_mountpoint(mount_point_);
    std::string rd = "not_exist_root_" + std::to_string(::getpid());
    // ensure not exists
    std::filesystem::remove_all(std::filesystem::path(mount_point_) / rd);
    cfg.set_root_dir(rd);
    EXPECT_FALSE(sdk_->CheckConfig(cfg));
}

TEST_F(Hf3fsSdkTest, CheckConfig_ReturnTrue_Valid) {
    Hf3fsSdkConfig cfg;
    cfg.set_mountpoint(mount_point_);
    std::string rd = "exist_root_" + std::to_string(::getpid());
    auto full = std::filesystem::path(mount_point_) / rd;
    std::error_code ec;
    std::filesystem::create_directories(full, ec);
    ASSERT_TRUE(std::filesystem::exists(full));
    cfg.set_root_dir(rd);
    EXPECT_TRUE(sdk_->CheckConfig(cfg));
}

// ------------- DeleteRemainingIovShm -------------
TEST_F(Hf3fsSdkTest, DeleteRemainingIovShm_ReturnKeep_RecentPrefixedFile) {
    namespace fs = std::filesystem;
    fs::path dir("/dev/shm/");
    ASSERT_TRUE(fs::exists(dir));
    auto fname = std::string("hf3fs-iov-kvcm-test-") + std::to_string(::getpid()) + "-recent";
    fs::path p = dir / fname;
    {
        std::ofstream o(p);
        o << "x";
    }
    auto now = fs::file_time_type::clock::now();
    fs::last_write_time(p, now - std::chrono::seconds(60)); // 1 min ago (below 5 min)

    sdk_->DeleteRemainingIovShm();

    EXPECT_TRUE(fs::exists(p));
    (void)fs::remove(p);
}

TEST_F(Hf3fsSdkTest, DeleteRemainingIovShm_ReturnRemove_OldPrefixedFile) {
    namespace fs = std::filesystem;
    fs::path dir("/dev/shm/");
    ASSERT_TRUE(fs::exists(dir));
    auto fname = std::string("hf3fs-iov-kvcm-test-") + std::to_string(::getpid()) + "-old";
    fs::path p = dir / fname;
    {
        std::ofstream o(p);
        o << "x";
    }
    auto now = fs::file_time_type::clock::now();
    fs::last_write_time(p, now - std::chrono::seconds(600)); // 10 min ago (> 5 min)

    sdk_->DeleteRemainingIovShm();

    EXPECT_FALSE(fs::exists(p));
    (void)fs::remove(p);
}

TEST_F(Hf3fsSdkTest, DeleteRemainingIovShm_ReturnKeep_NonPrefixedFile) {
    namespace fs = std::filesystem;
    fs::path dir("/dev/shm/");
    ASSERT_TRUE(fs::exists(dir));
    auto fname = std::string("kvcm-test-") + std::to_string(::getpid()) + "-plain";
    fs::path p = dir / fname;
    {
        std::ofstream o(p);
        o << "x";
    }
    auto now = fs::file_time_type::clock::now();
    fs::last_write_time(p, now - std::chrono::seconds(600)); // 10 min ago but non-prefixed

    sdk_->DeleteRemainingIovShm();

    EXPECT_TRUE(fs::exists(p));
    (void)fs::remove(p);
}

// ------------- InitIovHandle -------------
TEST_F(Hf3fsSdkTest, InitIovHandle_ReturnFalse_CreateIovFail) {
    auto mock = std::dynamic_pointer_cast<MockHf3fsUsrbioApi>(sdk_->usrbio_api_);
    ASSERT_TRUE(mock != nullptr);
    size_t captured_size = 0;
    EXPECT_CALL(*mock, Hf3fsIovCreate(::testing::NotNull(), ::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillOnce(::testing::Invoke([&](::hf3fs_iov *iov, const char *, size_t size, size_t, int) {
            captured_size = size;
            return -1; // fail to create
        }));

    Hf3fsIovHandle handle;
    auto cuda_util = std::make_shared<Hf3fsCudaUtil>();
    bool ok = sdk_->InitIovHandle(handle, 4096, 8192, cuda_util);
    EXPECT_FALSE(ok);
    EXPECT_EQ(captured_size, 8192u);
}

TEST_F(Hf3fsSdkTest, InitIovHandle_ReturnFalse_SizeRoundedUp_CreateIovFail) {
    auto mock = std::dynamic_pointer_cast<MockHf3fsUsrbioApi>(sdk_->usrbio_api_);
    ASSERT_TRUE(mock != nullptr);
    size_t captured_size = 0;
    // iov_size 5000 with block 4096 should round up to 8192
    EXPECT_CALL(*mock, Hf3fsIovCreate(::testing::NotNull(), ::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillOnce(::testing::Invoke([&](::hf3fs_iov *iov, const char *, size_t size, size_t, int) {
            captured_size = size;
            return -1; // fail to create so InitIovHandle returns false
        }));

    Hf3fsIovHandle handle;
    auto cuda_util = std::make_shared<Hf3fsCudaUtil>();
    bool ok = sdk_->InitIovHandle(handle, 4096, 5000, cuda_util);
    EXPECT_FALSE(ok);
    EXPECT_EQ(captured_size, 8192u);
}

TEST_F(Hf3fsSdkTest, InitIovHandle_ReturnFalse_MempoolInitFail_DestroyIov) {
    auto mock = std::dynamic_pointer_cast<MockHf3fsUsrbioApi>(sdk_->usrbio_api_);
    ASSERT_TRUE(mock != nullptr);
    EXPECT_CALL(*mock, Hf3fsIovCreate(::testing::NotNull(), ::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillOnce(::testing::Invoke([&](::hf3fs_iov *iov, const char *, size_t size, size_t, int) {
            // make iov invalid to force mempool init failure
            iov->base = nullptr;
            iov->size = size;
            return 0;
        }));
    EXPECT_CALL(*mock, Hf3fsIovDestroy(::testing::NotNull())).Times(1);

    Hf3fsIovHandle handle;
    auto cuda_util = std::make_shared<Hf3fsCudaUtil>();
    bool ok = sdk_->InitIovHandle(handle, 4096, 4096, cuda_util);
    EXPECT_FALSE(ok);
}

#ifdef USING_CUDA
TEST_F(Hf3fsSdkTest, InitIovHandle_ReturnTrue_Success) {
    auto mock = std::dynamic_pointer_cast<MockHf3fsUsrbioApi>(sdk_->usrbio_api_);
    ASSERT_TRUE(mock != nullptr);
    EXPECT_CALL(*mock, Hf3fsIovCreate(::testing::NotNull(), ::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillOnce(::testing::Invoke([&](::hf3fs_iov *iov, const char *, size_t size, size_t, int) {
            // simulate a valid created iov so mempool init succeeds
            iov->base = (uint8_t *)malloc(size);
            iov->size = size;
            return 0;
        }));
    Hf3fsIovHandle handle;
    auto cuda_util = std::make_shared<Hf3fsCudaUtil>();
    bool ok = sdk_->InitIovHandle(handle, 4096, 4096, cuda_util);
    EXPECT_TRUE(ok);

    EXPECT_CALL(*mock, Hf3fsIovDestroy(handle.iov)).WillOnce(::testing::Invoke([&](::hf3fs_iov *iov) {
        free(iov->base);
        iov->base = nullptr;
    }));
    sdk_->ReleaseIovHandle(handle);
    EXPECT_EQ(handle.iov, nullptr);
    EXPECT_EQ(handle.iov_mempool, nullptr);
    EXPECT_EQ(handle.cuda_util, nullptr);
}
#endif

// ------------- ReleaseIovHandle -------------
TEST_F(Hf3fsSdkTest, ReleaseIovHandle_ReturnNoop_NullIov) {
    Hf3fsIovHandle handle;
    // set non-null resources to verify they get reset
    handle.cuda_util = std::make_shared<Hf3fsCudaUtil>();
    handle.iov_mempool = std::make_shared<Hf3fsMempool>((void *)0x1, 16, 0);

    auto mock = std::dynamic_pointer_cast<MockHf3fsUsrbioApi>(sdk_->usrbio_api_);
    ASSERT_TRUE(mock != nullptr);
    EXPECT_CALL(*mock, Hf3fsIovDestroy(::testing::_)).Times(0);

    sdk_->ReleaseIovHandle(handle);

    EXPECT_EQ(handle.iov, nullptr);
    EXPECT_EQ(handle.iov_mempool, nullptr);
    EXPECT_EQ(handle.cuda_util, nullptr);
}

// ------------- CreateIov -------------
TEST_F(Hf3fsSdkTest, CreateIov_ReturnNull_EmptyMountpoint) {
    auto mock = std::dynamic_pointer_cast<MockHf3fsUsrbioApi>(sdk_->usrbio_api_);
    ASSERT_TRUE(mock != nullptr);
    EXPECT_CALL(*mock, Hf3fsIovCreate(::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_)).Times(0);
    auto *iov = sdk_->CreateIov("", 4096, 4096);
    EXPECT_EQ(iov, nullptr);
}

TEST_F(Hf3fsSdkTest, CreateIov_ReturnNull_InvalidSize) {
    auto mock = std::dynamic_pointer_cast<MockHf3fsUsrbioApi>(sdk_->usrbio_api_);
    ASSERT_TRUE(mock != nullptr);
    EXPECT_CALL(*mock, Hf3fsIovCreate(::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_)).Times(0);
    auto *iov = sdk_->CreateIov(mount_point_, 0, 4096);
    EXPECT_EQ(iov, nullptr);
}

TEST_F(Hf3fsSdkTest, CreateIov_ReturnNull_CreateFail) {
    auto mock = std::dynamic_pointer_cast<MockHf3fsUsrbioApi>(sdk_->usrbio_api_);
    ASSERT_TRUE(mock != nullptr);
    EXPECT_CALL(*mock, Hf3fsIovCreate(::testing::NotNull(), ::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillOnce(::testing::Return(-1));
    EXPECT_CALL(*mock, Hf3fsIovDestroy(::testing::_)).Times(0);
    auto *iov = sdk_->CreateIov(mount_point_, 8192, 4096);
    EXPECT_EQ(iov, nullptr);
}

TEST_F(Hf3fsSdkTest, CreateIov_ReturnPtr_Success) {
    auto mock = std::dynamic_pointer_cast<MockHf3fsUsrbioApi>(sdk_->usrbio_api_);
    ASSERT_TRUE(mock != nullptr);
    EXPECT_CALL(*mock, Hf3fsIovCreate(::testing::NotNull(), ::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillOnce(::testing::Return(0));
    auto *iov = sdk_->CreateIov(mount_point_, 16384, 4096);
    ASSERT_NE(iov, nullptr);
    EXPECT_CALL(*mock, Hf3fsIovDestroy(iov)).Times(1);
    sdk_->DestroyIov(iov);
}

// ------------- DestroyIov -------------
TEST_F(Hf3fsSdkTest, DestroyIov_ReturnNoop_Nullptr) {
    auto mock = std::dynamic_pointer_cast<MockHf3fsUsrbioApi>(sdk_->usrbio_api_);
    ASSERT_TRUE(mock != nullptr);
    EXPECT_CALL(*mock, Hf3fsIovDestroy(::testing::_)).Times(0);
    sdk_->DestroyIov(nullptr);
}

TEST_F(Hf3fsSdkTest, DestroyIov_ReturnVoid_Normal) {
    auto mock = std::dynamic_pointer_cast<MockHf3fsUsrbioApi>(sdk_->usrbio_api_);
    ASSERT_TRUE(mock != nullptr);
    auto *iov = new ::hf3fs_iov();
    EXPECT_CALL(*mock, Hf3fsIovDestroy(iov)).Times(1);
    sdk_->DestroyIov(iov);
}

// ------------- BuildHf3fsFileConfig -------------
TEST_F(Hf3fsSdkTest, BuildHf3fsFileConfig_ReturnValid) {
    auto filepath = mount_point_ + "/test.txt";
    auto cfg = sdk_->BuildHf3fsFileConfig(filepath);
    EXPECT_EQ(cfg.filepath, filepath);
    EXPECT_EQ(cfg.mountpoint, mount_point_);
}

// ------------- CreateDir -------------
TEST_F(Hf3fsSdkTest, CreateDir_ReturnTrue_Exists) {
    // use test runtime dir; it exists
    std::filesystem::path p(GetPrivateTestRuntimeDataPath());
    ASSERT_TRUE(std::filesystem::exists(p));
    EXPECT_TRUE(sdk_->CreateDir(p));
}

// ------------- GetFileOffset -------------
TEST_F(Hf3fsSdkTest, GetFileOffset_ReturnNullopt_SizeZero) {
    DataStorageUri uri;
    uri.SetParam("blkid", "10");
    uri.SetParam("size", "0");
    auto off = sdk_->GetFileOffset(uri);
    EXPECT_FALSE(off.has_value());
}

TEST_F(Hf3fsSdkTest, GetFileOffset_ReturnValue_Normal) {
    DataStorageUri uri;
    uri.SetParam("blkid", "3");
    uri.SetParam("size", "4096");
    auto off = sdk_->GetFileOffset(uri);
    ASSERT_TRUE(off.has_value());
    EXPECT_EQ(off.value(), 3ull * 4096ull);
}