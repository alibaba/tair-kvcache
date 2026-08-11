#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <gtest/gtest.h>
#include <unistd.h>

#include "kv_cache_manager/client/src/internal/sdk/deadline_util.h"
#include "kv_cache_manager/client/src/internal/sdk/local_file_sdk.h"
#include "kv_cache_manager/common/unittest.h"
#ifdef USING_CUDA
#include <cuda_runtime.h>
#endif

using namespace kv_cache_manager;

// 说明：GPU 相关用例（TestPutGetWithGpu 等）需要 GPU 环境与
// --config=client_with_cuda 构建（默认 --config=client 无 CUDA，用例自动跳过）；
// 无 CUDA 环境只保证非 GPU 用例 + 编译通过。

class LocalFileSdkTest : public TESTBASE {
public:
    void SetUp() override {
        root_path_ = GetPrivateTestRuntimeDataPath();
        sdk_backend_config_ = std::make_shared<NfsSdkConfig>();
        sdk_backend_config_->set_spec_byte_sizes_per_block({{"default", 1024}});
    }
    void TearDown() override {}

private:
    std::string root_path_;
    std::shared_ptr<NfsSdkConfig> sdk_backend_config_;
};

namespace {

DataStorageUri MakeUri(const std::string &file_path, uint64_t blkid, size_t size = 1024) {
    DataStorageUri uri("file://" + file_path);
    uri.SetParam("blkid", std::to_string(blkid));
    uri.SetParam("size", std::to_string(size));
    return uri;
}

// 稀疏文件：只分配空间不写数据，读回全零；比逐字节写文件快几个数量级。
void CreateSparseFile(const std::string &file_path, size_t size) {
    std::filesystem::create_directories(std::filesystem::path(file_path).parent_path());
    int fd = ::open(file_path.c_str(), O_CREAT | O_RDWR, 0644);
    ASSERT_TRUE(fd >= 0) << "open " << file_path << " failed";
    ASSERT_EQ(::fallocate(fd, 0, 0, static_cast<off_t>(size)), 0) << "fallocate " << file_path << " failed";
    ::close(fd);
}

BlockBuffer MakeCpuBuffer(size_t size, unsigned char fill) {
    BlockBuffer buf;
    Iov iov;
    iov.base = malloc(size);
    std::memset(iov.base, fill, size);
    iov.size = size;
    iov.type = MemoryType::CPU;
    iov.ignore = false;
    buf.iovs.push_back(iov);
    return buf;
}

void AssertBufferAllBytes(const BlockBuffer &buf, unsigned char expect) {
    for (const auto &iov : buf.iovs) {
        const auto *p = static_cast<const unsigned char *>(iov.base);
        for (size_t j = 0; j < iov.size; ++j) {
            ASSERT_EQ(p[j], expect);
        }
    }
}

void FreeBuffers(BlockBuffers &buffers) {
    for (auto &buf : buffers) {
        for (auto &iov : buf.iovs) {
            free(iov.base);
        }
    }
}

} // namespace

TEST_F(LocalFileSdkTest, TestInit) {
    LocalFileSdk sdk;
    ASSERT_EQ(ER_INVALID_SDKBACKEND_CONFIG, sdk.Init(nullptr, nullptr));
    sdk_backend_config_->set_spec_byte_sizes_per_block({});
    ASSERT_EQ(ER_INVALID_SDKBACKEND_CONFIG, sdk.Init(sdk_backend_config_, nullptr));
    sdk_backend_config_->set_spec_byte_sizes_per_block({{"default", 1024}});
    ASSERT_EQ(ER_OK, sdk.Init(sdk_backend_config_, nullptr));
}

TEST_F(LocalFileSdkTest, TestPutGetWithCpu) {
    LocalFileSdk sdk;
    ASSERT_EQ(ER_OK, sdk.Init(sdk_backend_config_, nullptr));
    // put
    DataStorageUri uri("file://" + root_path_ + "/local_file/test.txt");
    uri.SetParam("blkid", "0");
    uri.SetParam("size", "1024");
    DataStorageUri invalid_uri = uri;
    invalid_uri.SetParam("size", "1023");
    BlockBuffer buf;

    const char *test_data = "this is local file test";
    size_t len1 = strlen(test_data);
    const char *test_data_2 = "and this is another local file test";
    size_t len2 = strlen(test_data_2);

    auto put_buffer = malloc(1024 * 1024);
    std::memcpy(put_buffer, test_data, len1);
    std::memcpy(static_cast<char *>(put_buffer) + len1, test_data_2, len2);

    Iov iov1;
    iov1.base = put_buffer;
    iov1.size = len1;
    iov1.type = MemoryType::CPU;
    iov1.ignore = false;
    buf.iovs.push_back(iov1);

    Iov iov2;
    iov2.base = static_cast<char *>(put_buffer) + iov1.size; // pointer arithmetic ok
    iov2.size = len2;
    iov2.type = MemoryType::CPU;
    iov2.ignore = false;
    buf.iovs.push_back(iov2);

    const std::vector<DataStorageUri> &remote_uris = {uri};
    const std::vector<DataStorageUri> &invalid_remote_uris = {invalid_uri};
    BlockBuffers local_buffers = {buf};

    auto actual_remote_uris = std::make_shared<std::vector<DataStorageUri>>();
    ASSERT_EQ(ER_SDKWRITE_ERROR, sdk.Put(invalid_remote_uris, local_buffers, actual_remote_uris, /*deadline_ms=*/0));
    ASSERT_EQ(ER_OK, sdk.Put(remote_uris, local_buffers, actual_remote_uris, /*deadline_ms=*/0));
    ASSERT_EQ(actual_remote_uris->size(), 1);
    ASSERT_EQ(actual_remote_uris->at(0).ToUriString(), uri.ToUriString());
    free(put_buffer);

    // get
    auto get_buffer = malloc(1024 * 1024);
    size_t offset = 0;
    for (auto &iov : local_buffers[0].iovs) {
        iov.base = static_cast<char *>(get_buffer) + offset;
        offset += iov.size;
    }
    ASSERT_EQ(ER_SDKREAD_ERROR, sdk.Get(invalid_remote_uris, local_buffers, /*deadline_ms=*/0));
    ASSERT_EQ(ER_OK, sdk.Get(remote_uris, local_buffers, /*deadline_ms=*/0));
    auto &iov1_res = local_buffers[0].iovs[0];
    ASSERT_EQ(std::memcmp(iov1_res.base, test_data, iov1_res.size), 0);
    auto &iov2_res = local_buffers[0].iovs[1];
    ASSERT_EQ(std::memcmp(iov2_res.base, test_data_2, iov2_res.size), 0);
    free(get_buffer);
}

TEST_F(LocalFileSdkTest, TestPutGetWithGpu) {
#ifdef USING_CUDA
    LocalFileSdk sdk;
    ASSERT_EQ(ER_OK, sdk.Init(sdk_backend_config_, nullptr));
    // put
    DataStorageUri uri("file://" + root_path_ + "/local_file/test_gpu.txt");
    uri.SetParam("blkid", "0");
    uri.SetParam("size", "1024");
    DataStorageUri invalid_uri = uri;
    invalid_uri.SetParam("size", "1023");

    BlockBuffer buf;
    const char *test_data = "this is local file test";
    size_t len1 = strlen(test_data);
    const char *test_data_2 = "and this is another local file test";
    size_t len2 = strlen(test_data_2);

    void *host_put_buffer = malloc(len1 + len2);
    std::memcpy(static_cast<char *>(host_put_buffer), test_data, len1);
    std::memcpy(static_cast<char *>(host_put_buffer) + len1, test_data_2, len2);

    void *gpu_put_buffer = nullptr;
    ASSERT_EQ(cudaMalloc(&gpu_put_buffer, len1 + len2), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(gpu_put_buffer, host_put_buffer, len1 + len2, cudaMemcpyHostToDevice), cudaSuccess);

    Iov iov1;
    iov1.base = gpu_put_buffer;
    iov1.size = len1;
    iov1.type = MemoryType::GPU;
    iov1.ignore = false;
    buf.iovs.push_back(iov1);

    Iov iov2;
    iov2.base = static_cast<char *>(gpu_put_buffer) + len1;
    iov2.size = len2;
    iov2.type = MemoryType::GPU;
    iov2.ignore = false;
    buf.iovs.push_back(iov2);

    const std::vector<DataStorageUri> &remote_uris = {uri};
    const std::vector<DataStorageUri> &invalid_remote_uris = {invalid_uri};
    BlockBuffers local_buffers = {buf};

    // put
    auto actual_remote_uris = std::make_shared<std::vector<DataStorageUri>>();
    ASSERT_EQ(ER_SDKWRITE_ERROR, sdk.Put(invalid_remote_uris, local_buffers, actual_remote_uris, /*deadline_ms=*/0));
    ASSERT_EQ(ER_OK, sdk.Put(remote_uris, local_buffers, actual_remote_uris, /*deadline_ms=*/0));
    ASSERT_EQ(actual_remote_uris->size(), 1);
    ASSERT_EQ(actual_remote_uris->at(0).ToUriString(), uri.ToUriString());

    free(host_put_buffer);
    cudaFree(gpu_put_buffer);

    // get
    void *gpu_get_buffer = nullptr;
    ASSERT_EQ(cudaMalloc(&gpu_get_buffer, len1 + len2), cudaSuccess);

    size_t offset = 0;
    for (auto &iov : local_buffers[0].iovs) {
        iov.base = static_cast<char *>(gpu_get_buffer) + offset;
        offset += iov.size;
    }

    ASSERT_EQ(ER_SDKREAD_ERROR, sdk.Get(invalid_remote_uris, local_buffers, /*deadline_ms=*/0));
    ASSERT_EQ(ER_OK, sdk.Get(remote_uris, local_buffers, /*deadline_ms=*/0));

    void *host_get_buffer = malloc(len1 + len2);
    ASSERT_EQ(cudaMemcpy(host_get_buffer, gpu_get_buffer, len1 + len2, cudaMemcpyDeviceToHost), cudaSuccess);

    auto &iov1_res = local_buffers[0].iovs[0];
    ASSERT_EQ(std::memcmp(host_get_buffer, test_data, iov1_res.size), 0);
    auto &iov2_res = local_buffers[0].iovs[1];
    ASSERT_EQ(std::memcmp(static_cast<char *>(host_get_buffer) + len1, test_data_2, iov2_res.size), 0);

    free(host_get_buffer);
    cudaFree(gpu_get_buffer);
#else
    GTEST_SKIP() << "CUDA not enabled, skipping GPU buffer test";
#endif
}

// 验收 1（docs/design/client_sdk_io_contract.md）：已过期的 deadline 必须让 Get
// 在组级准入处立即返回超时，且一个 block 都不搬（caller buffer 哨兵值完好）。
TEST_F(LocalFileSdkTest, TestGetTimeoutStopsEarly) {
    constexpr size_t kBlockSize = 1024;
    constexpr size_t kNumBlocks = 4;
    std::string file_path = root_path_ + "/local_file/timeout_early.txt";
    CreateSparseFile(file_path, kNumBlocks * kBlockSize);

    std::vector<DataStorageUri> remote_uris;
    BlockBuffers buffers;
    for (uint64_t i = 0; i < kNumBlocks; ++i) {
        remote_uris.push_back(MakeUri(file_path, i));
        buffers.push_back(MakeCpuBuffer(kBlockSize, 0xA5)); // 哨兵值
    }

    LocalFileSdk sdk;
    ASSERT_EQ(ER_OK, sdk.Init(sdk_backend_config_, nullptr));
    {
        // 已过期的 deadline：任何 I/O 都不应发起。
        ASSERT_EQ(ER_SDK_TIMEOUT, sdk.Get(remote_uris, buffers, SteadyClockMs() - 1'000));
    }
    for (const auto &buf : buffers) {
        AssertBufferAllBytes(buf, 0xA5); // 一个 block 都没搬
    }
    FreeBuffers(buffers);
}

// 验收 2（可选但推荐）：很短但未过期的 deadline，构造"搬了一部分就超时"的中间态。
// 布局：块 0..7 为 1KB 快块（预算内必被搬完，读稀疏文件得全零）；块 8 为 512MB 大块
// （首次触碰稀疏文件 + 写 512MB 目标页，至少 ~10ms，预算 8ms 内必然搬不完）；
// 块 9..16 为 1KB 哨兵块（超时后必须保持完好）。
// 预算 8ms 的选择依据：远大于 open/mmap + 8KB 头块搬运（~0.1-0.5ms），
// 远小于 512MB 搬运时间，两个方向的裕度都 >10 倍，避免机器快慢导致抖动。
TEST_F(LocalFileSdkTest, TestGetTimeoutMidway) {
    constexpr size_t kBlockSize = 1024;
    constexpr size_t kHeadBlocks = 8;
    constexpr size_t kTailBlocks = 8;
    constexpr size_t kBigBlockSize = 512 * 1024 * 1024;
    std::string file_path = root_path_ + "/local_file/timeout_midway.txt";
    size_t file_size = kHeadBlocks * kBlockSize + kBigBlockSize + kTailBlocks * kBlockSize;
    CreateSparseFile(file_path, file_size);

    std::vector<DataStorageUri> remote_uris;
    BlockBuffers buffers;
    for (uint64_t i = 0; i < kHeadBlocks; ++i) {
        remote_uris.push_back(MakeUri(file_path, i));
        buffers.push_back(MakeCpuBuffer(kBlockSize, 0x5A)); // 头块：会被搬成 0x00
    }
    remote_uris.push_back(MakeUri(file_path, kHeadBlocks));
    buffers.push_back(MakeCpuBuffer(kBigBlockSize, 0x5A)); // 大块：预算内搬不完
    for (uint64_t i = kHeadBlocks + 1; i < kHeadBlocks + 1 + kTailBlocks; ++i) {
        remote_uris.push_back(MakeUri(file_path, i));
        buffers.push_back(MakeCpuBuffer(kBlockSize, 0xA5)); // 尾块：必须保持哨兵完好
    }

    LocalFileSdk sdk;
    ASSERT_EQ(ER_OK, sdk.Init(sdk_backend_config_, nullptr));
    { ASSERT_EQ(ER_SDK_TIMEOUT, sdk.Get(remote_uris, buffers, SteadyClockMs() + 8)); }
    // 头块已搬（稀疏文件读回全零）→ 证明是"中途停下"而不是"入口就返回"。
    // 注意：仅非 CUDA 构建可断言 —— GPU 构建下 Init 会对 512MB mmap 做
    // cudaHostRegister（本机实测 ~224ms），预算在第一个 block 之前就被注册消耗掉，
    // 此时"零块已搬"同样是正确的准入行为（入口即超时场景由 TestGetTimeoutStopsEarly 覆盖）。
#ifndef USING_CUDA
    for (size_t i = 0; i < kHeadBlocks; ++i) {
        AssertBufferAllBytes(buffers[i], 0x00);
    }
#endif
    // 尾块哨兵完好 → 证明超时后没有继续搬运（逐 block 检查真的停下来了）。
    for (size_t i = kHeadBlocks + 1; i < buffers.size(); ++i) {
        AssertBufferAllBytes(buffers[i], 0xA5);
    }
    FreeBuffers(buffers);
}

// 对称场景：已过期的 deadline 必须让 Put 在组级准入处立即返回超时，
// 且不执行 Alloc（文件不被创建）、不写数据。
TEST_F(LocalFileSdkTest, TestPutTimeoutStopsEarly) {
    constexpr size_t kBlockSize = 1024;
    std::string file_path = root_path_ + "/local_file/timeout_put_early.txt";
    std::filesystem::remove(file_path); // 确保文件不存在，验证 Alloc 未被调用

    std::vector<DataStorageUri> remote_uris = {MakeUri(file_path, 0)};
    BlockBuffers buffers = {MakeCpuBuffer(kBlockSize, 0x5A)};

    LocalFileSdk sdk;
    ASSERT_EQ(ER_OK, sdk.Init(sdk_backend_config_, nullptr));
    auto actual_remote_uris = std::make_shared<std::vector<DataStorageUri>>();
    { ASSERT_EQ(ER_SDK_TIMEOUT, sdk.Put(remote_uris, buffers, actual_remote_uris, SteadyClockMs() - 1'000)); }
    // 准入拦截发生在 Alloc/写文件之前。
    ASSERT_FALSE(std::filesystem::exists(file_path));
    FreeBuffers(buffers);
}

// 验收 3（F3 保序 + docs/design/client_sdk_io_contract.md）：交错多 path（A0,B0,A1,B1,A2）输入，
// 断言 actual_remote_uris[i] 逐一等于 remote_uris[i]（下标即身份），
// 且数据 Get 回来逐块与写入内容一致（跨 path 落位正确）。
TEST_F(LocalFileSdkTest, TestPutActualUrisOrderWithInterleavedPaths) {
    std::string file_a = root_path_ + "/local_file/interleaved_a.txt";
    std::string file_b = root_path_ + "/local_file/interleaved_b.txt";
    std::filesystem::remove(file_a);
    std::filesystem::remove(file_b);

    struct BlockSpec {
        std::string file_path;
        uint64_t blkid;
        const char *payload;
    };
    // 交错输入：同一 path 的 block 在入参中不连续 —— 旧实现按 map 迭代序 append 必然错位。
    std::vector<BlockSpec> blocks = {
        {file_a, 0, "payload-A0-0123456789"},
        {file_b, 0, "payload-B0-0123456789"},
        {file_a, 1, "payload-A1-0123456789"},
        {file_b, 1, "payload-B1-0123456789"},
        {file_a, 2, "payload-A2-0123456789"},
    };

    std::vector<DataStorageUri> remote_uris;
    BlockBuffers buffers;
    for (const auto &b : blocks) {
        remote_uris.push_back(MakeUri(b.file_path, b.blkid));
        BlockBuffer buf;
        Iov iov;
        size_t len = std::strlen(b.payload) + 1; // 含结尾 \0，便于 memcmp
        iov.base = malloc(len);
        std::memcpy(iov.base, b.payload, len);
        iov.size = len;
        iov.type = MemoryType::CPU;
        iov.ignore = false;
        buf.iovs.push_back(iov);
        buffers.push_back(buf);
    }

    LocalFileSdk sdk;
    ASSERT_EQ(ER_OK, sdk.Init(sdk_backend_config_, nullptr));
    auto actual_remote_uris = std::make_shared<std::vector<DataStorageUri>>();
    ASSERT_EQ(ER_OK, sdk.Put(remote_uris, buffers, actual_remote_uris, /*deadline_ms=*/0));
    ASSERT_EQ(actual_remote_uris->size(), remote_uris.size());
    // 保序契约：actual_remote_uris[i] 必须对应 remote_uris[i]。
    for (size_t i = 0; i < remote_uris.size(); ++i) {
        ASSERT_EQ(actual_remote_uris->at(i).ToUriString(), remote_uris[i].ToUriString());
    }
    // 数据按 blkid 落位：Get 回来逐块校验内容（含跨 path 交错）。
    for (auto &buf : buffers) {
        for (auto &iov : buf.iovs) {
            std::memset(iov.base, 0, iov.size);
        }
    }
    ASSERT_EQ(ER_OK, sdk.Get(remote_uris, buffers, /*deadline_ms=*/0));
    for (size_t i = 0; i < blocks.size(); ++i) {
        size_t len = std::strlen(blocks[i].payload) + 1;
        ASSERT_EQ(std::memcmp(buffers[i].iovs[0].base, blocks[i].payload, len), 0)
            << "block " << i << " content mismatch";
    }
    FreeBuffers(buffers);
}

// 验收 5（GPU，需 --config=client_with_cuda + GPU）：abort 路径返回后 stream 必须已同步。
// 场景：块 0 的 256MB GPU async copy 已入队（DMA 需数 ms），块 1 URI size 非法触发
// 既有错误分支提前返回 —— 断言返回后 SDK 私有 stream 空闲（cudaStreamQuery 成功），
// 证明 GpuStreamDrainGuard 在 munmap 之前完成了 cudaStreamSynchronize。
// 若 guard 缺失，此处 256MB DMA 仍在飞，cudaStreamQuery 必然返回 cudaErrorNotReady。
TEST_F(LocalFileSdkTest, TestGpuAbortPathDrainsStream) {
#ifdef USING_CUDA
    constexpr size_t kBigBlockSize = 256 * 1024 * 1024;
    std::string file_path = root_path_ + "/local_file/gpu_abort.txt";
    CreateSparseFile(file_path, kBigBlockSize);

    LocalFileSdk sdk;
    if (sdk.Init(sdk_backend_config_, nullptr) != ER_OK) {
        GTEST_SKIP() << "no usable GPU, skipping GPU abort drain test";
    }
    std::vector<DataStorageUri> remote_uris;
    BlockBuffers buffers;
    // 块 0：合法大块，GPU async copy 入队。
    remote_uris.push_back(MakeUri(file_path, 0));
    void *gpu_buf0 = nullptr;
    if (cudaMalloc(&gpu_buf0, kBigBlockSize) != cudaSuccess) {
        GTEST_SKIP() << "no usable GPU, skipping GPU abort drain test";
    }
    {
        BlockBuffer buf;
        Iov iov;
        iov.base = gpu_buf0;
        iov.size = kBigBlockSize;
        iov.type = MemoryType::GPU;
        iov.ignore = false;
        buf.iovs.push_back(iov);
        buffers.push_back(buf);
    }
    // 块 1：URI size 非法（不在 spec 内）→ 块 0 入队后必然提前返回 ER_INVALID_PARAMS。
    remote_uris.push_back(MakeUri(file_path, 1, /*size=*/1023));
    void *gpu_buf1 = nullptr;
    ASSERT_EQ(cudaMalloc(&gpu_buf1, 4096), cudaSuccess);
    {
        BlockBuffer buf;
        Iov iov;
        iov.base = gpu_buf1;
        iov.size = 4096;
        iov.type = MemoryType::GPU;
        iov.ignore = false;
        buf.iovs.push_back(iov);
        buffers.push_back(buf);
    }

    ASSERT_EQ(ER_SDKREAD_ERROR, sdk.Get(remote_uris, buffers, /*deadline_ms=*/0));
    // hard 契约：返回后 SDK stream 必须已同步（否则 256MB DMA 仍在写 caller 显存）。
    ASSERT_EQ(cudaStreamQuery(sdk.cuda_stream_), cudaSuccess);

    cudaFree(gpu_buf0);
    cudaFree(gpu_buf1);
#else
    GTEST_SKIP() << "CUDA not enabled, skipping GPU abort drain test";
#endif
}
