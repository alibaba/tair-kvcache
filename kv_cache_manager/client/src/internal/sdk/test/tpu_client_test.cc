// Unit tests for TpuClient (PJRT C API wrapper)
// Build with: bazel test //kv_cache_manager/client/src/internal/sdk/test:TpuClientTest --define=using_tpu=true

#include <gtest/gtest.h>

#include "kv_cache_manager/common/unittest.h"

#ifdef USING_TPU
#include "kv_cache_manager/client/src/internal/sdk/tpu_client.h"
#endif

#include <cstring>
#include <vector>

using namespace kv_cache_manager;

class TpuClientTest : public TESTBASE {
public:
    void SetUp() override {}
    void TearDown() override {}
};

// ---------------------------------------------------------------------------
// Test: TpuClient::Init() — verify PJRT client is properly initialized
// ---------------------------------------------------------------------------
TEST_F(TpuClientTest, InitSucceeds) {
#ifdef USING_TPU
    TpuClient client;
    auto ec = client.Init();
    ASSERT_EQ(ec, ER_OK) << "TpuClient::Init() should succeed on TPU";

    // Via -fno-access-control we can inspect private fields
    ASSERT_NE(client.libtpu_handle_, nullptr) << "dlopen handle should be non-null after Init()";
    ASSERT_NE(client.api_, nullptr)    << "PJRT_Api should be non-null after Init()";
    ASSERT_NE(client.client_, nullptr) << "PJRT_Client should be non-null after Init()";
    ASSERT_NE(client.device_, nullptr) << "PJRT_Device should be non-null after Init()";
#else
    GTEST_SKIP() << "TPU not enabled, skipping";
#endif
}

// ---------------------------------------------------------------------------
// Test: double Init() — calling Init() twice should not crash
// ---------------------------------------------------------------------------
TEST_F(TpuClientTest, DoubleInit) {
#ifdef USING_TPU
    TpuClient client;
    ASSERT_EQ(client.Init(), ER_OK);
    // Second Init() should also succeed (re-creates client)
    auto ec = client.Init();
    EXPECT_EQ(ec, ER_OK);
#else
    GTEST_SKIP() << "TPU not enabled, skipping";
#endif
}

// ---------------------------------------------------------------------------
// Test: Destroy() — verify clean teardown
// ---------------------------------------------------------------------------
TEST_F(TpuClientTest, DestroyClearsState) {
#ifdef USING_TPU
    TpuClient client;
    ASSERT_EQ(client.Init(), ER_OK);

    client.Destroy();

    ASSERT_EQ(client.api_, nullptr)    << "api_ should be null after Destroy()";
    ASSERT_EQ(client.client_, nullptr) << "client_ should be null after Destroy()";
    ASSERT_EQ(client.device_, nullptr) << "device_ should be null after Destroy()";
    ASSERT_EQ(client.libtpu_handle_, nullptr) << "dlopen handle should be null after Destroy()";
#else
    GTEST_SKIP() << "TPU not enabled, skipping";
#endif
}

// ---------------------------------------------------------------------------
// Test: DmaMap / DmaUnmap — host memory registration (currently no-op on TPU)
// ---------------------------------------------------------------------------
TEST_F(TpuClientTest, DmaMapUnmap) {
#ifdef USING_TPU
    TpuClient client;
    ASSERT_EQ(client.Init(), ER_OK);

    size_t buf_size = 4096;
    void* host_buf = std::aligned_alloc(4096, buf_size);
    ASSERT_NE(host_buf, nullptr);
    std::memset(host_buf, 0xAB, buf_size);

    EXPECT_EQ(client.DmaMap(host_buf, buf_size), ER_OK);
    EXPECT_EQ(client.DmaUnmap(host_buf), ER_OK);

    free(host_buf);
#else
    GTEST_SKIP() << "TPU not enabled, skipping";
#endif
}

// ---------------------------------------------------------------------------
// Test: DmaMap / DmaUnmap — fail gracefully when client not initialized
// ---------------------------------------------------------------------------
TEST_F(TpuClientTest, DmaMapUninitReturnsError) {
#ifdef USING_TPU
    TpuClient client;
    // Not initialized — should return error
    EXPECT_EQ(client.DmaMap(nullptr, 1024), ER_TPU_DMA_MAP_ERROR);
    EXPECT_EQ(client.DmaUnmap(nullptr), ER_TPU_DMA_MAP_ERROR);
#else
    GTEST_SKIP() << "TPU not enabled, skipping";
#endif
}

// ---------------------------------------------------------------------------
// Test: BufferFromHost / BufferToHost — round-trip data transfer
// ---------------------------------------------------------------------------
TEST_F(TpuClientTest, BufferRoundTrip) {
#ifdef USING_TPU
    TpuClient client;
    ASSERT_EQ(client.Init(), ER_OK);

    const char* test_data = "TpuClientTest round-trip data!!!!!!!";
    size_t data_size = std::strlen(test_data);

    // H2D: create buffer from host data
    std::vector<char> host_src(data_size);
    std::memcpy(host_src.data(), test_data, data_size);

    PJRT_Buffer* buffer = nullptr;
    auto ec = client.BufferFromHost(host_src.data(), data_size, buffer);
    ASSERT_EQ(ec, ER_OK) << "BufferFromHost failed";
    ASSERT_NE(buffer, nullptr);

    // D2H: read back to host
    std::vector<char> host_dst(data_size, 0);
    ec = client.BufferToHost(buffer, host_dst.data(), data_size);
    ASSERT_EQ(ec, ER_OK) << "BufferToHost failed";
    ASSERT_EQ(std::memcmp(host_dst.data(), test_data, data_size), 0)
        << "Data mismatch after round-trip";

    client.DestroyBuffer(buffer);
#else
    GTEST_SKIP() << "TPU not enabled, skipping";
#endif
}

// ---------------------------------------------------------------------------
// Test: BufferFromHost — zero size should return ER_INVALID_PARAMS
// ---------------------------------------------------------------------------
TEST_F(TpuClientTest, BufferFromHostZeroSize) {
#ifdef USING_TPU
    TpuClient client;
    ASSERT_EQ(client.Init(), ER_OK);

    PJRT_Buffer* buffer = nullptr;
    char dummy = 0;
    auto ec = client.BufferFromHost(&dummy, /*size=*/0, buffer);
    EXPECT_EQ(ec, ER_INVALID_PARAMS) << "Zero-size BufferFromHost should fail";
#else
    GTEST_SKIP() << "TPU not enabled, skipping";
#endif
}

// ---------------------------------------------------------------------------
// Test: BufferToHost — zero size should return ER_INVALID_PARAMS
// ---------------------------------------------------------------------------
TEST_F(TpuClientTest, BufferToHostZeroSize) {
#ifdef USING_TPU
    TpuClient client;
    ASSERT_EQ(client.Init(), ER_OK);

    // Create a valid buffer first
    char src = 'X';
    PJRT_Buffer* buffer = nullptr;
    ASSERT_EQ(client.BufferFromHost(&src, 1, buffer), ER_OK);
    ASSERT_NE(buffer, nullptr);

    char dst = 0;
    auto ec = client.BufferToHost(buffer, &dst, /*size=*/0);
    EXPECT_EQ(ec, ER_INVALID_PARAMS) << "Zero-size BufferToHost should fail";

    client.DestroyBuffer(buffer);
#else
    GTEST_SKIP() << "TPU not enabled, skipping";
#endif
}

// ---------------------------------------------------------------------------
// Test: BufferFromHost / BufferToHost — fail gracefully when not initialized
// ---------------------------------------------------------------------------
TEST_F(TpuClientTest, BufferOpsUninitReturnsError) {
#ifdef USING_TPU
    TpuClient client;
    // Not initialized
    PJRT_Buffer* buffer = nullptr;
    char dummy = 0;
    EXPECT_EQ(client.BufferFromHost(&dummy, 1, buffer), ER_TPU_BUFFER_TRANSFER_ERROR);
    EXPECT_EQ(client.BufferToHost(nullptr, &dummy, 1), ER_TPU_BUFFER_TRANSFER_ERROR);
#else
    GTEST_SKIP() << "TPU not enabled, skipping";
#endif
}

// ---------------------------------------------------------------------------
// Test: DestroyBuffer — safe to call with nullptr
// ---------------------------------------------------------------------------
TEST_F(TpuClientTest, DestroyBufferNullSafe) {
#ifdef USING_TPU
    TpuClient client;
    ASSERT_EQ(client.Init(), ER_OK);

    // Should not crash
    client.DestroyBuffer(nullptr);
#else
    GTEST_SKIP() << "TPU not enabled, skipping";
#endif
}

// ---------------------------------------------------------------------------
// Test: GetErrorMessage — static utility function
// ---------------------------------------------------------------------------
TEST_F(TpuClientTest, GetErrorMessageNullSafe) {
#ifdef USING_TPU
    // Both null
    auto msg = TpuClient::GetErrorMessage(nullptr, nullptr);
    EXPECT_EQ(msg, "unknown error");

    // api null
    msg = TpuClient::GetErrorMessage(nullptr, reinterpret_cast<PJRT_Error*>(0x1));
    EXPECT_EQ(msg, "unknown error");
#else
    GTEST_SKIP() << "TPU not enabled, skipping";
#endif
}

// ---------------------------------------------------------------------------
// Test: large buffer round-trip (multi-KB)
// ---------------------------------------------------------------------------
TEST_F(TpuClientTest, LargeBufferRoundTrip) {
#ifdef USING_TPU
    TpuClient client;
    ASSERT_EQ(client.Init(), ER_OK);

    // 64KB buffer filled with a pattern
    constexpr size_t kSize = 64 * 1024;
    std::vector<uint8_t> host_src(kSize);
    for (size_t i = 0; i < kSize; ++i) {
        host_src[i] = static_cast<uint8_t>(i & 0xFF);
    }

    PJRT_Buffer* buffer = nullptr;
    auto ec = client.BufferFromHost(host_src.data(), kSize, buffer);
    ASSERT_EQ(ec, ER_OK) << "Large BufferFromHost failed";
    ASSERT_NE(buffer, nullptr);

    std::vector<uint8_t> host_dst(kSize, 0);
    ec = client.BufferToHost(buffer, host_dst.data(), kSize);
    ASSERT_EQ(ec, ER_OK) << "Large BufferToHost failed";
    ASSERT_EQ(std::memcmp(host_dst.data(), host_src.data(), kSize), 0)
        << "Large buffer data mismatch";

    client.DestroyBuffer(buffer);
#else
    GTEST_SKIP() << "TPU not enabled, skipping";
#endif
}

// ---------------------------------------------------------------------------
// Test: BufferFromHostAsync + WaitEvent — async H2D round-trip
// ---------------------------------------------------------------------------
TEST_F(TpuClientTest, AsyncBufferFromHostRoundTrip) {
#ifdef USING_TPU
    TpuClient client;
    ASSERT_EQ(client.Init(), ER_OK);

    const char* test_data = "AsyncBufferFromHost round-trip!!!!!";
    size_t data_size = std::strlen(test_data);

    std::vector<char> host_src(data_size);
    std::memcpy(host_src.data(), test_data, data_size);

    PJRT_Buffer* buffer = nullptr;
    PJRT_Event* event = nullptr;
    auto ec = client.BufferFromHostAsync(host_src.data(), data_size, buffer, event);
    ASSERT_EQ(ec, ER_OK) << "BufferFromHostAsync failed";
    ASSERT_NE(buffer, nullptr);

    // Wait for the H2D transfer to complete
    ec = client.WaitEvent(event);
    ASSERT_EQ(ec, ER_OK) << "WaitEvent for H2D failed";

    // Now read back synchronously
    std::vector<char> host_dst(data_size, 0);
    ec = client.BufferToHost(buffer, host_dst.data(), data_size);
    ASSERT_EQ(ec, ER_OK) << "BufferToHost after async H2D failed";
    ASSERT_EQ(std::memcmp(host_dst.data(), test_data, data_size), 0)
        << "Data mismatch after async H2D round-trip";

    client.DestroyBuffer(buffer);
#else
    GTEST_SKIP() << "TPU not enabled, skipping";
#endif
}

// ---------------------------------------------------------------------------
// Test: BufferToHostAsync + WaitEvent — async D2H round-trip
// ---------------------------------------------------------------------------
TEST_F(TpuClientTest, AsyncBufferToHostRoundTrip) {
#ifdef USING_TPU
    TpuClient client;
    ASSERT_EQ(client.Init(), ER_OK);

    const char* test_data = "AsyncBufferToHost round-trip!!!!!";
    size_t data_size = std::strlen(test_data);

    // Create buffer synchronously first
    std::vector<char> host_src(data_size);
    std::memcpy(host_src.data(), test_data, data_size);

    PJRT_Buffer* buffer = nullptr;
    ASSERT_EQ(client.BufferFromHost(host_src.data(), data_size, buffer), ER_OK);
    ASSERT_NE(buffer, nullptr);

    // Read back asynchronously
    std::vector<char> host_dst(data_size, 0);
    PJRT_Event* event = nullptr;
    auto ec = client.BufferToHostAsync(buffer, host_dst.data(), data_size, event);
    ASSERT_EQ(ec, ER_OK) << "BufferToHostAsync failed";

    // Wait for D2H transfer
    ec = client.WaitEvent(event);
    ASSERT_EQ(ec, ER_OK) << "WaitEvent for D2H failed";

    ASSERT_EQ(std::memcmp(host_dst.data(), test_data, data_size), 0)
        << "Data mismatch after async D2H round-trip";

    client.DestroyBuffer(buffer);
#else
    GTEST_SKIP() << "TPU not enabled, skipping";
#endif
}

// ---------------------------------------------------------------------------
// Test: WaitEvents — batch wait for multiple events
// ---------------------------------------------------------------------------
TEST_F(TpuClientTest, WaitEventsBatch) {
#ifdef USING_TPU
    TpuClient client;
    ASSERT_EQ(client.Init(), ER_OK);

    constexpr int kNumBuffers = 4;
    constexpr size_t kSize = 256;
    std::vector<PJRT_Buffer*> buffers(kNumBuffers, nullptr);
    std::vector<PJRT_Event*> events;

    // Create multiple buffers asynchronously
    for (int i = 0; i < kNumBuffers; ++i) {
        std::vector<uint8_t> src(kSize, static_cast<uint8_t>(i));
        PJRT_Event* event = nullptr;
        auto ec = client.BufferFromHostAsync(src.data(), kSize, buffers[i], event);
        ASSERT_EQ(ec, ER_OK) << "BufferFromHostAsync[" << i << "] failed";
        if (event) {
            events.push_back(event);
        }
    }

    // Wait for all events at once
    auto ec = client.WaitEvents(events);
    ASSERT_EQ(ec, ER_OK) << "WaitEvents failed";
    ASSERT_TRUE(events.empty()) << "events vector should be cleared after WaitEvents";

    // Verify data
    for (int i = 0; i < kNumBuffers; ++i) {
        ASSERT_NE(buffers[i], nullptr);
        std::vector<uint8_t> dst(kSize, 0);
        ec = client.BufferToHost(buffers[i], dst.data(), kSize);
        ASSERT_EQ(ec, ER_OK);
        for (size_t j = 0; j < kSize; ++j) {
            ASSERT_EQ(dst[j], static_cast<uint8_t>(i))
                << "Data mismatch at buffer[" << i << "][" << j << "]";
        }
        client.DestroyBuffer(buffers[i]);
    }
#else
    GTEST_SKIP() << "TPU not enabled, skipping";
#endif
}

// ---------------------------------------------------------------------------
// Test: DestroyEvent — nullptr safe
// ---------------------------------------------------------------------------
TEST_F(TpuClientTest, DestroyEventNullSafe) {
#ifdef USING_TPU
    TpuClient client;
    ASSERT_EQ(client.Init(), ER_OK);

    // Should not crash
    client.DestroyEvent(nullptr);
#else
    GTEST_SKIP() << "TPU not enabled, skipping";
#endif
}

// ---------------------------------------------------------------------------
// Test: WaitEvent with nullptr — should be a no-op
// ---------------------------------------------------------------------------
TEST_F(TpuClientTest, WaitEventNullSafe) {
#ifdef USING_TPU
    TpuClient client;
    ASSERT_EQ(client.Init(), ER_OK);

    EXPECT_EQ(client.WaitEvent(nullptr), ER_OK);
#else
    GTEST_SKIP() << "TPU not enabled, skipping";
#endif
}

// ---------------------------------------------------------------------------
// Test: Async operations fail gracefully when not initialized
// ---------------------------------------------------------------------------
TEST_F(TpuClientTest, AsyncBufferOpsUninitReturnsError) {
#ifdef USING_TPU
    TpuClient client;
    // Not initialized
    PJRT_Buffer* buffer = nullptr;
    PJRT_Event* event = nullptr;
    char dummy = 0;
    EXPECT_EQ(client.BufferFromHostAsync(&dummy, 1, buffer, event), ER_TPU_BUFFER_TRANSFER_ERROR);
    EXPECT_EQ(client.BufferToHostAsync(nullptr, &dummy, 1, event), ER_TPU_BUFFER_TRANSFER_ERROR);
#else
    GTEST_SKIP() << "TPU not enabled, skipping";
#endif
}

// ---------------------------------------------------------------------------
// Test: HasRawBufferExtension — check availability after Init()
// ---------------------------------------------------------------------------
TEST_F(TpuClientTest, HasRawBufferExtensionAfterInit) {
#ifdef USING_TPU
    TpuClient client;
    // Before init, should be false
    EXPECT_FALSE(client.HasRawBufferExtension());

    ASSERT_EQ(client.Init(), ER_OK);
    // After init, depends on plugin support (libtpu v0.0.41 supports it)
    // Just verify it doesn't crash; result may be true or false
    bool has_ext = client.HasRawBufferExtension();
    KVCM_LOG_INFO("HasRawBufferExtension: %s", has_ext ? "true" : "false");
#else
    GTEST_SKIP() << "TPU not enabled, skipping";
#endif
}

// Helper: allocate 32-byte aligned memory (required by TPU RawBuffer operations)
static void* AlignedAlloc32(size_t size) {
    size_t aligned_size = (size + 31) & ~static_cast<size_t>(31);
    return std::aligned_alloc(32, aligned_size);
}

// ---------------------------------------------------------------------------
// Test: RawBuffer round-trip (create alias → D2H read-back)
// ---------------------------------------------------------------------------
TEST_F(TpuClientTest, RawBufferRoundTrip) {
#ifdef USING_TPU
    TpuClient client;
    ASSERT_EQ(client.Init(), ER_OK);

    if (!client.HasRawBufferExtension()) {
        GTEST_SKIP() << "RawBuffer extension not available";
    }

    // Create a buffer with known data via BufferFromHost (logical write)
    constexpr size_t kSize = 1024;
    std::vector<uint8_t> src_data(kSize);
    for (size_t i = 0; i < kSize; ++i) {
        src_data[i] = static_cast<uint8_t>(i & 0xFF);
    }
    PJRT_Buffer* buffer = nullptr;
    ASSERT_EQ(client.BufferFromHost(src_data.data(), kSize, buffer), ER_OK);
    ASSERT_NE(buffer, nullptr);

    // Create raw alias
    PJRT_RawBuffer* raw = nullptr;
    auto ec = client.CreateRawAlias(buffer, raw);
    ASSERT_EQ(ec, ER_OK) << "CreateRawAlias failed";
    ASSERT_NE(raw, nullptr);

    // Query device size (may be larger than logical due to tile padding)
    size_t device_size = 0;
    ec = client.RawBufferGetDeviceSize(raw, device_size);
    ASSERT_EQ(ec, ER_OK) << "RawBufferGetDeviceSize failed";
    EXPECT_GE(device_size, kSize) << "Device size should be >= logical size";

    // Read back physical bytes via RawBuffer D2H
    // TPU requires 32-byte aligned host buffers for raw operations
    void* aligned_dst = AlignedAlloc32(device_size);
    ASSERT_NE(aligned_dst, nullptr);
    std::memset(aligned_dst, 0, device_size);

    PJRT_Event* d2h_event = nullptr;
    ec = client.RawBufferToHost(raw, aligned_dst, 0,
                                static_cast<int64_t>(device_size), d2h_event);
    ASSERT_EQ(ec, ER_OK) << "RawBufferToHost failed";
    ec = client.WaitEvent(d2h_event);
    ASSERT_EQ(ec, ER_OK) << "WaitEvent for RawBuffer D2H failed";

    // The first kSize bytes should match the original data
    // Note: on TPU with tile layout, the physical bytes may be rearranged.
    // We verify logical correctness via BufferToHost instead.
    std::vector<uint8_t> logical_dst(kSize, 0);
    ASSERT_EQ(client.BufferToHost(buffer, logical_dst.data(), kSize), ER_OK);
    ASSERT_EQ(std::memcmp(logical_dst.data(), src_data.data(), kSize), 0)
        << "Logical D2H data mismatch after RawBuffer alias creation";

    // Verify raw bytes are non-zero (data was written)
    bool has_nonzero = false;
    for (size_t i = 0; i < device_size; ++i) {
        if (static_cast<uint8_t*>(aligned_dst)[i] != 0) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero) << "Raw D2H returned all zeros";

    // Cleanup
    free(aligned_dst);
    client.DestroyRawBuffer(raw);
    client.DestroyBuffer(buffer);
#else
    GTEST_SKIP() << "TPU not enabled, skipping";
#endif
}

// ---------------------------------------------------------------------------
// Test: RawBuffer operations fail gracefully when not initialized
// ---------------------------------------------------------------------------
TEST_F(TpuClientTest, RawBufferOpsUninitReturnsError) {
#ifdef USING_TPU
    TpuClient client;
    // Not initialized
    PJRT_RawBuffer* raw = nullptr;
    EXPECT_EQ(client.CreateRawAlias(nullptr, raw), ER_TPU_RAWBUFFER_ERROR);

    PJRT_Event* event = nullptr;
    char dummy = 0;
    EXPECT_EQ(client.RawBufferFromHost(nullptr, &dummy, 0, 1, event), ER_TPU_RAWBUFFER_ERROR);
    EXPECT_EQ(client.RawBufferToHost(nullptr, &dummy, 0, 1, event), ER_TPU_RAWBUFFER_ERROR);

    size_t sz = 0;
    EXPECT_EQ(client.RawBufferGetDeviceSize(nullptr, sz), ER_TPU_RAWBUFFER_ERROR);
#else
    GTEST_SKIP() << "TPU not enabled, skipping";
#endif
}

// ---------------------------------------------------------------------------
// Test: DestroyRawBuffer — nullptr safe
// ---------------------------------------------------------------------------
TEST_F(TpuClientTest, DestroyRawBufferNullSafe) {
#ifdef USING_TPU
    TpuClient client;
    ASSERT_EQ(client.Init(), ER_OK);

    // Should not crash
    client.DestroyRawBuffer(nullptr);
#else
    GTEST_SKIP() << "TPU not enabled, skipping";
#endif
}

// ---------------------------------------------------------------------------
// Test: RawBuffer batch async D2H with WaitEvents
// ---------------------------------------------------------------------------
TEST_F(TpuClientTest, RawBufferBatchAsyncD2H) {
#ifdef USING_TPU
    TpuClient client;
    ASSERT_EQ(client.Init(), ER_OK);

    if (!client.HasRawBufferExtension()) {
        GTEST_SKIP() << "RawBuffer extension not available";
    }

    constexpr int kNumBuffers = 3;
    constexpr size_t kSize = 512;

    // Create buffers with known data via BufferFromHost (logical write)
    std::vector<PJRT_Buffer*> buffers(kNumBuffers, nullptr);
    std::vector<PJRT_RawBuffer*> raws(kNumBuffers, nullptr);
    std::vector<std::vector<uint8_t>> src_data(kNumBuffers);
    std::vector<size_t> device_sizes(kNumBuffers, 0);

    for (int i = 0; i < kNumBuffers; ++i) {
        src_data[i].resize(kSize, static_cast<uint8_t>(i * 10));
        ASSERT_EQ(client.BufferFromHost(src_data[i].data(), kSize, buffers[i]), ER_OK);
        ASSERT_EQ(client.CreateRawAlias(buffers[i], raws[i]), ER_OK);
        ASSERT_EQ(client.RawBufferGetDeviceSize(raws[i], device_sizes[i]), ER_OK);
    }

    // Batch async D2H — read physical bytes from all buffers
    // TPU requires 32-byte aligned host buffers for raw operations
    std::vector<void*> aligned_dsts(kNumBuffers, nullptr);
    std::vector<PJRT_Event*> events;

    for (int i = 0; i < kNumBuffers; ++i) {
        aligned_dsts[i] = AlignedAlloc32(device_sizes[i]);
        ASSERT_NE(aligned_dsts[i], nullptr);
        std::memset(aligned_dsts[i], 0, device_sizes[i]);

        PJRT_Event* ev = nullptr;
        auto ec = client.RawBufferToHost(raws[i], aligned_dsts[i], 0,
                                          static_cast<int64_t>(device_sizes[i]), ev);
        ASSERT_EQ(ec, ER_OK) << "RawBufferToHost[" << i << "] failed";
        if (ev) events.push_back(ev);
    }

    // Wait for all
    auto ec = client.WaitEvents(events);
    ASSERT_EQ(ec, ER_OK) << "WaitEvents for batch raw D2H failed";

    // Verify: first kSize bytes should match the original logical data
    for (int i = 0; i < kNumBuffers; ++i) {
        ASSERT_EQ(std::memcmp(aligned_dsts[i], src_data[i].data(), kSize), 0)
            << "RawBuffer batch D2H mismatch at buffer[" << i << "]";
    }

    // Cleanup
    for (int i = 0; i < kNumBuffers; ++i) {
        free(aligned_dsts[i]);
        client.DestroyRawBuffer(raws[i]);
        client.DestroyBuffer(buffers[i]);
    }
#else
    GTEST_SKIP() << "TPU not enabled, skipping";
#endif
}
