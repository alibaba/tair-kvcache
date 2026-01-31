#include <gtest/gtest.h>

#include "kv_cache_manager/client/src/internal/sdk/hf3fs_mempool.h"
#include "kv_cache_manager/common/unittest.h"

using namespace ::testing;
using namespace kv_cache_manager;

class Hf3fsMempoolTest : public ::testing::Test {
public:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(Hf3fsMempoolTest, Init_ReturnFalse_BufferNullOrZeroSize) {
    Hf3fsMempool pool(nullptr, 0, 0);
    EXPECT_FALSE(pool.Init());
}

TEST_F(Hf3fsMempoolTest, Init_ReturnTrue_BufferValid) {
    std::vector<uint8_t> buf(1024);
    Hf3fsMempool pool(buf.data(), buf.size(), 0);
    EXPECT_TRUE(pool.Init());
}

TEST_F(Hf3fsMempoolTest, Alloc_ReturnNullptr_SizeZero) {
    std::vector<uint8_t> buf(1024);
    Hf3fsMempool pool(buf.data(), buf.size(), 0);
    ASSERT_TRUE(pool.Init());
    EXPECT_EQ(pool.Alloc(0), nullptr);
    EXPECT_EQ(pool.FreeSize(), 1024);
}

TEST_F(Hf3fsMempoolTest, Alloc_ReturnNonNull_BestFitExactAndSplit) {
    std::vector<uint8_t> buf(1024);
    Hf3fsMempool pool(buf.data(), buf.size(), 0);
    ASSERT_TRUE(pool.Init());

    // 第一次分配 256
    void *p1 = pool.Alloc(256);
    ASSERT_NE(p1, nullptr);
    ASSERT_EQ(pool.FreeSize(), 1024 - 256);
    ASSERT_EQ(pool.AllocatedBlockCount(), 1);
    ASSERT_EQ(pool.FreeBlockCount(), 1);

    // 第二次分配 128
    void *p2 = pool.Alloc(128);
    ASSERT_NE(p2, nullptr);
    ASSERT_EQ(pool.FreeSize(), 1024 - 256 - 128);
    ASSERT_EQ(pool.AllocatedBlockCount(), 2);
    ASSERT_EQ(pool.FreeBlockCount(), 1);

    // 第三次分配 256
    void *p3 = pool.Alloc(256);
    ASSERT_NE(p3, nullptr);
    ASSERT_EQ(pool.FreeSize(), 1024 - 256 - 128 - 256);
    ASSERT_EQ(pool.AllocatedBlockCount(), 3);
    ASSERT_EQ(pool.FreeBlockCount(), 1);

    // 释放 p2 后再次分配 128, 验证从较小空闲块分配
    pool.Free(p2);
    ASSERT_EQ(pool.FreeSize(), 1024 - 256 - 256);
    ASSERT_EQ(pool.AllocatedBlockCount(), 2);
    ASSERT_EQ(pool.FreeBlockCount(), 2);

    void *p4 = pool.Alloc(128);
    ASSERT_NE(p4, nullptr);
    ASSERT_EQ(pool.FreeSize(), 1024 - 256 - 256 - 128);
    ASSERT_EQ(pool.AllocatedBlockCount(), 3);
    ASSERT_EQ(pool.FreeBlockCount(), 1);
}

TEST_F(Hf3fsMempoolTest, Alloc_ReturnNonNull_WithAlignmentRoundUp) {
    std::vector<uint8_t> buf(1024);
    Hf3fsMempool pool(buf.data(), buf.size(), 128);
    ASSERT_TRUE(pool.Init());
    ASSERT_EQ(pool.AllocatedBlockCount(), 0);
    ASSERT_EQ(pool.FreeBlockCount(), 1);

    void *p1 = pool.Alloc(100); // 向上对齐为 128
    ASSERT_NE(p1, nullptr);
    ASSERT_EQ(pool.FreeSize(), 1024 - 128);
    ASSERT_EQ(pool.AllocatedBlockCount(), 1);
    ASSERT_EQ(pool.FreeBlockCount(), 1);

    void *p2 = pool.Alloc(129); // 向上对齐为 256
    ASSERT_NE(p2, nullptr);
    ASSERT_EQ(pool.FreeSize(), 1024 - 128 - 256);
    ASSERT_EQ(pool.AllocatedBlockCount(), 2);
    ASSERT_EQ(pool.FreeBlockCount(), 1);

    void *p3 = pool.Alloc(128); // 已经对齐
    ASSERT_NE(p3, nullptr);
    ASSERT_EQ(pool.FreeSize(), 1024 - 128 - 256 - 128);
    ASSERT_EQ(pool.AllocatedBlockCount(), 3);
    ASSERT_EQ(pool.FreeBlockCount(), 1);

    pool.Free(p1);
    pool.Free(p2);
    pool.Free(p3);
    ASSERT_EQ(pool.FreeSize(), 1024);
    ASSERT_EQ(pool.AllocatedBlockCount(), 0);
    ASSERT_EQ(pool.FreeBlockCount(), 1);
}

TEST_F(Hf3fsMempoolTest, Alloc_ReturnNullptr_NoFreeBlock) {
    std::vector<uint8_t> buf(128);
    Hf3fsMempool pool(buf.data(), buf.size(), 0);
    ASSERT_TRUE(pool.Init());
    void *p1 = pool.Alloc(128);
    ASSERT_NE(p1, nullptr);
    ASSERT_EQ(pool.FreeSize(), 128 - 128);
    // 再申请应失败
    EXPECT_EQ(pool.Alloc(1), nullptr);
}

TEST_F(Hf3fsMempoolTest, Free_ReturnVoid_PtrNotAllocated) {
    std::vector<uint8_t> buf(256);
    Hf3fsMempool pool(buf.data(), buf.size(), 0);
    ASSERT_TRUE(pool.Init());
    uint8_t dummy;
    pool.Free(&dummy); // 非池内指针，函数内部会告警并返回
    SUCCEED();
}

TEST_F(Hf3fsMempoolTest, Free_ReturnVoid_MergePrevAndNext) {
    std::vector<uint8_t> buf(1024);
    Hf3fsMempool pool(buf.data(), buf.size(), 0);
    ASSERT_TRUE(pool.Init());

    void *p1 = pool.Alloc(128);
    ASSERT_NE(p1, nullptr);
    ASSERT_EQ(pool.FreeSize(), 1024 - 128);

    void *p2 = pool.Alloc(128);
    ASSERT_NE(p2, nullptr);
    ASSERT_EQ(pool.FreeSize(), 1024 - 128 - 128);

    pool.Free(p1);
    ASSERT_EQ(pool.FreeSize(), 1024 - 128);

    pool.Free(p2); // 应该与前一个空闲块合并
    ASSERT_EQ(pool.FreeSize(), 1024);

    // 再次分配 256 应成功（证明合并生效）
    void *p3 = pool.Alloc(256);
    ASSERT_NE(p3, nullptr);
    ASSERT_EQ(pool.FreeSize(), 1024 - 256);
}

TEST_F(Hf3fsMempoolTest, AllocatedSize_Return300_AfterTwoAlloc) {
    std::vector<uint8_t> buf(1024);
    Hf3fsMempool pool(buf.data(), buf.size(), 0);
    ASSERT_TRUE(pool.Init());
    auto *p1 = pool.Alloc(100);
    auto *p2 = pool.Alloc(200);
    ASSERT_NE(p1, nullptr);
    ASSERT_NE(p2, nullptr);
    EXPECT_EQ(pool.AllocatedSize(), 300u);
}

TEST_F(Hf3fsMempoolTest, FreeSize_ReturnDecrease_AfterAlloc) {
    std::vector<uint8_t> buf(1024);
    Hf3fsMempool pool(buf.data(), buf.size(), 0);
    ASSERT_TRUE(pool.Init());

    auto before = pool.FreeSize();
    EXPECT_EQ(before, 1024);

    auto *p1 = pool.Alloc(100);
    ASSERT_NE(p1, nullptr);
    auto after1 = pool.FreeSize();
    EXPECT_EQ(after1, before - 100);

    auto *p2 = pool.Alloc(200);
    ASSERT_NE(p2, nullptr);
    auto after2 = pool.FreeSize();
    EXPECT_EQ(after2, after1 - 200);

    pool.Free(p2);
    pool.Free(p1);
    auto after3 = pool.FreeSize();
    EXPECT_EQ(after3, before);
}

TEST_F(Hf3fsMempoolTest, AllocatedBlockCount_ReturnIncDec_AfterAllocAndFree) {
    std::vector<uint8_t> buf(1024);
    Hf3fsMempool pool(buf.data(), buf.size(), 0);
    ASSERT_TRUE(pool.Init());

    auto c0 = pool.AllocatedBlockCount();
    EXPECT_EQ(c0, 0);

    auto *p = pool.Alloc(64);
    ASSERT_NE(p, nullptr);
    auto c1 = pool.AllocatedBlockCount();
    EXPECT_EQ(c1, 1);

    pool.Free(p);
    auto c2 = pool.AllocatedBlockCount();
    EXPECT_EQ(c2, 0);
}

TEST_F(Hf3fsMempoolTest, PrintStatus_ReturnVoid_Smoke) {
    std::vector<uint8_t> buf(256);
    Hf3fsMempool pool(buf.data(), buf.size(), 0);
    ASSERT_TRUE(pool.Init());
    pool.PrintStatus();
    SUCCEED();
}

TEST_F(Hf3fsMempoolTest, FreeBlockCount) {
    std::vector<uint8_t> buf(1024);
    Hf3fsMempool pool(buf.data(), buf.size(), 0);
    ASSERT_TRUE(pool.Init());
    EXPECT_EQ(pool.FreeBlockCount(), 1);

    auto *p1 = pool.Alloc(100);
    ASSERT_NE(p1, nullptr);
    EXPECT_EQ(pool.FreeBlockCount(), 1);

    pool.Free(p1);
    EXPECT_EQ(pool.FreeBlockCount(), 1);
}