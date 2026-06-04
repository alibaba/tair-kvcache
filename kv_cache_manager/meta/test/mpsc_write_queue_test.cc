#include <atomic>
#include <thread>
#include <unordered_set>
#include <vector>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/meta/mpsc_write_queue.h"

namespace kv_cache_manager {

class MpscWriteQueueTest : public TESTBASE {
public:
    void SetUp() override { queue_ = std::make_unique<MpscWriteQueue>(); }
    void TearDown() override { queue_.reset(); }

protected:
    WriteOp MakeWriteOp(WriteOpType type, const KeyTypeVec &keys) {
        WriteOp op;
        op.type = type;
        op.keys = keys;
        return op;
    }

    std::unique_ptr<MpscWriteQueue> queue_;
    int64_t taken_keys_ = 0;
};

TEST_F(MpscWriteQueueTest, TestEmptyPopBatch) {
    auto items = queue_->PopBatch(100, taken_keys_);
    ASSERT_TRUE(items.empty());
    ASSERT_EQ(0, taken_keys_);
    ASSERT_EQ(0, queue_->GetKeySize());
}

TEST_F(MpscWriteQueueTest, TestPushAndPopBatch) {
    queue_->Push(QueueItem{MakeWriteOp(WriteOpType::kPut, {1, 2, 3})});
    queue_->Push(QueueItem{MakeWriteOp(WriteOpType::kDelete, {4, 5})});
    ASSERT_EQ(5, queue_->GetKeySize());

    auto items = queue_->PopBatch(10, taken_keys_);
    ASSERT_EQ(2, items.size());
    ASSERT_EQ(5, taken_keys_);
    ASSERT_EQ(0, queue_->GetKeySize());

    // Verify FIFO order
    ASSERT_TRUE(std::holds_alternative<WriteOp>(items[0]));
    ASSERT_TRUE(std::holds_alternative<WriteOp>(items[1]));
    auto &op0 = std::get<WriteOp>(items[0]);
    auto &op1 = std::get<WriteOp>(items[1]);
    ASSERT_EQ(WriteOpType::kPut, op0.type);
    ASSERT_EQ(3, op0.keys.size());
    ASSERT_EQ(WriteOpType::kDelete, op1.type);
    ASSERT_EQ(2, op1.keys.size());
}

TEST_F(MpscWriteQueueTest, TestPopBatchLimited) {
    queue_->Push(QueueItem{MakeWriteOp(WriteOpType::kPut, {1})});
    queue_->Push(QueueItem{MakeWriteOp(WriteOpType::kPut, {2})});
    queue_->Push(QueueItem{MakeWriteOp(WriteOpType::kPut, {3})});
    ASSERT_EQ(3, queue_->GetKeySize());

    // Pop with limit 2 — remaining item stays in consumer-local buffer, still counted in GetKeySize
    auto items = queue_->PopBatch(2, taken_keys_);
    ASSERT_EQ(2, items.size());
    ASSERT_EQ(2, taken_keys_);
    ASSERT_EQ(1, queue_->GetKeySize());

    // Pop remaining
    auto remaining = queue_->PopBatch(10, taken_keys_);
    ASSERT_EQ(1, remaining.size());
    ASSERT_EQ(1, taken_keys_);
    ASSERT_EQ(0, queue_->GetKeySize());
}

TEST_F(MpscWriteQueueTest, TestRepeatedPopBatchClearsConsumerLeftover) {
    for (int i = 0; i < 6; ++i) {
        queue_->Push(QueueItem{MakeWriteOp(WriteOpType::kPut, {i})});
    }
    ASSERT_EQ(6, queue_->GetKeySize());

    auto first_items = queue_->PopBatch(2, taken_keys_);
    ASSERT_EQ(2, first_items.size());
    ASSERT_EQ(2, taken_keys_);
    ASSERT_EQ(4, queue_->GetKeySize());

    auto second_items = queue_->PopBatch(2, taken_keys_);
    ASSERT_EQ(2, second_items.size());
    ASSERT_EQ(2, taken_keys_);
    ASSERT_EQ(2, queue_->GetKeySize());

    auto remaining_items = queue_->PopBatch(2, taken_keys_);
    ASSERT_EQ(2, remaining_items.size());
    ASSERT_EQ(2, taken_keys_);
    ASSERT_EQ(0, queue_->GetKeySize());

    auto empty_items = queue_->PopBatch(2, taken_keys_);
    ASSERT_TRUE(empty_items.empty());
    ASSERT_EQ(0, taken_keys_);
}

TEST_F(MpscWriteQueueTest, TestPopBatchWaitTimeout) {
    auto start = std::chrono::steady_clock::now();
    auto items = queue_->PopBatchWait(10, 50000, taken_keys_); // 50ms timeout
    auto elapsed = std::chrono::steady_clock::now() - start;
    ASSERT_TRUE(items.empty());
    ASSERT_EQ(0, taken_keys_);
    ASSERT_GE(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(), 40);
}

TEST_F(MpscWriteQueueTest, TestPopBatchWaitWakeup) {
    std::atomic<bool> popped{false};

    std::thread consumer([&] {
        int64_t tk = 0;
        auto items = queue_->PopBatchWait(10, 5000000, tk); // 5s timeout - should be woken up much earlier
        ASSERT_EQ(1, items.size());
        ASSERT_EQ(1, tk);
        popped.store(true);
    });

    // Give consumer time to enter wait
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    queue_->Push(QueueItem{MakeWriteOp(WriteOpType::kPut, {42})});
    queue_->NotifyConsumer();

    consumer.join();
    ASSERT_TRUE(popped.load());
}

TEST_F(MpscWriteQueueTest, TestNotifyConsumer) {
    // NotifyConsumer should wake up waiting PopBatchWait even with empty queue
    std::atomic<bool> returned{false};

    std::thread consumer([&] {
        int64_t tk = 0;
        auto items = queue_->PopBatchWait(10, 5000000, tk); // 5s timeout
        returned.store(true);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    queue_->NotifyConsumer();

    consumer.join();
    ASSERT_TRUE(returned.load());
}

TEST_F(MpscWriteQueueTest, TestBarrierItem) {
    auto ctx = std::make_shared<BarrierContext>();
    ctx->remain.store(1, std::memory_order_release);

    SyncBarrierItem barrier;
    barrier.barrier_ctx = ctx;
    queue_->Push(QueueItem{std::move(barrier)});
    ASSERT_EQ(0, queue_->GetKeySize());

    auto items = queue_->PopBatch(10, taken_keys_);
    ASSERT_EQ(1, items.size());
    ASSERT_EQ(0, taken_keys_);
    ASSERT_TRUE(std::holds_alternative<SyncBarrierItem>(items[0]));
    auto &popped_barrier = std::get<SyncBarrierItem>(items[0]);
    ASSERT_EQ(ctx, popped_barrier.barrier_ctx);

    // Fence the barrier
    ctx->Fence();
    ASSERT_TRUE(ctx->Wait(std::chrono::milliseconds{100}));
}

TEST_F(MpscWriteQueueTest, TestMultiProducerSingleConsumer) {
    constexpr int kProducerCount = 8;
    constexpr int kOpsPerProducer = 1000;
    std::vector<std::thread> producers;

    for (int p = 0; p < kProducerCount; ++p) {
        producers.emplace_back([&, p] {
            for (int i = 0; i < kOpsPerProducer; ++i) {
                KeyType key = p * kOpsPerProducer + i;
                queue_->Push(QueueItem{MakeWriteOp(WriteOpType::kPut, {key})});
            }
        });
    }

    for (auto &t : producers) {
        t.join();
    }

    ASSERT_EQ(kProducerCount * kOpsPerProducer, queue_->GetKeySize());

    // Consume all
    std::unordered_set<KeyType> seen_keys;
    int total_items = 0;
    while (true) {
        int64_t tk = 0;
        auto items = queue_->PopBatch(512, tk);
        if (items.empty()) {
            break;
        }
        for (auto &item : items) {
            ASSERT_TRUE(std::holds_alternative<WriteOp>(item));
            auto &op = std::get<WriteOp>(item);
            ASSERT_EQ(1, op.keys.size());
            seen_keys.insert(op.keys[0]);
            ++total_items;
        }
    }
    ASSERT_EQ(kProducerCount * kOpsPerProducer, total_items);
    ASSERT_EQ(kProducerCount * kOpsPerProducer, (int)seen_keys.size());
}

TEST_F(MpscWriteQueueTest, TestMixedWriteOpsAndBarriers) {
    queue_->Push(QueueItem{MakeWriteOp(WriteOpType::kPut, {1})});
    queue_->Push(QueueItem{MakeWriteOp(WriteOpType::kUpsert, {2})});

    auto ctx = std::make_shared<BarrierContext>();
    ctx->remain.store(1);
    SyncBarrierItem barrier;
    barrier.barrier_ctx = ctx;
    queue_->Push(QueueItem{std::move(barrier)});

    queue_->Push(QueueItem{MakeWriteOp(WriteOpType::kDelete, {3})});
    ASSERT_EQ(3, queue_->GetKeySize());

    auto items = queue_->PopBatch(100, taken_keys_);
    ASSERT_EQ(4, items.size());
    ASSERT_EQ(3, taken_keys_);

    ASSERT_TRUE(std::holds_alternative<WriteOp>(items[0]));
    ASSERT_TRUE(std::holds_alternative<WriteOp>(items[1]));
    ASSERT_TRUE(std::holds_alternative<SyncBarrierItem>(items[2]));
    ASSERT_TRUE(std::holds_alternative<WriteOp>(items[3]));
}

// --- BarrierContext tests ---

TEST_F(MpscWriteQueueTest, TestBarrierContextMultiFence) {
    auto ctx = std::make_shared<BarrierContext>();
    ctx->remain.store(3, std::memory_order_release);

    std::thread t1([&] { ctx->Fence(); });
    std::thread t2([&] { ctx->Fence(); });
    std::thread t3([&] { ctx->Fence(); });

    t1.join();
    t2.join();
    t3.join();

    ASSERT_TRUE(ctx->Wait(std::chrono::milliseconds{100}));
    ASSERT_FALSE(ctx->failed.load());
}

TEST_F(MpscWriteQueueTest, TestBarrierContextTimeout) {
    auto ctx = std::make_shared<BarrierContext>();
    ctx->remain.store(2, std::memory_order_release);
    ctx->Fence(); // Only fence once, remain=1

    ASSERT_FALSE(ctx->Wait(std::chrono::milliseconds{50}));
}

TEST_F(MpscWriteQueueTest, TestBarrierContextSetFailed) {
    auto ctx = std::make_shared<BarrierContext>();
    ctx->remain.store(2, std::memory_order_release);

    std::thread t([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        ctx->SetFailed();
    });

    bool result = ctx->Wait(std::chrono::milliseconds{5000});
    ASSERT_FALSE(result);
    t.join();
}

// --- WaitForCapacity tests ---

TEST_F(MpscWriteQueueTest, TestWaitForCapacityImmediate) {
    ASSERT_TRUE(queue_->WaitForCapacity(10, 1, 100000));
}

TEST_F(MpscWriteQueueTest, TestWaitForCapacityConsidersIncomingKeys) {
    queue_->Push(QueueItem{MakeWriteOp(WriteOpType::kPut, {1, 2, 3, 4, 5, 6, 7, 8})});
    ASSERT_EQ(8, queue_->GetKeySize());

    ASSERT_TRUE(queue_->WaitForCapacity(10, 2, 100000));
    ASSERT_FALSE(queue_->WaitForCapacity(10, 3, 1000));
}

TEST_F(MpscWriteQueueTest, TestWaitForCapacityAllowsOversizedItemWhenEmpty) {
    ASSERT_TRUE(queue_->WaitForCapacity(3, 10, 100000));
}

TEST_F(MpscWriteQueueTest, TestWaitForCapacityTimeout) {
    for (int i = 0; i < 5; ++i) {
        queue_->Push(QueueItem{MakeWriteOp(WriteOpType::kPut, {i})});
    }
    ASSERT_EQ(5, queue_->GetKeySize());

    auto start = std::chrono::steady_clock::now();
    ASSERT_FALSE(queue_->WaitForCapacity(3, 1, 50000));
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - start)
                          .count();
    ASSERT_GE(elapsed_ms, 40);
}

TEST_F(MpscWriteQueueTest, TestWaitForCapacityWakeup) {
    for (int i = 0; i < 10; ++i) {
        queue_->Push(QueueItem{MakeWriteOp(WriteOpType::kPut, {i})});
    }
    ASSERT_EQ(10, queue_->GetKeySize());

    std::atomic<bool> got_capacity{false};
    std::thread producer([&] {
        got_capacity.store(queue_->WaitForCapacity(5, 1, 5000000));
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    ASSERT_FALSE(got_capacity.load());

    int64_t tk = 0;
    auto items = queue_->PopBatch(10, tk);
    ASSERT_EQ(10, items.size());

    producer.join();
    ASSERT_TRUE(got_capacity.load());
}

} // namespace kv_cache_manager
