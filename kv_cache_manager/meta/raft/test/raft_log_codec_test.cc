#include "kv_cache_manager/meta/raft/raft_log_codec.h"

#include <libnuraft/buffer.hxx>
#include <libnuraft/buffer_serializer.hxx>

#include <memory>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/meta/cache_location.h"

namespace kv_cache_manager {
namespace raft_meta {

class RaftLogCodecTest : public TESTBASE {};

namespace {

// CacheLocation has many fields; we only need a couple set so JSON round-trip
// is exercised on the "non-empty payload" path.
CacheLocationConstPtr MakeLocation(const std::string &id) {
    auto loc = std::make_shared<CacheLocation>();
    loc->set_id(id);
    loc->set_spec_size(4096);
    return std::const_pointer_cast<const CacheLocation>(loc);
}

LogOp DecodeBuffer(nuraft::ptr<nuraft::buffer> buf) {
    LogOp out;
    EXPECT_EQ(EC_OK, Decode(*buf, out));
    return out;
}

} // namespace

TEST_F(RaftLogCodecTest, PutRoundTrip) {
    LogOp op;
    op.type = OpType::kPut;
    op.key = 12345;
    op.locations.emplace("loc-a", MakeLocation("storage-a"));
    op.locations.emplace("loc-b", MakeLocation("storage-b"));
    op.properties.emplace("__uri__", "tair://x");
    op.properties.emplace("__ttl__", "60");

    auto buf = Encode(op);
    LogOp decoded = DecodeBuffer(buf);

    EXPECT_EQ(OpType::kPut, decoded.type);
    EXPECT_EQ(op.key, decoded.key);
    ASSERT_EQ(2u, decoded.locations.size());
    ASSERT_EQ(2u, decoded.properties.size());
    EXPECT_EQ("storage-a", decoded.locations.at("loc-a")->id());
    EXPECT_EQ("storage-b", decoded.locations.at("loc-b")->id());
    EXPECT_EQ("tair://x", decoded.properties.at("__uri__"));
    EXPECT_EQ("60", decoded.properties.at("__ttl__"));
}

TEST_F(RaftLogCodecTest, UpsertRoundTrip) {
    LogOp op;
    op.type = OpType::kUpsert;
    op.key = 7;
    op.locations.emplace("loc-1", MakeLocation("s1"));

    auto buf = Encode(op);
    LogOp decoded = DecodeBuffer(buf);
    EXPECT_EQ(OpType::kUpsert, decoded.type);
    EXPECT_EQ(7, decoded.key);
    ASSERT_EQ(1u, decoded.locations.size());
    EXPECT_EQ("s1", decoded.locations.at("loc-1")->id());
}

TEST_F(RaftLogCodecTest, PutIfAbsentRoundTrip) {
    LogOp op;
    op.type = OpType::kPutIfAbsent;
    op.key = 99;
    op.properties.emplace("k", "v");

    auto buf = Encode(op);
    LogOp decoded = DecodeBuffer(buf);
    EXPECT_EQ(OpType::kPutIfAbsent, decoded.type);
    EXPECT_EQ(99, decoded.key);
    ASSERT_EQ(1u, decoded.properties.size());
    EXPECT_EQ("v", decoded.properties.at("k"));
}

TEST_F(RaftLogCodecTest, DeleteRoundTrip) {
    LogOp op;
    op.type = OpType::kDelete;
    op.key = -42;

    auto buf = Encode(op);
    LogOp decoded = DecodeBuffer(buf);
    EXPECT_EQ(OpType::kDelete, decoded.type);
    EXPECT_EQ(-42, decoded.key);
    EXPECT_TRUE(decoded.locations.empty());
}

TEST_F(RaftLogCodecTest, DeleteLocationsRoundTrip) {
    LogOp op;
    op.type = OpType::kDeleteLocations;
    op.key = 100;
    op.location_ids = {"loc-1", "loc-2", "loc-3"};

    auto buf = Encode(op);
    LogOp decoded = DecodeBuffer(buf);
    EXPECT_EQ(OpType::kDeleteLocations, decoded.type);
    EXPECT_EQ(100, decoded.key);
    ASSERT_EQ(3u, decoded.location_ids.size());
    EXPECT_EQ("loc-1", decoded.location_ids[0]);
    EXPECT_EQ("loc-2", decoded.location_ids[1]);
    EXPECT_EQ("loc-3", decoded.location_ids[2]);
}

TEST_F(RaftLogCodecTest, PutMetaDataRoundTrip) {
    LogOp op;
    op.type = OpType::kPutMetaData;
    op.meta_fields.emplace("__key_count__", "1024");
    op.meta_fields.emplace("custom", "value");

    auto buf = Encode(op);
    LogOp decoded = DecodeBuffer(buf);
    EXPECT_EQ(OpType::kPutMetaData, decoded.type);
    ASSERT_EQ(2u, decoded.meta_fields.size());
    EXPECT_EQ("1024", decoded.meta_fields.at("__key_count__"));
    EXPECT_EQ("value", decoded.meta_fields.at("custom"));
}

TEST_F(RaftLogCodecTest, EmptyMapsRoundTrip) {
    LogOp op;
    op.type = OpType::kPut;
    op.key = 1;
    // locations and properties intentionally empty

    auto buf = Encode(op);
    LogOp decoded = DecodeBuffer(buf);
    EXPECT_EQ(OpType::kPut, decoded.type);
    EXPECT_EQ(1, decoded.key);
    EXPECT_TRUE(decoded.locations.empty());
    EXPECT_TRUE(decoded.properties.empty());
}

TEST_F(RaftLogCodecTest, CorruptUnknownVersion) {
    auto buf = nuraft::buffer::alloc(4);
    nuraft::buffer_serializer bs(buf);
    bs.put_u8(99); // bogus version
    bs.put_u8(1);
    LogOp out;
    EXPECT_EQ(EC_CORRUPTION, Decode(*buf, out));
}

TEST_F(RaftLogCodecTest, CorruptUnknownOpType) {
    auto buf = nuraft::buffer::alloc(4);
    nuraft::buffer_serializer bs(buf);
    bs.put_u8(1);
    bs.put_u8(0xEE); // unknown op type
    LogOp out;
    EXPECT_EQ(EC_CORRUPTION, Decode(*buf, out));
}

} // namespace raft_meta
} // namespace kv_cache_manager
