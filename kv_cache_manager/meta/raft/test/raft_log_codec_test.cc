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
    op.instance_id = "inst-A";
    op.key = 12345;
    op.locations.emplace("loc-a", MakeLocation("storage-a"));
    op.locations.emplace("loc-b", MakeLocation("storage-b"));
    op.properties.emplace("__uri__", "tair://x");
    op.properties.emplace("__ttl__", "60");

    auto buf = Encode(op);
    LogOp decoded = DecodeBuffer(buf);

    EXPECT_EQ(OpType::kPut, decoded.type);
    EXPECT_EQ("inst-A", decoded.instance_id);
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
    op.instance_id = "inst-U";
    op.key = 7;
    op.locations.emplace("loc-1", MakeLocation("s1"));

    auto buf = Encode(op);
    LogOp decoded = DecodeBuffer(buf);
    EXPECT_EQ(OpType::kUpsert, decoded.type);
    EXPECT_EQ("inst-U", decoded.instance_id);
    EXPECT_EQ(7, decoded.key);
    ASSERT_EQ(1u, decoded.locations.size());
    EXPECT_EQ("s1", decoded.locations.at("loc-1")->id());
}

TEST_F(RaftLogCodecTest, PutIfAbsentRoundTrip) {
    LogOp op;
    op.type = OpType::kPutIfAbsent;
    op.instance_id = "inst-P";
    op.key = 99;
    op.properties.emplace("k", "v");

    auto buf = Encode(op);
    LogOp decoded = DecodeBuffer(buf);
    EXPECT_EQ(OpType::kPutIfAbsent, decoded.type);
    EXPECT_EQ("inst-P", decoded.instance_id);
    EXPECT_EQ(99, decoded.key);
    ASSERT_EQ(1u, decoded.properties.size());
    EXPECT_EQ("v", decoded.properties.at("k"));
}

TEST_F(RaftLogCodecTest, DeleteRoundTrip) {
    LogOp op;
    op.type = OpType::kDelete;
    op.instance_id = "inst-D";
    op.key = -42;

    auto buf = Encode(op);
    LogOp decoded = DecodeBuffer(buf);
    EXPECT_EQ(OpType::kDelete, decoded.type);
    EXPECT_EQ("inst-D", decoded.instance_id);
    EXPECT_EQ(-42, decoded.key);
    EXPECT_TRUE(decoded.locations.empty());
}

TEST_F(RaftLogCodecTest, DeleteLocationsRoundTrip) {
    LogOp op;
    op.type = OpType::kDeleteLocations;
    op.instance_id = "inst-DL";
    op.key = 100;
    op.location_ids = {"loc-1", "loc-2", "loc-3"};

    auto buf = Encode(op);
    LogOp decoded = DecodeBuffer(buf);
    EXPECT_EQ(OpType::kDeleteLocations, decoded.type);
    EXPECT_EQ("inst-DL", decoded.instance_id);
    EXPECT_EQ(100, decoded.key);
    ASSERT_EQ(3u, decoded.location_ids.size());
    EXPECT_EQ("loc-1", decoded.location_ids[0]);
    EXPECT_EQ("loc-2", decoded.location_ids[1]);
    EXPECT_EQ("loc-3", decoded.location_ids[2]);
}

TEST_F(RaftLogCodecTest, PutMetaDataRoundTrip) {
    LogOp op;
    op.type = OpType::kPutMetaData;
    op.instance_id = "inst-M";
    op.meta_fields.emplace("__key_count__", "1024");
    op.meta_fields.emplace("custom", "value");

    auto buf = Encode(op);
    LogOp decoded = DecodeBuffer(buf);
    EXPECT_EQ(OpType::kPutMetaData, decoded.type);
    EXPECT_EQ("inst-M", decoded.instance_id);
    ASSERT_EQ(2u, decoded.meta_fields.size());
    EXPECT_EQ("1024", decoded.meta_fields.at("__key_count__"));
    EXPECT_EQ("value", decoded.meta_fields.at("custom"));
}

TEST_F(RaftLogCodecTest, EmptyMapsRoundTrip) {
    LogOp op;
    op.type = OpType::kPut;
    op.instance_id = "inst-E";
    op.key = 1;
    // locations and properties intentionally empty

    auto buf = Encode(op);
    LogOp decoded = DecodeBuffer(buf);
    EXPECT_EQ(OpType::kPut, decoded.type);
    EXPECT_EQ("inst-E", decoded.instance_id);
    EXPECT_EQ(1, decoded.key);
    EXPECT_TRUE(decoded.locations.empty());
    EXPECT_TRUE(decoded.properties.empty());
}

TEST_F(RaftLogCodecTest, EmptyInstanceIdRoundTrip) {
    LogOp op;
    op.type = OpType::kDelete;
    op.key = 7;

    auto buf = Encode(op);
    LogOp decoded = DecodeBuffer(buf);
    EXPECT_EQ(OpType::kDelete, decoded.type);
    EXPECT_TRUE(decoded.instance_id.empty());
    EXPECT_EQ(7, decoded.key);
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
    // Build version + op + empty instance_id, then truncate after op so the
    // unknown-op branch exercises the explicit default in the switch (the
    // bs.get_str() for instance_id is allowed because we wrote an empty one).
    LogOp probe;
    probe.type = OpType::kPut;
    probe.instance_id = "";
    auto sample = Encode(probe);
    nuraft::buffer_serializer bs_in(*sample);
    (void)bs_in.get_u8();
    (void)bs_in.get_u8();
    (void)bs_in.get_str();
    auto buf = nuraft::buffer::alloc(64);
    nuraft::buffer_serializer bs(buf);
    bs.put_u8(1);
    bs.put_u8(0xEE);
    bs.put_str(std::string());
    LogOp out;
    EXPECT_EQ(EC_CORRUPTION, Decode(*buf, out));
}

TEST_F(RaftLogCodecTest, RegistrySaveRoundTrip) {
    LogOp op;
    op.type = OpType::kRegistrySave;
    op.registry_key = "instance_group";
    op.registry_fields = {{"group-a", "{\"name\":\"a\"}"}, {"group-b", "{\"name\":\"b\"}"}};

    auto buf = Encode(op);
    LogOp decoded = DecodeBuffer(buf);
    EXPECT_EQ(OpType::kRegistrySave, decoded.type);
    EXPECT_EQ("instance_group", decoded.registry_key);
    ASSERT_EQ(2u, decoded.registry_fields.size());
    EXPECT_EQ("{\"name\":\"a\"}", decoded.registry_fields.at("group-a"));
    EXPECT_EQ("{\"name\":\"b\"}", decoded.registry_fields.at("group-b"));
}

TEST_F(RaftLogCodecTest, RegistryDeleteRoundTrip) {
    LogOp op;
    op.type = OpType::kRegistryDelete;
    op.registry_key = "storage";

    auto buf = Encode(op);
    LogOp decoded = DecodeBuffer(buf);
    EXPECT_EQ(OpType::kRegistryDelete, decoded.type);
    EXPECT_EQ("storage", decoded.registry_key);
    EXPECT_TRUE(decoded.registry_fields.empty());
}

TEST_F(RaftLogCodecTest, RegistrySaveEmptyFieldsRoundTrip) {
    LogOp op;
    op.type = OpType::kRegistrySave;
    op.registry_key = "empty";

    auto buf = Encode(op);
    LogOp decoded = DecodeBuffer(buf);
    EXPECT_EQ(OpType::kRegistrySave, decoded.type);
    EXPECT_EQ("empty", decoded.registry_key);
    EXPECT_TRUE(decoded.registry_fields.empty());
}

} // namespace raft_meta
} // namespace kv_cache_manager
