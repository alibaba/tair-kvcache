#include <gtest/gtest.h>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/service/util/manager_message_proto_util.h"

using namespace kv_cache_manager;

class ManagerMessageProtoUtilTest : public TESTBASE {};

TEST_F(ManagerMessageProtoUtilTest, EmbeddedNulChecksumSurvivesProtoAndQueryView) {
    CacheLocation loc;
    std::vector<LocationSpec> specs;
    specs.emplace_back("tp0", "file://x");
    const std::string checksum("crc\0tail", 8);
    specs.back().set_checksum(checksum);
    loc.set_location_specs(std::move(specs));

    proto::meta::LocationSpec meta_spec;
    ProtoConvert::LocationSpecToProto(loc.location_specs()[0], &meta_spec);
    EXPECT_EQ(checksum, meta_spec.checksum());
    LocationSpec parsed;
    ProtoConvert::LocationSpecFromProto(&meta_spec, parsed);
    EXPECT_TRUE(parsed.has_checksum());
    EXPECT_EQ(checksum, parsed.checksum());

    proto::admin::LocationSpec admin_spec;
    ProtoConvert::LocationSpecToProto(parsed, &admin_spec);
    EXPECT_EQ(checksum, admin_spec.checksum());
    ProtoConvert::LocationSpecFromProto(&admin_spec, parsed);
    EXPECT_EQ(checksum, parsed.checksum());

    CacheLocationView view(loc);
    proto::meta::CacheLocation meta_loc;
    ProtoConvert::CacheLocationViewToProto(view, &meta_loc, true);
    ASSERT_EQ(1, meta_loc.location_specs_size());
    EXPECT_TRUE(meta_loc.location_specs(0).checksum_present());
    EXPECT_EQ(checksum, meta_loc.location_specs(0).checksum());

    proto::admin::CacheLocation admin_loc;
    ProtoConvert::CacheLocationViewToProto(view, &admin_loc, true);
    ASSERT_EQ(1, admin_loc.location_specs_size());
    EXPECT_TRUE(admin_loc.location_specs(0).checksum_present());
    EXPECT_EQ(checksum, admin_loc.location_specs(0).checksum());
}

TEST_F(ManagerMessageProtoUtilTest, LocationSpecChecksumsRoundTripThroughMetaProto) {
    CacheLocation loc;
    loc.set_type(DataStorageType::DATA_STORAGE_TYPE_NFS);
    loc.set_spec_size(1);
    std::vector<LocationSpec> specs;
    specs.emplace_back("tp0", "file://test_nfs/key?offset=0&size=1024");
    specs.back().set_checksum("crc64:00001234");
    loc.set_location_specs(std::move(specs));

    proto::meta::CacheLocation proto_loc;
    ProtoConvert::CacheLocationToProto(loc, &proto_loc);
    ASSERT_EQ(proto_loc.location_specs_size(), 1);
    EXPECT_EQ(proto_loc.location_specs(0).checksum(), "crc64:00001234");
    EXPECT_TRUE(proto_loc.location_specs(0).checksum_present());

    CacheLocation parsed;
    ProtoConvert::CacheLocationFromProto(&proto_loc, parsed);
    EXPECT_EQ(parsed.type(), DataStorageType::DATA_STORAGE_TYPE_NFS);
    ASSERT_EQ(parsed.location_specs().size(), 1u);
    EXPECT_EQ(parsed.location_specs()[0].name(), "tp0");
    EXPECT_EQ(parsed.location_specs()[0].checksum(), "crc64:00001234");
    EXPECT_TRUE(parsed.location_specs()[0].has_checksum());
}

TEST_F(ManagerMessageProtoUtilTest, LocationSpecChecksumsRoundTripThroughAdminProto) {
    CacheLocation loc;
    loc.set_type(DataStorageType::DATA_STORAGE_TYPE_NFS);
    loc.set_spec_size(1);
    std::vector<LocationSpec> specs;
    specs.emplace_back("tp1", "file://test_nfs/key?offset=1024&size=1024");
    specs.back().set_checksum("base64:AbCd+/==");
    loc.set_location_specs(std::move(specs));

    proto::admin::CacheLocation proto_loc;
    ProtoConvert::CacheLocationToProto(loc, &proto_loc);
    ASSERT_EQ(proto_loc.location_specs_size(), 1);
    EXPECT_EQ(proto_loc.location_specs(0).checksum(), "base64:AbCd+/==");
    EXPECT_TRUE(proto_loc.location_specs(0).checksum_present());

    CacheLocation parsed;
    ProtoConvert::CacheLocationFromProto(&proto_loc, parsed);
    EXPECT_EQ(parsed.type(), DataStorageType::DATA_STORAGE_TYPE_NFS);
    ASSERT_EQ(parsed.location_specs().size(), 1u);
    EXPECT_EQ(parsed.location_specs()[0].name(), "tp1");
    EXPECT_EQ(parsed.location_specs()[0].checksum(), "base64:AbCd+/==");
    EXPECT_TRUE(parsed.location_specs()[0].has_checksum());
}

TEST_F(ManagerMessageProtoUtilTest, EmptyChecksumPresenceRoundTripsThroughMetaProto) {
    CacheLocation loc;
    std::vector<LocationSpec> specs;
    specs.emplace_back("tp0", "file://test");
    specs.back().set_checksum("");
    loc.set_location_specs(std::move(specs));

    proto::meta::CacheLocation proto_loc;
    ProtoConvert::CacheLocationToProto(loc, &proto_loc);
    ASSERT_EQ(proto_loc.location_specs_size(), 1);
    EXPECT_EQ(proto_loc.location_specs(0).checksum(), "");
    EXPECT_TRUE(proto_loc.location_specs(0).checksum_present());

    CacheLocation parsed;
    ProtoConvert::CacheLocationFromProto(&proto_loc, parsed);
    ASSERT_EQ(parsed.location_specs().size(), 1u);
    EXPECT_TRUE(parsed.location_specs()[0].has_checksum());
    EXPECT_EQ(parsed.location_specs()[0].checksum(), "");
}

TEST_F(ManagerMessageProtoUtilTest, LegacyProtoWithoutChecksumKeepsPresenceFalse) {
    proto::meta::CacheLocation proto_loc;
    proto_loc.add_location_specs()->set_name("tp0");
    CacheLocation parsed;

    ProtoConvert::CacheLocationFromProto(&proto_loc, parsed);

    ASSERT_EQ(parsed.location_specs().size(), 1u);
    EXPECT_FALSE(parsed.location_specs()[0].has_checksum());
    EXPECT_EQ(parsed.location_specs()[0].checksum(), "");
}

TEST_F(ManagerMessageProtoUtilTest, CacheLocationViewOnlySerializesChecksumWhenRequested) {
    CacheLocation location;
    std::vector<LocationSpec> specs;
    specs.emplace_back("tp0", "file://test");
    specs.back().set_checksum("-42");
    location.set_location_specs(std::move(specs));
    CacheLocationView view(location);

    proto::meta::CacheLocation hidden;
    ProtoConvert::CacheLocationViewToProto(view, &hidden, false);
    ASSERT_EQ(hidden.location_specs_size(), 1);
    EXPECT_FALSE(hidden.location_specs(0).checksum_present());
    EXPECT_EQ("", hidden.location_specs(0).checksum());

    proto::meta::CacheLocation included;
    ProtoConvert::CacheLocationViewToProto(view, &included, true);
    ASSERT_EQ(included.location_specs_size(), 1);
    EXPECT_TRUE(included.location_specs(0).checksum_present());
    EXPECT_EQ("-42", included.location_specs(0).checksum());
}

TEST_F(ManagerMessageProtoUtilTest, ReusedProtoClearsChecksumWhenSourceHasNoValue) {
    LocationSpec source("tp0", "file://test");
    source.set_checksum("-42");
    proto::meta::LocationSpec proto_spec;
    ProtoConvert::LocationSpecToProto(source, &proto_spec);
    ASSERT_TRUE(proto_spec.checksum_present());

    source.clear_checksum();
    ProtoConvert::LocationSpecToProto(source, &proto_spec);
    EXPECT_FALSE(proto_spec.checksum_present());
    EXPECT_EQ("", proto_spec.checksum());

    LocationSpec restored;
    ProtoConvert::LocationSpecFromProto(&proto_spec, restored);
    EXPECT_FALSE(restored.has_checksum());
}

TEST_F(ManagerMessageProtoUtilTest, NonEmptyChecksumImpliesPresenceAndAbsentProtoClearsReusedObject) {
    proto::meta::LocationSpec proto_spec;
    proto_spec.set_checksum("0000");
    LocationSpec restored;
    ProtoConvert::LocationSpecFromProto(&proto_spec, restored);
    EXPECT_TRUE(restored.has_checksum());
    EXPECT_EQ("0000", restored.checksum());

    proto_spec.clear_checksum();
    ProtoConvert::LocationSpecFromProto(&proto_spec, restored);
    EXPECT_FALSE(restored.has_checksum());
    EXPECT_EQ("", restored.checksum());
}
