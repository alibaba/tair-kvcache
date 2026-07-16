#include <gtest/gtest.h>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/service/util/manager_message_proto_util.h"

using namespace kv_cache_manager;

class ManagerMessageProtoUtilTest : public TESTBASE {};

TEST_F(ManagerMessageProtoUtilTest, CacheLocationChecksumRoundTripsThroughMetaProto) {
    CacheLocation loc;
    loc.set_type(DataStorageType::DATA_STORAGE_TYPE_NFS);
    loc.set_spec_size(1);
    std::vector<LocationSpec> specs;
    specs.emplace_back("tp0", "file://test_nfs/key?offset=0&size=1024");
    loc.set_location_specs(std::move(specs));
    loc.set_checksum(0x1234);

    proto::meta::CacheLocation proto_loc;
    ProtoConvert::CacheLocationToProto(loc, &proto_loc);
    EXPECT_EQ(proto_loc.checksum(), 0x1234);

    CacheLocation parsed;
    ProtoConvert::CacheLocationFromProto(&proto_loc, parsed);
    EXPECT_EQ(parsed.checksum(), 0x1234);
    EXPECT_EQ(parsed.type(), DataStorageType::DATA_STORAGE_TYPE_NFS);
    ASSERT_EQ(parsed.location_specs().size(), 1u);
    EXPECT_EQ(parsed.location_specs()[0].name(), "tp0");
}

TEST_F(ManagerMessageProtoUtilTest, CacheLocationChecksumRoundTripsThroughAdminProto) {
    CacheLocation loc;
    loc.set_type(DataStorageType::DATA_STORAGE_TYPE_NFS);
    loc.set_spec_size(1);
    std::vector<LocationSpec> specs;
    specs.emplace_back("tp1", "file://test_nfs/key?offset=1024&size=1024");
    loc.set_location_specs(std::move(specs));
    loc.set_checksum(-1);

    proto::admin::CacheLocation proto_loc;
    ProtoConvert::CacheLocationToProto(loc, &proto_loc);
    EXPECT_EQ(proto_loc.checksum(), -1);

    CacheLocation parsed;
    ProtoConvert::CacheLocationFromProto(&proto_loc, parsed);
    EXPECT_EQ(parsed.checksum(), -1);
    EXPECT_EQ(parsed.type(), DataStorageType::DATA_STORAGE_TYPE_NFS);
    ASSERT_EQ(parsed.location_specs().size(), 1u);
    EXPECT_EQ(parsed.location_specs()[0].name(), "tp1");
}
