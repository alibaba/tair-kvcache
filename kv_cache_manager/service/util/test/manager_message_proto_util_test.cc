#include <gtest/gtest.h>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/service/util/manager_message_proto_util.h"

using namespace kv_cache_manager;

class ManagerMessageProtoUtilTest : public TESTBASE {};

TEST_F(ManagerMessageProtoUtilTest, LocationSpecChecksumsRoundTripThroughMetaProto) {
    CacheLocation loc;
    loc.set_type(DataStorageType::DATA_STORAGE_TYPE_NFS);
    loc.set_spec_size(1);
    std::vector<LocationSpec> specs;
    specs.emplace_back("tp0", "file://test_nfs/key?offset=0&size=1024");
    specs.back().set_checksum(0x1234);
    loc.set_location_specs(std::move(specs));

    proto::meta::CacheLocation proto_loc;
    ProtoConvert::CacheLocationToProto(loc, &proto_loc);
    ASSERT_EQ(proto_loc.location_specs_size(), 1);
    EXPECT_EQ(proto_loc.location_specs(0).checksum(), 0x1234);
    EXPECT_TRUE(proto_loc.location_specs(0).checksum_present());

    CacheLocation parsed;
    ProtoConvert::CacheLocationFromProto(&proto_loc, parsed);
    EXPECT_EQ(parsed.type(), DataStorageType::DATA_STORAGE_TYPE_NFS);
    ASSERT_EQ(parsed.location_specs().size(), 1u);
    EXPECT_EQ(parsed.location_specs()[0].name(), "tp0");
    EXPECT_EQ(parsed.location_specs()[0].checksum(), 0x1234);
    EXPECT_TRUE(parsed.location_specs()[0].has_checksum());
}

TEST_F(ManagerMessageProtoUtilTest, LocationSpecChecksumsRoundTripThroughAdminProto) {
    CacheLocation loc;
    loc.set_type(DataStorageType::DATA_STORAGE_TYPE_NFS);
    loc.set_spec_size(1);
    std::vector<LocationSpec> specs;
    specs.emplace_back("tp1", "file://test_nfs/key?offset=1024&size=1024");
    specs.back().set_checksum(-1);
    loc.set_location_specs(std::move(specs));

    proto::admin::CacheLocation proto_loc;
    ProtoConvert::CacheLocationToProto(loc, &proto_loc);
    ASSERT_EQ(proto_loc.location_specs_size(), 1);
    EXPECT_EQ(proto_loc.location_specs(0).checksum(), -1);
    EXPECT_TRUE(proto_loc.location_specs(0).checksum_present());

    CacheLocation parsed;
    ProtoConvert::CacheLocationFromProto(&proto_loc, parsed);
    EXPECT_EQ(parsed.type(), DataStorageType::DATA_STORAGE_TYPE_NFS);
    ASSERT_EQ(parsed.location_specs().size(), 1u);
    EXPECT_EQ(parsed.location_specs()[0].name(), "tp1");
    EXPECT_EQ(parsed.location_specs()[0].checksum(), -1);
    EXPECT_TRUE(parsed.location_specs()[0].has_checksum());
}

TEST_F(ManagerMessageProtoUtilTest, ZeroChecksumPresenceRoundTripsThroughMetaProto) {
    CacheLocation loc;
    std::vector<LocationSpec> specs;
    specs.emplace_back("tp0", "file://test");
    specs.back().set_checksum(0);
    loc.set_location_specs(std::move(specs));

    proto::meta::CacheLocation proto_loc;
    ProtoConvert::CacheLocationToProto(loc, &proto_loc);
    ASSERT_EQ(proto_loc.location_specs_size(), 1);
    EXPECT_EQ(proto_loc.location_specs(0).checksum(), 0);
    EXPECT_TRUE(proto_loc.location_specs(0).checksum_present());

    CacheLocation parsed;
    ProtoConvert::CacheLocationFromProto(&proto_loc, parsed);
    ASSERT_EQ(parsed.location_specs().size(), 1u);
    EXPECT_TRUE(parsed.location_specs()[0].has_checksum());
    EXPECT_EQ(parsed.location_specs()[0].checksum(), 0);
}

TEST_F(ManagerMessageProtoUtilTest, LegacyProtoWithoutChecksumKeepsPresenceFalse) {
    proto::meta::CacheLocation proto_loc;
    proto_loc.add_location_specs()->set_name("tp0");
    CacheLocation parsed;

    ProtoConvert::CacheLocationFromProto(&proto_loc, parsed);

    ASSERT_EQ(parsed.location_specs().size(), 1u);
    EXPECT_FALSE(parsed.location_specs()[0].has_checksum());
    EXPECT_EQ(parsed.location_specs()[0].checksum(), 0);
}

TEST_F(ManagerMessageProtoUtilTest, CacheLocationViewOnlySerializesChecksumWhenRequested) {
    CacheLocation location;
    std::vector<LocationSpec> specs;
    specs.emplace_back("tp0", "file://test");
    specs.back().set_checksum(-42);
    location.set_location_specs(std::move(specs));
    CacheLocationView view(location);

    proto::meta::CacheLocation hidden;
    ProtoConvert::CacheLocationViewToProto(view, &hidden, false);
    ASSERT_EQ(hidden.location_specs_size(), 1);
    EXPECT_FALSE(hidden.location_specs(0).checksum_present());
    EXPECT_EQ(0, hidden.location_specs(0).checksum());

    proto::meta::CacheLocation included;
    ProtoConvert::CacheLocationViewToProto(view, &included, true);
    ASSERT_EQ(included.location_specs_size(), 1);
    EXPECT_TRUE(included.location_specs(0).checksum_present());
    EXPECT_EQ(-42, included.location_specs(0).checksum());
}

TEST_F(ManagerMessageProtoUtilTest, StorageIntegrityRoundTripsThroughAdminProto) {
    auto nfs_spec = std::make_shared<NfsStorageSpec>();
    nfs_spec->set_root_path("/mnt/nfs");
    nfs_spec->set_key_count_per_file(8);
    StorageConfig config(DataStorageType::DATA_STORAGE_TYPE_NFS, "nfs_with_checksum", nfs_spec);
    config.mutable_integrity().set_enable_meta_checksum(true);
    config.mutable_integrity().set_algo(ChecksumAlgo::CA_CRC32_XOR_INT64);

    proto::admin::StorageConfig proto_config;
    ProtoConvert::StorageConfigToProto(config, &proto_config);
    ASSERT_TRUE(proto_config.has_integrity());
    EXPECT_TRUE(proto_config.integrity().enable_meta_checksum());
    EXPECT_EQ(proto::admin::CA_CRC32_XOR_INT64, proto_config.integrity().algo());

    StorageConfig restored;
    ProtoConvert::StorageFromProto(&proto_config, restored);
    EXPECT_TRUE(restored.integrity().enable_meta_checksum());
    EXPECT_EQ(ChecksumAlgo::CA_CRC32_XOR_INT64, restored.integrity().algo());
}

TEST_F(ManagerMessageProtoUtilTest, DefaultStorageIntegrityIsOmittedFromAdminProto) {
    auto nfs_spec = std::make_shared<NfsStorageSpec>();
    nfs_spec->set_root_path("/mnt/nfs");
    StorageConfig config(DataStorageType::DATA_STORAGE_TYPE_NFS, "legacy_nfs", nfs_spec);

    proto::admin::StorageConfig proto_config;
    ProtoConvert::StorageConfigToProto(config, &proto_config);
    EXPECT_FALSE(proto_config.has_integrity());

    StorageConfig restored;
    restored.mutable_integrity().set_enable_meta_checksum(true);
    ProtoConvert::StorageFromProto(&proto_config, restored);
    EXPECT_FALSE(restored.integrity().enable_meta_checksum());
    EXPECT_EQ(ChecksumAlgo::CA_CRC32_XOR_INT64, restored.integrity().algo());
}

TEST_F(ManagerMessageProtoUtilTest, UnknownAdminChecksumAlgorithmRemainsInvalidWhenDisabled) {
    proto::admin::StorageConfig proto_config;
    auto *integrity = proto_config.mutable_integrity();
    integrity->set_enable_meta_checksum(false);
    integrity->set_algo(static_cast<proto::admin::ChecksumAlgo>(999));

    StorageConfig restored;
    ProtoConvert::StorageFromProto(&proto_config, restored);

    std::string invalid_fields;
    EXPECT_FALSE(restored.integrity().ValidateRequiredFields(invalid_fields));
    EXPECT_NE(std::string::npos, invalid_fields.find("unsupported checksum algorithm"));
}
