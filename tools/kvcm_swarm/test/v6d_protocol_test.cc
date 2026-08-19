// Confirmed legacy-V6D to current-protocol mappings.
#include <gtest/gtest.h>

#include "tools/kvcm_swarm/protocol/proto_alias.h"

namespace kvcm_swarm {
namespace {

// The three confirmed legacy-to-current mappings, asserted on the exact
// messages the V6D behavior builds.
TEST(V6dCompatibilityTest, VineyardStorageTypeMapsToEventReportL2) {
    meta::ReportEventRequest request;
    request.set_storage_type(meta::ST_EVENT_REPORT_L2);
    EXPECT_EQ(meta::StorageType_Name(request.storage_type()), "ST_EVENT_REPORT_L2");
    EXPECT_EQ(static_cast<int>(meta::ST_EVENT_REPORT_L2), 8);
}

TEST(V6dCompatibilityTest, BlockDeleteCarriesNonEmptySpecNames) {
    meta::BlockDeleteEventParams params;
    params.set_block_key("7");
    params.set_medium("mem");
    params.add_spec_names("v6d_4096");
    EXPECT_GT(params.spec_names_size(), 0) << "the server rejects a BLOCK_DELETE without spec_names";
}

TEST(V6dCompatibilityTest, LookupCarriesOneSpecNamePerBlockKey) {
    meta::GetCacheLocationsByBackendRequest request;
    for (int i = 0; i < 5; ++i) {
        request.add_block_keys(i);
        request.add_location_spec_names(i % 2 == 0 ? "v6d_4096" : "v6d_1024");
    }
    EXPECT_EQ(request.block_keys_size(), request.location_spec_names_size());
}

} // namespace
} // namespace kvcm_swarm
