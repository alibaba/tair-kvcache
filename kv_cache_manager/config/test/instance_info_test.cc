#include <algorithm>
#include <random>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/instance_info.h"

namespace kv_cache_manager {

namespace {

bool IsSpecNameInSpecGroup(std::string_view spec_name,
                           std::string_view group_name,
                           const std::vector<kv_cache_manager::LocationSpecGroup> &location_spec_groups) {
    // we have sorted location_spec_groups before
    auto it_group = std::lower_bound(location_spec_groups.begin(),
                                     location_spec_groups.end(),
                                     group_name,
                                     [](const auto &location_spec_group, std::string_view group_name) {
                                         return location_spec_group.name() < group_name;
                                     });
    if (it_group == location_spec_groups.end() || it_group->name() != group_name) {
        return false;
    }
    const auto &group = *it_group;
    auto it_spec_name = std::lower_bound(
        group.spec_names().begin(),
        group.spec_names().end(),
        spec_name,
        [](const std::string &src_spec_name, std::string_view dst_spec_name) { return src_spec_name < dst_spec_name; });
    if (it_spec_name == group.spec_names().end() || *it_spec_name != spec_name) {
        return false;
    }
    return true;
}

} // namespace

class InstanceInfoTest : public TESTBASE {
public:
    void SetUp() override {}
    void TearDown() override {}

private:
    std::vector<std::string> getRandomOrderLocationSpecGroupName(size_t group_id, size_t count) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::vector<size_t> ids;
        for (size_t i = 0; i < count; i++) {
            ids.push_back(i);
        }
        std::shuffle(ids.begin(), ids.end(), g);
        std::vector<std::string> res;
        for (size_t id : ids) {
            res.push_back("fake_spec_" + std::to_string(group_id) + std::to_string(id));
        }
        return res;
    }

    std::vector<LocationSpecGroup> getRandomOrderLocationGroups(size_t group_count, size_t spec_count) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::vector<size_t> ids;
        for (size_t i = 0; i < group_count; i++) {
            ids.push_back(i);
        }
        std::shuffle(ids.begin(), ids.end(), g);
        std::vector<LocationSpecGroup> res;

        for (size_t id : ids) {
            if (id % 2 == 0) {
                res.push_back({});
                res.back().set_name("fake_group_" + std::to_string(id));
                res.back().set_spec_names(getRandomOrderLocationSpecGroupName(id, spec_count));
            } else {
                res.emplace_back("fake_group_" + std::to_string(id),
                                 getRandomOrderLocationSpecGroupName(id, spec_count));
            }
        }
        return res;
    }
};

TEST_F(InstanceInfoTest, TestLocationSpecGroupSort) {
    for (size_t test_count = 0; test_count < 100; test_count++) {
        InstanceInfo instance_info("", "", "", 0, {}, {}, getRandomOrderLocationGroups(9, 9));
        const auto &location_spec_groups = instance_info.location_spec_groups();
        ASSERT_EQ(9, location_spec_groups.size());
        for (size_t i = 0; i < 9; i++) {
            std::string group_name = "fake_group_" + std::to_string(i);
            ASSERT_EQ(group_name, location_spec_groups[i].name());
            const auto &spec_names = location_spec_groups[i].spec_names();
            ASSERT_EQ(9, spec_names.size());
            for (size_t j = 0; j < 9; j++) {
                std::string spec_name = "fake_spec_" + std::to_string(i) + std::to_string(j);
                ASSERT_EQ(spec_name, spec_names[j]);
                ASSERT_TRUE(IsSpecNameInSpecGroup(spec_name, group_name, location_spec_groups));
            }
        }
    }
}

} // namespace kv_cache_manager