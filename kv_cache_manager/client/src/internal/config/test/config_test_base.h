#pragma once

#include <fstream>
#include <gtest/gtest.h>

#include "kv_cache_manager/common/unittest.h"

namespace kv_cache_manager {

class ConfigTestBase : public TESTBASE {
public:
    void SetUp() override { root_path_ = GetPrivateTestDataPath(); }

    void TearDown() override {}

protected:
    std::string getFileContent(const std::string &file_name) const {
        std::string path = root_path_ + file_name;
        std::ifstream infile(path);
        if (!infile.is_open()) {
            KVCM_LOG_ERROR("path [%s] not exist or not open", path.c_str());
            return "";
        }
        return std::string((std::istreambuf_iterator<char>(infile)), std::istreambuf_iterator<char>());
    }

    std::string root_path_;
};

} // namespace kv_cache_manager