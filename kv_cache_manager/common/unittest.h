#pragma once

#include <assert.h>
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <execinfo.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <string.h>
#include <string>
#include <sys/mman.h>
#include <typeinfo>
#include <unistd.h>

#include "kv_cache_manager/common/env_util.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/timestamp_util.h"

using namespace ::testing;

class TESTBASE : public testing::Test {
public:
    TESTBASE() { init(); }

public:
    std::string GetWorkspacePath() { return workspace_path_; }
    std::string GetTestTempRootPath() { return test_temp_root_path_; }
    std::string GetPrivateTestDataPath() { return private_test_data_path_; }
    std::string GetPrivateTestRuntimeDataPath() { return private_test_runtime_data_path_; }
    std::string GetCurrentTestName() const {
        const ::testing::TestInfo *const test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        return std::string(test_info->test_case_name()) + "_" + test_info->name();
    }

protected:
    void init() {
        const char *tmp_path = getenv("TEST_TMPDIR");
        if (tmp_path == nullptr) {
            test_temp_root_path_ = "./";
        } else {
            test_temp_root_path_ = std::string(tmp_path);
        }
        std::string current_path;
        const char *src_path = getenv("TEST_SRCDIR");
        const char *workspace = getenv("TEST_WORKSPACE");

        if (tmp_path && workspace) {
            size_t pos = std::string(tmp_path).find(workspace);
            workspace_path_ = std::string(tmp_path).substr(0, pos + strlen(workspace));
        }
        if (src_path && workspace) {
            current_path = std::string(src_path) + "/" + std::string(workspace) + "/";
        } else {
            char cwd_path[PATH_MAX];
            auto unused = getcwd(cwd_path, PATH_MAX);
            (void)unused;
            current_path = cwd_path;
            current_path += "/";
        }
        const char *test_binary = getenv("TEST_BINARY");
        std::string test_binary_str;
        if (test_binary) {
            test_binary_str = std::string(test_binary);
        }
        // when run gdb, separate test_binary_str from currentPath
        if (test_binary_str.empty() || test_binary_str[0] == '/') {
            std::string test_binary_prefix = ".runfiles/kv_cache_manager/";
            size_t pos = current_path.find(test_binary_prefix);
            if (std::string::npos != pos) {
                test_binary_str = current_path.substr(pos + test_binary_prefix.size(),
                                                      current_path.size() - (pos + test_binary_prefix.size()));
                current_path = current_path.substr(0, pos + test_binary_prefix.size());
            }
        }
        size_t file_pos = test_binary_str.rfind('/') + 1;
        private_test_data_path_ = current_path + test_binary_str.substr(0, file_pos) + "testdata/";

        if (test_temp_root_path_ == "./") {
            char buf[PATH_MAX];
            if (getcwd(buf, PATH_MAX) != nullptr) {
                test_temp_root_path_ = buf;
            }
        }

        SetupTestRuntimeDataPath();

        std::cerr << "TEST_START_TIME: "
                  << kv_cache_manager::TimestampUtil::FormatTimestampUs(
                         kv_cache_manager::TimestampUtil::GetCurrentTimeUs())
                  << std::endl;
        // std::cerr << "WORKSPACE_PATH: " << (workspace_path_) << std::endl;
        // std::cerr << "TEST_TMPDIR: " << (tmp_path ? tmp_path : "") << std::endl;
        std::cerr << "TEST_TEMP_ROOT: " << test_temp_root_path_ << std::endl;
        std::cerr << "TEST_SRCDIR: " << (src_path ? src_path : "") << std::endl;
        std::cerr << "TEST_LOG_DIR: " << current_path << std::endl;
        // std::cerr << "TEST_WORKSPACE: " << (workspace ? workspace : "") << std::endl;
        std::cerr << "TEST_BINARY: " << (test_binary ? test_binary : "") << std::endl;
        std::cerr << "PRIVATE_TEST_DATA_PATH: " << private_test_data_path_ << std::endl;
        std::cerr << "PRIVATE_TEST_RUNTIME_DATA_PATH: " << private_test_runtime_data_path_ << std::endl;
    }

    void ReplaceSlashWithUnderscore(std::string &s) {
        for (auto pos = s.find("/"); pos != std::string::npos; pos = s.find("/")) {
            s.replace(pos, 1, "_");
        }
    }

    bool IsFile(const std::string &path) {
        struct stat buf;
        if (stat(path.c_str(), &buf) < 0) {
            return false;
        }
        if (S_ISREG(buf.st_mode)) {
            return true;
        }
        return false;
    }
    bool IsDir(const std::string &path) {
        struct stat buf;
        if (stat(path.c_str(), &buf) < 0) {
            return false;
        }
        if (S_ISDIR(buf.st_mode)) {
            return true;
        }
        return false;
    }
    bool Remove(const std::string &path) {
        if (IsFile(path)) {
            return ::remove(path.c_str()) == 0;
        }
        if (IsDir(path)) {
            DIR *p_dir;
            struct dirent *ent;
            std::string child_name;
            p_dir = opendir(path.c_str());
            if (p_dir == NULL) {
                return errno == ENOENT;
            }
            while ((ent = readdir(p_dir)) != NULL) {
                child_name = path + '/' + ent->d_name;
                if (ent->d_type & DT_DIR) {
                    if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) {
                        continue;
                    }
                    if (!Remove(child_name)) {
                        closedir(p_dir);
                        return false;
                    }
                } else if (::remove(child_name.c_str()) != 0) {
                    return false;
                }
            }

            closedir(p_dir);
            if (::remove(path.c_str()) != 0) {
                if (errno != ENOENT) {
                    return false;
                }
            }
            return true;
        }

        if (::remove(path.c_str()) != 0) {
            return false;
        }
        return true;
    }

    virtual void SetupTestRuntimeDataPath() {
        std::string temp_runtime_data = test_temp_root_path_ + "/test_runtime_data/";
        if (0 != mkdir(temp_runtime_data.c_str(), 0755) && EEXIST != errno) {
            ASSERT_FALSE(true) << "mkdir failed, " << temp_runtime_data << " err " << strerror(errno);
        }
        std::string folder_name = GetCurrentTestName();
        ReplaceSlashWithUnderscore(folder_name);
        private_test_runtime_data_path_ = temp_runtime_data + folder_name + "/";

        CleanTestRuntimeDataPath();
        if (0 != mkdir(private_test_runtime_data_path_.c_str(), 0755) && EEXIST != errno) {
            ASSERT_FALSE(true) << "mkdir failed, " << private_test_runtime_data_path_ << " err " << strerror(errno);
        }
    }

    virtual void CleanTestRuntimeDataPath() {
        if (!private_test_runtime_data_path_.empty() &&
            private_test_runtime_data_path_.rfind(test_temp_root_path_) != std::string::npos &&
            access(private_test_runtime_data_path_.c_str(), F_OK) == 0) {
            Remove(private_test_runtime_data_path_);
        }
    }

private:
    std::string test_temp_root_path_;
    std::string workspace_path_;
    // UT当前路径下的testdata目录
    std::string private_test_data_path_;
    // UT当前路径下的test_runtime_data目录
    std::string private_test_runtime_data_path_;
};

int main(int argc, char **argv) {
#ifndef NDEBUG
    kv_cache_manager::ScopedEnv env_log_to_console("KVCM_LOG_TO_CONSOLE", "1");
#endif
    kv_cache_manager::LoggerBroker::InitLogger("");
    if (kv_cache_manager::EnvUtil::GetEnv("KVCM_LOG_LEVEL", "").empty()) {
        kv_cache_manager::LoggerBroker::SetLogLevel(kv_cache_manager::Logger::LEVEL_DEBUG);
    };

    ::testing::InitGoogleMock(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}
