#pragma once

#include "kv_cache_manager/common/jsonizable.h"

namespace kv_cache_manager {
enum class AccountRole : uint8_t {
    ROLE_USER = 0,
    ROLE_ADMIN = 1,
};

class Account : public Jsonizable {
public:
    Account() : user_name_(""), password_(""), role_(AccountRole::ROLE_USER) {}
    Account(const std::string &user_name, const std::string &password, const AccountRole &role)
        : user_name_(user_name), password_(password), role_(role) {}
    bool FromRapidValue(const rapidjson::Value &rapid_value) override { 
        KVCM_JSON_GET_MACRO(rapid_value, "user_name", user_name_);
        KVCM_JSON_GET_MACRO(rapid_value, "password", password_);
        KVCM_JSON_GET_MACRO(rapid_value, "role", role_);
        return true;
    }
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {
        Put(writer, "user_name", user_name_);
        Put(writer, "password", password_);
        Put(writer, "role", role_);
    }
    AccountRole role() const { return role_; }
    const std::string &user_name() const { return user_name_; }
    void set_user_name(const std::string &name) { user_name_ = name; }

    void set_role(AccountRole r) { role_ = r; }

private:
    std::string user_name_;
    std::string password_;
    AccountRole role_;
};
} // namespace kv_cache_manager
