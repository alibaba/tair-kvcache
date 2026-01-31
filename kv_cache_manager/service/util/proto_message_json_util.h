#pragma once

#include <google/protobuf/message.h>
#include <string>

namespace kv_cache_manager {

class ProtoMessageJsonUtil {
public:
    static bool ToJson(const ::google::protobuf::Message *message, std::string &json);
    static bool FromJson(const std::string &json, ::google::protobuf::Message *message);
};

} // namespace kv_cache_manager