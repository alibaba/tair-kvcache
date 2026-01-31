#pragma once
#include <string>
#include <vector>

namespace kv_cache_manager {

class VIPServerSubscriber {
public:
    bool init(const std::string &domain);
    bool getOneAddress(std::string &address) const;
    bool getAllAddresses(std::vector<std::string> &addresses) const;
};

} // namespace kv_cache_manager