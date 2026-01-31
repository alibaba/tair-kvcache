#include "kv_cache_manager/event/event_manager.h"

#include <sstream>

#include "kv_cache_manager/common/logger.h"
namespace kv_cache_manager {
EventManager::EventManager() = default;
EventManager::~EventManager() { Stop(); }

bool EventManager::Init() {
    if (initialized_.load()) {
        KVCM_LOG_WARN("EventManager already initialized");
        return true;
    }
    initialized_ = true;
    KVCM_LOG_INFO("EventManager initialized");
    return true;
}

bool EventManager::Stop() {
    if (!initialized_.load()) {
        return true;
    }

    for (auto &publisher_pair : registered_publishers_) {
        if (publisher_pair.second) {
            publisher_pair.second->Stop();
        }
    }
    registered_publishers_.clear();
    return true;
}

bool EventManager::RegisterPublisher(const std::string &unique_name, std::shared_ptr<EventPublisher> publisher) {
    if (unique_name.empty() || !publisher) {
        KVCM_LOG_ERROR("Invalid parameters to register publisher");
        return false;
    }
    std::unique_lock<std::shared_mutex> lock(publishers_mutex_);

    if (registered_publishers_.find(unique_name) != registered_publishers_.end()) {
        KVCM_LOG_WARN("Publisher with name %s already exists", unique_name.c_str());
        return false;
    }
    registered_publishers_[unique_name] = publisher;
    publisher->set_name(unique_name);
    KVCM_LOG_INFO("Publisher with name %s registered", unique_name.c_str());
    return true;
}

bool EventManager::RemovePublisher(const std::string &unique_name) {
    std::unique_lock<std::shared_mutex> lock(publishers_mutex_);
    auto it = registered_publishers_.find(unique_name);
    if (it == registered_publishers_.end()) {
        KVCM_LOG_WARN("Publisher with name %s not found", unique_name.c_str());
        return false;
    }
    if (it->second) {
        it->second->Stop();
    }
    registered_publishers_.erase(it);
    KVCM_LOG_INFO("Publisher with name %s removed", unique_name.c_str());
    return true;
}
std::shared_ptr<EventPublisher> EventManager::GetPublisher(const std::string &unique_name) const {
    std::shared_lock<std::shared_mutex> lock(publishers_mutex_);
    auto it = registered_publishers_.find(unique_name);
    if (it != registered_publishers_.end()) {
        return it->second;
    }
    return nullptr;
}
bool EventManager::HasPublisher(const std::string &unique_name) const {
    std::shared_lock<std::shared_mutex> lock(publishers_mutex_);
    return registered_publishers_.find(unique_name) != registered_publishers_.end();
}

std::vector<std::string> EventManager::ListPublishers() const {
    std::vector<std::string> publisher_names;
    std::shared_lock<std::shared_mutex> lock(publishers_mutex_);
    for (const auto &publisher_pair : registered_publishers_) {
        publisher_names.push_back(publisher_pair.first);
    }
    return publisher_names;
}
void EventManager::ClearPublishers() {
    std::unique_lock<std::shared_mutex> lock(publishers_mutex_);
    for (auto &publisher_pair : registered_publishers_) {
        if (publisher_pair.second) {
            publisher_pair.second->Stop();
        }
    }
    registered_publishers_.clear();
    KVCM_LOG_INFO("All publishers cleared");
}

bool EventManager::Publish(const std::shared_ptr<BaseEvent> &event) {
    if (!event) {
        KVCM_LOG_ERROR("Invalid event to publish");
        return false;
    }
    KVCM_LOG_DEBUG("Publishing event ID: %s", event->event_id().c_str());

    std::vector<std::pair<std::string, std::shared_ptr<EventPublisher>>> publishers_copy;
    {
        std::shared_lock<std::shared_mutex> lock(publishers_mutex_);
        publishers_copy.reserve(registered_publishers_.size());
        for (const auto &pair : registered_publishers_) {
            if (pair.second) {
                publishers_copy.emplace_back(pair.first, pair.second);
            }
        }
    }
    for (const auto &publisher_pair : publishers_copy) {
        if (!publisher_pair.second->Publish(event)) {
            KVCM_LOG_WARN("Failed to push event to publisher %s", publisher_pair.first.c_str());
        }
    }

    return true;
}

size_t EventManager::GetPublisherCount() const {
    std::shared_lock<std::shared_mutex> lock(publishers_mutex_);
    return registered_publishers_.size();
}
} // namespace kv_cache_manager