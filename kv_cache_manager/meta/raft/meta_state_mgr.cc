#include "kv_cache_manager/meta/raft/meta_state_mgr.h"

#include <libnuraft/buffer.hxx>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {
namespace raft_meta {

using nuraft::buffer;
using nuraft::cluster_config;
using nuraft::cs_new;
using nuraft::log_store;
using nuraft::ptr;
using nuraft::srv_config;
using nuraft::srv_state;

MetaStateMgr::MetaStateMgr(int32_t server_id,
                           std::string self_endpoint,
                           std::string self_aux,
                           std::vector<PeerEntry> initial_peers,
                           std::string state_dir,
                           ptr<log_store> log_store_arg)
    : server_id_(server_id),
      self_endpoint_(std::move(self_endpoint)),
      self_aux_(std::move(self_aux)),
      initial_peers_(std::move(initial_peers)),
      state_dir_(std::move(state_dir)),
      log_store_(std::move(log_store_arg)) {
    if (!EnsureDir(state_dir_)) {
        KVCM_LOG_ERROR("MetaStateMgr: failed to create state dir[%s]", state_dir_.c_str());
    }
}

bool MetaStateMgr::EnsureDir(const std::string &dir) {
    if (dir.empty()) {
        return false;
    }
    // mkdir -p semantics for multi-segment paths.
    std::string acc;
    for (size_t i = 0; i < dir.size(); ++i) {
        if (dir[i] == '/' && i > 0 && !acc.empty()) {
            if (mkdir(acc.c_str(), 0755) != 0 && errno != EEXIST) {
                return false;
            }
        }
        acc.push_back(dir[i]);
    }
    if (mkdir(acc.c_str(), 0755) != 0 && errno != EEXIST) {
        return false;
    }
    return true;
}

bool MetaStateMgr::AtomicWriteFile(const std::string &path, const void *data, size_t len) {
    std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) {
            return false;
        }
        out.write(static_cast<const char *>(data), static_cast<std::streamsize>(len));
        if (!out) {
            return false;
        }
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        return false;
    }
    return true;
}

bool MetaStateMgr::ReadFile(const std::string &path, std::string &out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    out = ss.str();
    return true;
}

std::string MetaStateMgr::ConfigPath() const { return state_dir_ + "/cluster_config"; }

std::string MetaStateMgr::StatePath() const { return state_dir_ + "/srv_state"; }

ptr<cluster_config> MetaStateMgr::BuildDefaultConfig() const {
    auto cfg = cs_new<cluster_config>();
    bool found_self = false;
    for (const auto &peer : initial_peers_) {
        // For the local node, prefer self_aux_ (authoritative locally) over
        // anything the operator wrote into the peers list — they may have
        // forgotten or mistyped it. Other peers' aux is replicated to us
        // later by the leader once the cluster forms.
        std::string aux = (peer.server_id == server_id_) ? self_aux_ : peer.aux;
        auto srv = cs_new<srv_config>(peer.server_id, 0, peer.endpoint, aux, peer.is_learner);
        cfg->get_servers().push_back(srv);
        if (peer.server_id == server_id_) {
            found_self = true;
        }
    }
    if (!found_self) {
        // Always include self even if caller forgot — without this NuRaft
        // refuses to start.
        cfg->get_servers().push_back(cs_new<srv_config>(server_id_, 0, self_endpoint_, self_aux_, false));
    }
    return cfg;
}

ptr<cluster_config> MetaStateMgr::load_config() {
    std::lock_guard<std::mutex> g(mutex_);
    std::string raw;
    if (!ReadFile(ConfigPath(), raw) || raw.empty()) {
        return BuildDefaultConfig();
    }
    ptr<buffer> buf = buffer::alloc(raw.size());
    std::memcpy(buf->data_begin(), raw.data(), raw.size());
    return cluster_config::deserialize(*buf);
}

void MetaStateMgr::save_config(const cluster_config &config) {
    std::lock_guard<std::mutex> g(mutex_);
    ptr<buffer> buf = const_cast<cluster_config &>(config).serialize();
    buf->pos(0);
    if (!AtomicWriteFile(ConfigPath(), buf->data_begin(), buf->size())) {
        KVCM_LOG_ERROR("MetaStateMgr: failed to persist cluster_config to[%s]", ConfigPath().c_str());
    }
}

void MetaStateMgr::save_state(const srv_state &state) {
    std::lock_guard<std::mutex> g(mutex_);
    ptr<buffer> buf = state.serialize();
    buf->pos(0);
    if (!AtomicWriteFile(StatePath(), buf->data_begin(), buf->size())) {
        KVCM_LOG_ERROR("MetaStateMgr: failed to persist srv_state to[%s]", StatePath().c_str());
    }
}

ptr<srv_state> MetaStateMgr::read_state() {
    std::lock_guard<std::mutex> g(mutex_);
    std::string raw;
    if (!ReadFile(StatePath(), raw) || raw.empty()) {
        return nullptr;
    }
    ptr<buffer> buf = buffer::alloc(raw.size());
    std::memcpy(buf->data_begin(), raw.data(), raw.size());
    return srv_state::deserialize(*buf);
}

ptr<log_store> MetaStateMgr::load_log_store() { return log_store_; }

nuraft::int32 MetaStateMgr::server_id() { return server_id_; }

void MetaStateMgr::system_exit(const int exit_code) {
    KVCM_LOG_ERROR("MetaStateMgr: NuRaft requested system_exit code[%d]", exit_code);
}

} // namespace raft_meta
} // namespace kv_cache_manager
