#pragma once

#include <map>
#include <string>

namespace kv_cache_manager {

// Helper for V6D LocationSpec.uri (V8 §2.1.2).
//
// Standard form:
//   vineyard://{ip}:{port}/{medium}?{key1}={val1}&{key2}={val2}
//
// VineyardUri does NOT replace the generic StandardUri parser; it provides:
//   - structured composition (Build) so callers don't hand-format URIs
//   - bidirectional conversion to/from V6D's legacy {addr,type,gpu,expire_at}
//     JSON shape, kept for V6D clients that still emit/consume JSON
//
// KVCM itself never interprets the medium / query parameters; they are forwarded
// transparently to V6D clients that resolve the URI via StandardUri::Parse.
class VineyardUri {
public:
    // Compose `vineyard://{host_ip_port}/{medium}?k=v&...` from parts.
    // params may be empty. host_ip_port should be a literal "ip:port" string;
    // medium is appended as the URI path (without leading '/').
    static std::string Build(const std::string &host_ip_port,
                             const std::string &medium,
                             const std::map<std::string, std::string> &params = {});

    // Parse a vineyard URI back into structured pieces.
    // Returns false if scheme != "vineyard" or the URI fails StandardUri::Parse.
    // On success: out_host_ip_port = "ip:port"; out_medium = path without
    // leading '/'; out_params = key/value pairs from the query string.
    static bool Parse(const std::string &uri,
                      std::string &out_host_ip_port,
                      std::string &out_medium,
                      std::map<std::string, std::string> &out_params);

    // Convert legacy V6D {"addr","type","gpu","expire_at"} JSON to vineyard URI.
    // Returns empty string on parse failure.
    //   - "addr"      -> host_ip_port (required)
    //   - "type"      -> medium (path; default "mem" if missing)
    //   - other keys (gpu, expire_at, ...) -> query params
    static std::string FromJson(const std::string &json);

    // Convert vineyard URI back to legacy JSON shape:
    //   {"addr":"ip:port","type":"<medium>","gpu":"...","expire_at":"..."}
    // Returns empty string on parse failure.
    static std::string ToJson(const std::string &uri);
};

} // namespace kv_cache_manager
