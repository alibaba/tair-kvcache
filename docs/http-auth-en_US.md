# HTTP Bearer Authentication

KVCacheManager exposes three HTTP services on two ports:

| Service | Default Port | Authentication |
|---|---|---|
| Meta   | 6382 | always open (data plane) |
| Admin  | 6492 | optional Bearer token |
| Debug  | 6492 | optional Bearer token |

The Admin and Debug HTTP services can be protected with HTTP Bearer
authentication ([RFC 6750]). The Meta HTTP service is the data-plane
endpoint used by inference engines and is intentionally left open; it
is expected to be reachable only over a trusted network.

[RFC 6750]: https://datatracker.ietf.org/doc/html/rfc6750

## Quick Start

Set one or more accepted tokens via the `kvcm.service.admin_auth_token`
config key:

```bash
# single token
kv_cache_manager_bin -e 'kvcm.service.admin_auth_token=s3cret-token'

# multiple tokens (comma-separated, for staged rotation)
kv_cache_manager_bin -e 'kvcm.service.admin_auth_token=tok-old,tok-new'
```

Then call the protected endpoints with an `Authorization: Bearer …`
header:

```bash
# Prometheus scrape (Admin port 6492)
curl -H 'Authorization: Bearer s3cret-token' \
     http://<host>:6492/metrics

# Health check
curl -H 'Authorization: Bearer s3cret-token' \
     http://<host>:6492/api/healthy

# Debug fault injection
curl -X POST \
     -H 'Authorization: Bearer s3cret-token' \
     -H 'Content-Type: application/json' \
     -d '{"fault":"…"}' \
     http://<host>:6492/api/injectFault
```

When `kvcm.service.admin_auth_token` is empty (default), Admin and
Debug run unauthenticated and the server logs a `WARN` at startup:

```
admin/debug HTTP auth disabled (kvcm.service.admin_auth_token not set);
do not expose admin/debug ports on untrusted networks
```

## Configuration

| Key | Default | Description |
|---|---|---|
| `kvcm.service.admin_auth_token` | empty | comma-separated list of accepted Bearer tokens; empty disables auth |

The value can be set via the config file, the `--env` / `-e` flag, or
an environment variable (with `.` replaced by `_`):

```bash
# config file
kvcm.service.admin_auth_token=tok-old, tok-new

# command line
kv_cache_manager_bin -e 'kvcm.service.admin_auth_token=tok-old,tok-new'

# environment
export kvcm_service_admin_auth_token='tok-old,tok-new'
```

Whitespace around each comma-separated entry is trimmed. Empty entries
(including trailing commas) are silently dropped.

### Token rotation

Listing multiple tokens is the supported way to rotate without
downtime:

1. Deploy with `old,new` — both old and new clients are accepted.
2. Migrate clients to the new token.
3. Redeploy with only `new` to retire the old token.

A configuration reload requires a process restart.

## Prometheus Scraping

The `/metrics` endpoint is served by the Admin HTTP service and is
therefore subject to the same Bearer auth when enabled. Configure your
Prometheus scrape job with `authorization`:

```yaml
scrape_configs:
  - job_name: kvcache_manager
    metrics_path: /metrics
    static_configs:
      - targets: ["<host>:6492"]
    authorization:
      type: Bearer
      credentials: s3cret-token
      # or: credentials_file: /etc/prometheus/kvcm_token
```

## Response Behavior

Authenticated requests are dispatched to the backend handler unchanged.
Failed authentication produces an HTTP `401 Unauthorized` response with
a `WWW-Authenticate` challenge per [RFC 7235] §4.1 and [RFC 6750] §3:

[RFC 7235]: https://datatracker.ietf.org/doc/html/rfc7235

```http
HTTP/1.1 401 Unauthorized
WWW-Authenticate: Bearer realm="kvcm"
Content-Type: application/json

{"error":"unauthorized"}
```

When the credential is present but malformed or rejected, the
`WWW-Authenticate` header carries an `error` parameter so clients can
distinguish the cases:

| Condition | `WWW-Authenticate` value |
|---|---|
| no `Authorization` header | `Bearer realm="kvcm"` |
| header is not the Bearer scheme, or is malformed | `Bearer realm="kvcm", error="invalid_request"` |
| Bearer scheme with an unknown token | `Bearer realm="kvcm", error="invalid_token"` |

Each rejected request is logged at `WARN` level for audit:

```
[AUTH] denied api=/metrics outcome=3 ip=10.0.0.42
```

`outcome` is the numeric value of the `AuthOutcome` enum
(`1=missing`, `2=invalid_request`, `3=invalid_token`).

## Design Considerations

The implementation lives under
`kv_cache_manager/service/http_service/auth/`.

### Scope: Admin & Debug only

Bearer auth is wired only on the Admin and Debug HTTP services because
they expose mutating operations (config snapshot load, fault
injection, debug RPCs) and observability data (`/metrics`). The Meta
HTTP service is the hot data-plane endpoint and stays open by design;
deployments are expected to firewall it to inference clusters.

### Wrapping order: `logger(auth(handler))`

`CoroHttpService::Start` wraps every registered handler with a logger
middleware on the outside and an auth middleware on the inside:

```
client request -> logger -> auth -> handler -> response
```

Placing the logger outermost ensures `401` responses are still
captured by the request/response audit log.

### Pluggable verifier

Authentication is dispatched through a `TokenVerifier` interface
(`token_verifier.h`):

```cpp
class TokenVerifier {
public:
    virtual AuthOutcome Verify(std::string_view authz_header) const = 0;
    virtual std::string Realm() const { return "kvcm"; }
};
```

The current concrete implementation, `StaticBearerTokenVerifier`,
checks the header against a fixed in-memory list. The interface leaves
room for future verifiers (e.g. JWT validation, OAuth2 introspection)
without touching the HTTP service plumbing.

### Header parsing

`StaticBearerTokenVerifier` follows [RFC 7235] §2.1 and [RFC 6750] §2:

- the `Authorization` header value is trimmed of leading/trailing
  optional whitespace (SP / HTAB)
- the scheme name `Bearer` is matched case-insensitively
- at least one SP or HTAB must separate the scheme and the token; an
  adjacent token (e.g. `BearerXYZ`) is rejected as `invalid_request`
- internal whitespace inside the token is rejected
- the resulting token is compared against the accepted list using
  constant-time equality

### Constant-time comparison

`AuthUtil::ConstantTimeEquals` performs a length-revealing constant-
time compare in `O(min(len(a), len(b)))` regardless of where the first
byte mismatches. This defeats naive timing oracles on the matching
prefix. Token lengths should be kept bounded so that the length
itself is not a useful side channel.

### Zero-overhead opt-out

If no token verifier is attached, `WrapWithAuth` returns the original
handler unchanged. Disabling auth therefore costs nothing per
request — it is exactly the previous code path.

### Out of scope

The following are intentionally not part of this feature:

- **TLS / HTTPS termination.** Bearer tokens travel in clear text in
  the `Authorization` header. Production deployments should terminate
  TLS at a reverse proxy or load balancer in front of the Admin and
  Debug ports.
- **Per-route ACLs.** All routes on the Admin and Debug services share
  a single token list. There is no notion of read-only vs. admin
  tokens; if you need that, use separate deployments or front the
  service with a proxy that enforces it.
- **Hot reload.** Tokens are read once at startup. Rotating a token
  requires a restart with the new list.
- **Token issuance.** Tokens are operator-supplied opaque strings;
  this service does not mint them.

## Files

| Path | Role |
|---|---|
| `kv_cache_manager/service/http_service/auth/token_verifier.h` | `TokenVerifier` interface and `AuthOutcome` enum |
| `kv_cache_manager/service/http_service/auth/static_bearer_token_verifier.{h,cc}` | static-list Bearer verifier |
| `kv_cache_manager/service/http_service/auth/auth_util.{h,cc}` | constant-time and case-insensitive helpers |
| `kv_cache_manager/service/http_service/coro_http_service.{h,cc}` | `WrapWithAuth` middleware and `SetTokenVerifier` |
| `kv_cache_manager/service/server.cc` | wires the verifier onto Admin and Debug at startup |
| `kv_cache_manager/service/server_config.{h,cc}` | parses `kvcm.service.admin_auth_token` |
