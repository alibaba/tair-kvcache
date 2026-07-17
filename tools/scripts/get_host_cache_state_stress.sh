#!/usr/bin/env bash

# Stress and correctness checks for ReportEvent + GetHostCacheState.
#
# The script uses curl for every KVCM request and creates a unique synthetic
# instance by default. Reusing an existing instance requires SKIP_REGISTER=1
# and an explicitly supplied INSTANCE_ID.

set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:6382}"
ADMIN_URL="${ADMIN_URL:-http://127.0.0.1:6492}"
INSTANCE_GROUP="${INSTANCE_GROUP:-kvcm-ghcs-stress-group-$$}"
EXPLICIT_INSTANCE_ID="${INSTANCE_ID:-}"
INSTANCE_ID="${INSTANCE_ID:-kvcm-ghcs-stress-$(date +%Y%m%d%H%M%S)-$$}"
HOST_COUNT="${HOST_COUNT:-3}"
BLOCK_COUNT="${BLOCK_COUNT:-1000}"
WRITE_BATCH_SIZE="${WRITE_BATCH_SIZE:-100}"
WRITE_CONCURRENCY="${WRITE_CONCURRENCY:-10}"
READ_CONCURRENCY="${READ_CONCURRENCY:-10}"
READ_ROUNDS="${READ_ROUNDS:-20}"
CONCURRENT_READ_ROUNDS="${CONCURRENT_READ_ROUNDS:-30}"
CURL_CONNECT_TIMEOUT="${CURL_CONNECT_TIMEOUT:-3}"
CURL_MAX_TIME="${CURL_MAX_TIME:-15}"
MEDIUM="${MEDIUM:-mem}"
SECONDARY_MEDIUM="${SECONDARY_MEDIUM:-disk}"
FULL_SPEC_NAME="${FULL_SPEC_NAME:-full_attention}"
MAMBA_SPEC_NAME="${MAMBA_SPEC_NAME:-mamba_state}"
EVENT_HEARTBEAT_TIMEOUT_MS="${EVENT_HEARTBEAT_TIMEOUT_MS:-300000}"
EVENT_CLEANUP_GRACE_MS="${EVENT_CLEANUP_GRACE_MS:-300000}"
EVENT_LIVENESS_CHECK_INTERVAL_MS="${EVENT_LIVENESS_CHECK_INTERVAL_MS:-5000}"
# Keep the default below 2^53 because jq 1.6 represents JSON numbers as doubles.
KEY_BASE="${KEY_BASE:-$((2000000000000 + ($$ % 1000000) * 1000000))}"
SKIP_REGISTER="${SKIP_REGISTER:-0}"
BOOTSTRAP_TEST_GROUP="${BOOTSTRAP_TEST_GROUP:-0}"
CLEANUP="${CLEANUP:-1}"
KEEP_ARTIFACTS="${KEEP_ARTIFACTS:-0}"
RUN_CASES="${RUN_CASES:-all}"

WORK_DIR=""
REGISTERED=0
BOOTSTRAPPED=0
PASS_COUNT=0
FAIL_COUNT=0
DUMMY_STORAGE_NAME="${DUMMY_STORAGE_NAME:-${INSTANCE_GROUP}-dummy}"
EVENT_STORAGE_NAME="${EVENT_STORAGE_NAME:-${INSTANCE_GROUP}-event-report}"

usage() {
    cat <<'EOF'
GetHostCacheState stress and correctness tester

USAGE
  ./tools/scripts/get_host_cache_state_stress.sh
  ./tools/scripts/get_host_cache_state_stress.sh --help

The tool drives ReportEvent and GetHostCacheState exclusively with curl. jq is
used only to build and validate JSON. Required commands: bash, curl, jq, xargs,
and awk.

QUICK START: ISOLATED MAMBA/FULL-ATTENTION REPRODUCTION
  BOOTSTRAP_TEST_GROUP=1 \
  RUN_CASES=mamba \
  BLOCK_COUNT=1000 \
  WRITE_CONCURRENCY=10 \
  READ_CONCURRENCY=10 \
  ./tools/scripts/get_host_cache_state_stress.sh

This creates uniquely named dummy/event-report storages, an instance group, a
synthetic instance, and synthetic hosts. CLEANUP=1 removes them on exit.

ENDPOINTS
  BASE_URL                 MetaService URL
                           default: http://127.0.0.1:6382
  ADMIN_URL                AdminService URL; used for bootstrap and cleanup
                           default: http://127.0.0.1:6492
  CURL_CONNECT_TIMEOUT     curl connection timeout in seconds (default 3)
  CURL_MAX_TIME            per-request timeout in seconds (default 15)

TARGET AND SAFETY
  INSTANCE_GROUP           Existing group with an event-report backend. It is
                           required unless BOOTSTRAP_TEST_GROUP=1.
  INSTANCE_ID              Synthetic instance id (unique value by default)
  BOOTSTRAP_TEST_GROUP     1: create an isolated group and storages; 0: use the
                           existing INSTANCE_GROUP (default 0)
  SKIP_REGISTER            1: operate on an explicitly supplied INSTANCE_ID.
                           This disables cleanup and can modify that instance.
                           With default 0, leave INSTANCE_ID unset to use an
                           auto-generated unique synthetic instance.
  CLEANUP                  Remove created resources on exit (default 1)
  KEEP_ARTIFACTS           Keep generated request/response files (default 0)

LOAD
  HOST_COUNT               Base synthetic hosts; must be >= 3 (default 3)
  BLOCK_COUNT              Blocks per host; must be >= 8 (default 1000)
  WRITE_BATCH_SIZE         BLOCK_ADD items per HTTP request (default 100)
  WRITE_CONCURRENCY        Concurrent write requests (default 10)
  READ_CONCURRENCY         Concurrent GetHostCacheState requests (default 10)
  READ_ROUNDS              Read rounds per worker (default 20)
  CONCURRENT_READ_ROUNDS   Reads per worker during add-only writes (default 30)

DATA MODEL
  MEDIUM                   Primary medium (default mem)
  SECONDARY_MEDIUM         Secondary medium for filter tests (default disk)
  FULL_SPEC_NAME           Full-attention LocationSpec name
                           default: full_attention
  MAMBA_SPEC_NAME          Mamba/linear LocationSpec name
                           default: mamba_state
  EVENT_HEARTBEAT_TIMEOUT_MS
                           Synthetic event-report node heartbeat timeout.
                           Increase this for long read stress runs.
                           default: 300000
  EVENT_CLEANUP_GRACE_MS   Synthetic event-report cleanup grace.
                           default: 300000
  EVENT_LIVENESS_CHECK_INTERVAL_MS
                           Synthetic event-report liveness check interval.
                           default: 5000
  KEY_BASE                 First synthetic int64 block key (unique by default)

CASE SELECTION
  RUN_CASES                Comma-separated cases, or all (default all).
                           bulk always runs first as the common fixture.

Cases:
  bulk          concurrent batched BLOCK_ADD and full-prefix verification
  read_stress   concurrent reads; every response must match all blocks
  prefix        per-host deletes at different positions, then repair
  idempotent    duplicate BLOCK_ADD requests remain correct
  medium        union and medium-filter behavior
  ordering      reversed and duplicate query keys
  concurrent    ordered ADD while concurrent GET; prefixes cannot regress
  race          same-key ADD/DELETE races followed by deterministic repair
  mamba         full/Mamba add order, query isolation, and component deletes

RESULTS
  Exit 0 only when every selected assertion passes. [BUG], [FAIL], [FATAL], or
  a non-zero exit indicates incorrect behavior or an invalid environment.
  Cleanup is best effort; warnings include the artifact directory when manual
  inspection or cleanup may be required.

EXAMPLES
  # Run every case in a fresh isolated group.
  BOOTSTRAP_TEST_GROUP=1 BLOCK_COUNT=10000 WRITE_CONCURRENCY=20 \
    READ_CONCURRENCY=10 READ_ROUNDS=100 \
    ./tools/scripts/get_host_cache_state_stress.sh

  # Use an existing group; only the unique synthetic instance is created.
  BLOCK_COUNT=10000 WRITE_CONCURRENCY=20 READ_CONCURRENCY=10 READ_ROUNDS=100 \
    BASE_URL=http://127.0.0.1:6382 INSTANCE_GROUP=event_report_test_group \
    ./tools/scripts/get_host_cache_state_stress.sh

  # Keep all curl payloads and responses for diagnosis.
  RUN_CASES=bulk,read_stress READ_CONCURRENCY=10 READ_ROUNDS=300 \
    KEEP_ARTIFACTS=1 INSTANCE_GROUP=event_report_test_group \
    ./tools/scripts/get_host_cache_state_stress.sh
EOF
}

log() {
    printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

pass() {
    PASS_COUNT=$((PASS_COUNT + 1))
    printf '[PASS] %s\n' "$*"
}

fail() {
    FAIL_COUNT=$((FAIL_COUNT + 1))
    printf '[FAIL] %s\n' "$*" >&2
    return 1
}

die() {
    printf '[FATAL] %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

check_admin_service() {
    local endpoint="${ADMIN_URL}/api/healthy"
    local response="$WORK_DIR/admin_healthy.response.json"
    local result

    result="$(curl --silent --show-error \
        --connect-timeout "$CURL_CONNECT_TIMEOUT" \
        --max-time "$CURL_MAX_TIME" \
        --output "$response" \
        --write-out '%{http_code} %{time_total}' \
        "$endpoint")" || die "AdminService health check failed at $endpoint"
    case "$result" in
        2*) ;;
        *)
            die "AdminService health check failed at $endpoint: $result; set ADMIN_URL to the AdminService HTTP endpoint"
            ;;
    esac
    jq -e '.status == "OK"' "$response" >/dev/null || {
        printf '[ERROR] unexpected AdminService health response: ' >&2
        jq -c . "$response" >&2 || true
        die "AdminService health check returned an unexpected response"
    }
    pass "AdminService is healthy at $ADMIN_URL"
}

case_enabled() {
    local name="$1"
    [[ "$RUN_CASES" == "all" || ",$RUN_CASES," == *",$name,"* ]]
}

status_code() {
    jq -r '.header.status.code // empty' "$1"
}

assert_ok_response() {
    local response_file="$1"
    local context="$2"
    local code
    code="$(status_code "$response_file")"
    if [[ "$code" != "OK" ]]; then
        printf '[ERROR] %s returned code=%s: ' "$context" "${code:-missing}" >&2
        jq -c . "$response_file" >&2 || true
        return 1
    fi
}

post_json() {
    local url="$1"
    local payload_file="$2"
    local response_file="$3"
    local meta_file="${response_file}.meta"
    local http_code

    http_code="$(curl --silent --show-error \
        --connect-timeout "$CURL_CONNECT_TIMEOUT" \
        --max-time "$CURL_MAX_TIME" \
        --header 'Content-Type: application/json' \
        --header 'Accept: application/json' \
        --request POST \
        --data-binary "@$payload_file" \
        --output "$response_file" \
        --write-out '%{http_code} %{time_total}' \
        "$url")" || return 1
    printf '%s\n' "$http_code" >"$meta_file"
    [[ "$http_code" == 2* ]] || {
        printf '[ERROR] HTTP failure for %s: %s\n' "$url" "$http_code" >&2
        return 1
    }
}

post_and_check() {
    local endpoint="$1"
    local payload_file="$2"
    local response_file="$3"
    local context="$4"
    post_json "${BASE_URL}${endpoint}" "$payload_file" "$response_file"
    assert_ok_response "$response_file" "$context"
}

parallel_post_dir() {
    local endpoint="$1"
    local payload_dir="$2"
    local concurrency="$3"
    local label="$4"
    local file_count

    file_count="$(find "$payload_dir" -type f -name '*.json' | wc -l | tr -d ' ')"
    [[ "$file_count" -gt 0 ]] || die "$label has no payload files"
    log "$label: posting $file_count requests with concurrency=$concurrency"

    export BASE_URL CURL_CONNECT_TIMEOUT CURL_MAX_TIME
    find "$payload_dir" -type f -name '*.json' -print0 | \
        xargs -0 -n 1 -P "$concurrency" sh -c '
            payload=$1
            response="${payload}.response"
            meta="${response}.meta"
            result=$(curl --silent --show-error \
                --connect-timeout "$CURL_CONNECT_TIMEOUT" \
                --max-time "$CURL_MAX_TIME" \
                --header "Content-Type: application/json" \
                --header "Accept: application/json" \
                --request POST \
                --data-binary "@$payload" \
                --output "$response" \
                --write-out "%{http_code} %{time_total}" \
                "${BASE_URL}'"$endpoint"'") || exit 10
            printf "%s\n" "$result" >"$meta"
            case "$result" in
                2*) ;;
                *) printf "HTTP failure payload=%s result=%s\n" "$payload" "$result" >&2; exit 11 ;;
            esac
            code=$(jq -r ".header.status.code // empty" "$response")
            if [ "$code" != OK ]; then
                printf "KVCM failure payload=%s code=%s response=" "$payload" "$code" >&2
                jq -c . "$response" >&2
                exit 12
            fi
        ' sh

    pass "$label completed without HTTP or KVCM errors"
}

host_for_index() {
    local index="$1"
    printf '198.18.%d.%d:8080' $((index / 250)) $((index % 250 + 1))
}

build_register_payload() {
    jq -nc \
        --arg trace "ghcs_register_${INSTANCE_ID}" \
        --arg group "$INSTANCE_GROUP" \
        --arg instance "$INSTANCE_ID" \
        --arg full_spec "$FULL_SPEC_NAME" \
        --arg mamba_spec "$MAMBA_SPEC_NAME" \
        '{
            trace_id: $trace,
            instance_group: $group,
            instance_id: $instance,
            block_size: 128,
            model_deployment: {
                model_name: "get_host_cache_state_stress",
                dtype: "FP8",
                use_mla: false,
                tp_size: 1,
                dp_size: 1,
                pp_size: 1
            },
            location_spec_infos: [
                {name: "tp0", size: 1024},
                {name: $full_spec, size: 1024},
                {name: $mamba_spec, size: 1024}
            ],
            location_spec_groups: [
                {name: "full", spec_names: [$full_spec]},
                {name: "linear", spec_names: [$mamba_spec]}
            ]
        }'
}

build_report_payload() {
    local host="$1"
    local event_type="$2"
    local medium="$3"
    local key_base="$4"
    local start="$5"
    local end="$6"
    local trace="$7"

    jq -nc \
        --arg trace "$trace" \
        --arg instance "$INSTANCE_ID" \
        --arg host "$host" \
        --arg event_type "$event_type" \
        --arg medium "$medium" \
        --argjson key_base "$key_base" \
        --argjson start "$start" \
        --argjson end "$end" '
        def add_event($key): {
            event_type: "EVENT_BLOCK_ADD",
            block_add: {
                block_key: ($key | tostring),
                medium: $medium,
                specs: [{
                    name: "tp0",
                    uri: ("vineyard://" + $host + "/" + $medium)
                }]
            }
        };
        def delete_event($key): {
            event_type: "EVENT_BLOCK_DELETE",
            block_delete: {
                block_key: ($key | tostring),
                medium: $medium,
                spec_names: ["tp0"]
            }
        };
        {
            trace_id: $trace,
            instance_id: $instance,
            host_ip_port: $host,
            storage_type: "ST_EVENT_REPORT",
            events: [range($start; $end) as $i |
                if $event_type == "add" then add_event($key_base + $i)
                else delete_event($key_base + $i)
                end]
        }'
}

build_node_register_payload() {
    local host="$1"
    local trace="$2"
    jq -nc \
        --arg trace "$trace" \
        --arg instance "$INSTANCE_ID" \
        --arg host "$host" \
        --arg medium "$MEDIUM" \
        --arg secondary "$SECONDARY_MEDIUM" '
        {
            trace_id: $trace,
            instance_id: $instance,
            host_ip_port: $host,
            storage_type: "ST_EVENT_REPORT",
            events: [{
                event_type: "EVENT_NODE_REGISTER",
                node_register: {mediums: [$medium, $secondary]}
            }]
        }'
}

build_hybrid_add_payload() {
    local host="$1"
    local key_base="$2"
    local start="$3"
    local end="$4"
    local trace="$5"
    jq -nc \
        --arg trace "$trace" \
        --arg instance "$INSTANCE_ID" \
        --arg host "$host" \
        --arg medium "$MEDIUM" \
        --arg full_spec "$FULL_SPEC_NAME" \
        --arg mamba_spec "$MAMBA_SPEC_NAME" \
        --argjson key_base "$key_base" \
        --argjson start "$start" \
        --argjson end "$end" '
        {
            trace_id: $trace,
            instance_id: $instance,
            host_ip_port: $host,
            storage_type: "ST_EVENT_REPORT",
            events: [range($start; $end) as $i | {
                event_type: "EVENT_BLOCK_ADD",
                block_add: {
                    block_key: ($key_base + $i | tostring),
                    medium: $medium,
                    specs: [
                        {name: $full_spec, uri: ("vineyard://" + $host + "/" + $medium)},
                        {name: $mamba_spec, uri: ("vineyard://" + $host + "/" + $medium)}
                    ]
                }
            }]
        }'
}

build_single_spec_add_payload() {
    local host="$1"
    local key_base="$2"
    local start="$3"
    local end="$4"
    local spec_name="$5"
    local trace="$6"
    jq -nc \
        --arg trace "$trace" \
        --arg instance "$INSTANCE_ID" \
        --arg host "$host" \
        --arg medium "$MEDIUM" \
        --arg spec_name "$spec_name" \
        --argjson key_base "$key_base" \
        --argjson start "$start" \
        --argjson end "$end" '
        {
            trace_id: $trace,
            instance_id: $instance,
            host_ip_port: $host,
            storage_type: "ST_EVENT_REPORT",
            events: [range($start; $end) as $i | {
                event_type: "EVENT_BLOCK_ADD",
                block_add: {
                    block_key: ($key_base + $i | tostring),
                    medium: $medium,
                    specs: [{
                        name: $spec_name,
                        uri: ("vineyard://" + $host + "/" + $medium)
                    }]
                }
            }]
        }'
}

build_component_delete_payload() {
    local host="$1"
    local key="$2"
    local spec_name="$3"
    local trace="$4"
    jq -nc \
        --arg trace "$trace" \
        --arg instance "$INSTANCE_ID" \
        --arg host "$host" \
        --arg medium "$MEDIUM" \
        --arg key "$key" \
        --arg spec_name "$spec_name" '
        {
            trace_id: $trace,
            instance_id: $instance,
            host_ip_port: $host,
            storage_type: "ST_EVENT_REPORT",
            events: [{
                event_type: "EVENT_BLOCK_DELETE",
                block_delete: {
                    block_key: $key,
                    medium: $medium,
                    spec_names: [$spec_name]
                }
            }]
        }'
}

build_query_payload() {
    local key_base="$1"
    local count="$2"
    local medium="$3"
    local output="$4"
    local trace="$5"
    local query_type="${6:-QT_PREFIX_MATCH}"
    jq -nc \
        --arg trace "$trace" \
        --arg instance "$INSTANCE_ID" \
        --arg query_type "$query_type" \
        --arg medium "$medium" \
        --argjson key_base "$key_base" \
        --argjson count "$count" '
        {
            trace_id: $trace,
            instance_id: $instance,
            query_type: $query_type,
            block_cache_keys: [range(0; $count) | $key_base + .],
            medium: (if $medium == "" then [] else [$medium] end)
        }' >"$output"
}

query_custom_keys() {
    local keys_json="$1"
    local medium="$2"
    local output="$3"
    local trace="$4"
    local query_type="${5:-QT_PREFIX_MATCH}"
    local payload="${output}.payload.json"
    jq -nc \
        --arg trace "$trace" \
        --arg instance "$INSTANCE_ID" \
        --arg query_type "$query_type" \
        --arg medium "$medium" \
        --argjson keys "$keys_json" '
        {
            trace_id: $trace,
            instance_id: $instance,
            query_type: $query_type,
            block_cache_keys: $keys,
            medium: (if $medium == "" then [] else [$medium] end)
        }' >"$payload"
    post_and_check /api/getHostCacheState "$payload" "$output" "$trace"
}

prefix_for_host() {
    local response_file="$1"
    local host="$2"
    jq -r --arg host "$host" \
        '[.hosts[]? | select(.host_ip_port == $host) | (.prefix_match_blocks | tonumber)][0] // 0' \
        "$response_file"
}

assert_host_prefix() {
    local response_file="$1"
    local host="$2"
    local expected="$3"
    local context="$4"
    local actual
    actual="$(prefix_for_host "$response_file" "$host")"
    if [[ "$actual" != "$expected" ]]; then
        fail "$context: host=$host expected=$expected actual=$actual response=$(jq -c . "$response_file")"
    fi
}

assert_all_base_hosts_full() {
    local response_file="$1"
    local context="$2"
    local index host
    for ((index = 0; index < HOST_COUNT; ++index)); do
        host="$(host_for_index "$index")"
        assert_host_prefix "$response_file" "$host" "$BLOCK_COUNT" "$context"
    done
}

register_instance() {
    local payload="$WORK_DIR/register.json"
    local response="$WORK_DIR/register.response.json"
    build_register_payload >"$payload"
    post_json "${BASE_URL}/api/registerInstance" "$payload" "$response"
    assert_ok_response "$response" registerInstance
    REGISTERED=1
    pass "registered isolated instance $INSTANCE_ID in group $INSTANCE_GROUP"
}

remove_instance() {
    local payload="$WORK_DIR/remove.json"
    local response="$WORK_DIR/remove.response.json"
    jq -nc \
        --arg trace "ghcs_remove_${INSTANCE_ID}" \
        --arg group "$INSTANCE_GROUP" \
        --arg instance "$INSTANCE_ID" \
        '{trace_id: $trace, instance_group: $group, instance_id: $instance}' >"$payload"
    if post_json "${ADMIN_URL}/api/removeInstance" "$payload" "$response" && \
        assert_ok_response "$response" removeInstance; then
        REGISTERED=0
        pass "removed isolated instance $INSTANCE_ID"
    else
        printf '[WARN] failed to remove synthetic instance %s; artifacts=%s\n' "$INSTANCE_ID" "$WORK_DIR" >&2
    fi
}

bootstrap_test_group() {
    local payload response
    payload="$WORK_DIR/bootstrap_dummy.json"
    response="$WORK_DIR/bootstrap_dummy.response.json"
    jq -nc \
        --arg trace "ghcs_add_dummy_$$" \
        --arg storage "$DUMMY_STORAGE_NAME" \
        --arg root "/tmp/${DUMMY_STORAGE_NAME}" '
        {
            trace_id: $trace,
            storage: {
                global_unique_name: $storage,
                dummy: {root_path: $root, key_count_per_file: 8},
                check_storage_available_when_open: false
            }
        }' >"$payload"
    post_json "${ADMIN_URL}/api/addStorage" "$payload" "$response"
    assert_ok_response "$response" bootstrap-dummy-storage

    payload="$WORK_DIR/bootstrap_event.json"
    response="$WORK_DIR/bootstrap_event.response.json"
    jq -nc \
        --arg trace "ghcs_add_event_$$" \
        --arg storage "$EVENT_STORAGE_NAME" \
        --argjson heartbeat_timeout_ms "$EVENT_HEARTBEAT_TIMEOUT_MS" \
        --argjson cleanup_grace_ms "$EVENT_CLEANUP_GRACE_MS" \
        --argjson liveness_check_interval_ms "$EVENT_LIVENESS_CHECK_INTERVAL_MS" '
        {
            trace_id: $trace,
            storage: {
                global_unique_name: $storage,
                event_report: {
                    heartbeat_timeout_ms: $heartbeat_timeout_ms,
                    cleanup_grace_ms: $cleanup_grace_ms,
                    liveness_check_interval_ms: $liveness_check_interval_ms
                },
                check_storage_available_when_open: false
            }
        }' >"$payload"
    post_json "${ADMIN_URL}/api/addStorage" "$payload" "$response"
    assert_ok_response "$response" bootstrap-event-storage

    payload="$WORK_DIR/bootstrap_group.json"
    response="$WORK_DIR/bootstrap_group.response.json"
    jq -nc \
        --arg trace "ghcs_create_group_$$" \
        --arg group "$INSTANCE_GROUP" \
        --arg dummy "$DUMMY_STORAGE_NAME" \
        --arg event "$EVENT_STORAGE_NAME" '
        {
            trace_id: $trace,
            instance_group: {
                name: $group,
                storage_candidates: [$dummy],
                event_report_storage_candidates: [$event],
                global_quota_group_name: "default_quota_group",
                max_instance_count: 10,
                quota: {
                    capacity: 10737418240,
                    quota_config: [{storage_type: "ST_DUMMY", capacity: 10737418240}]
                },
                cache_config: {
                    reclaim_strategy: {
                        storage_unique_name: $dummy,
                        reclaim_policy: "POLICY_LRU",
                        trigger_strategy: {used_size: 1073741824, used_percentage: 0.8},
                        trigger_period_seconds: 60,
                        reclaim_step_size: 1073741824,
                        reclaim_step_percentage: 10
                    },
                    data_storage_strategy: "CPS_PREFER_3FS",
                    meta_indexer_config: {
                        max_key_count: 10000000,
                        mutex_shard_num: 64,
                        batch_key_size: 128,
                        meta_storage_backend_config: {storage_type: "local", storage_uri: ""},
                        meta_cache_policy_config: {type: "LRU", capacity: 100000}
                    }
                },
                version: 1
            }
        }' >"$payload"
    post_json "${ADMIN_URL}/api/createInstanceGroup" "$payload" "$response"
    assert_ok_response "$response" bootstrap-instance-group
    BOOTSTRAPPED=1
    pass "created isolated instance group $INSTANCE_GROUP"
}

remove_bootstrap_resources() {
    local payload response storage tag
    payload="$WORK_DIR/remove_group.json"
    response="$WORK_DIR/remove_group.response.json"
    jq -nc --arg trace "ghcs_remove_group_$$" --arg name "$INSTANCE_GROUP" \
        '{trace_id: $trace, name: $name}' >"$payload"
    if ! post_json "${ADMIN_URL}/api/removeInstanceGroup" "$payload" "$response" || \
        ! assert_ok_response "$response" remove-instance-group; then
        printf '[WARN] failed to remove test instance group %s\n' "$INSTANCE_GROUP" >&2
        return 1
    fi
    for storage in "$EVENT_STORAGE_NAME" "$DUMMY_STORAGE_NAME"; do
        tag="$(printf '%s' "$storage" | tr -c 'A-Za-z0-9_' '_')"
        payload="$WORK_DIR/remove_storage_${tag}.json"
        response="$WORK_DIR/remove_storage_${tag}.response.json"
        jq -nc --arg trace "ghcs_remove_storage_${tag}_$$" --arg storage "$storage" \
            '{trace_id: $trace, storage_unique_name: $storage}' >"$payload"
        if ! post_json "${ADMIN_URL}/api/removeStorage" "$payload" "$response" || \
            ! assert_ok_response "$response" "remove-storage-$storage"; then
            printf '[WARN] failed to remove test storage %s\n' "$storage" >&2
            return 1
        fi
    done
    BOOTSTRAPPED=0
    pass "removed isolated test group and storages"
}

cleanup() {
    local exit_code=$?
    if [[ "$REGISTERED" == 1 && "$CLEANUP" == 1 ]]; then
        remove_instance || true
    fi
    if [[ "$BOOTSTRAPPED" == 1 && "$CLEANUP" == 1 && "$REGISTERED" == 0 ]]; then
        remove_bootstrap_resources || true
    fi
    if [[ "$KEEP_ARTIFACTS" == 0 && "$REGISTERED" == 0 && "$BOOTSTRAPPED" == 0 ]]; then
        rm -rf "$WORK_DIR"
    else
        log "artifacts kept at $WORK_DIR"
    fi
    exit "$exit_code"
}

register_hosts() {
    local index host payload response
    for ((index = 0; index < HOST_COUNT; ++index)); do
        host="$(host_for_index "$index")"
        payload="$WORK_DIR/node_${index}.json"
        response="$WORK_DIR/node_${index}.response.json"
        build_node_register_payload "$host" "ghcs_node_${index}_$$" >"$payload"
        post_and_check /api/reportEvent "$payload" "$response" "register host $host"
    done
    pass "registered $HOST_COUNT synthetic hosts"
}

prepare_batch_payloads() {
    local output_dir="$1"
    local event_type="$2"
    local key_base="$3"
    local block_count="$4"
    local medium="$5"
    local host_start="$6"
    local host_end="$7"
    local index start end host payload
    mkdir -p "$output_dir"
    for ((index = host_start; index < host_end; ++index)); do
        host="$(host_for_index "$index")"
        for ((start = 0; start < block_count; start += WRITE_BATCH_SIZE)); do
            end=$((start + WRITE_BATCH_SIZE))
            ((end > block_count)) && end="$block_count"
            payload="$output_dir/h${index}_${start}.json"
            build_report_payload "$host" "$event_type" "$medium" "$key_base" "$start" "$end" \
                "ghcs_${event_type}_h${index}_${start}_$$" >"$payload"
        done
    done
}

case_bulk() {
    local payload_dir="$WORK_DIR/bulk_add"
    local query_payload="$WORK_DIR/bulk_query.json"
    local response="$WORK_DIR/bulk_query.response.json"
    prepare_batch_payloads "$payload_dir" add "$KEY_BASE" "$BLOCK_COUNT" "$MEDIUM" 0 "$HOST_COUNT"
    parallel_post_dir /api/reportEvent "$payload_dir" "$WRITE_CONCURRENCY" "concurrent bulk BLOCK_ADD"
    build_query_payload "$KEY_BASE" "$BLOCK_COUNT" "" "$query_payload" "ghcs_bulk_query_$$"
    post_and_check /api/getHostCacheState "$query_payload" "$response" bulk-query
    assert_all_base_hosts_full "$response" bulk-query
    pass "bulk add is fully visible on every host"
}

case_read_stress() {
    local payload="$WORK_DIR/read_stress_query.json"
    local output_dir="$WORK_DIR/read_stress"
    local total=$((READ_CONCURRENCY * READ_ROUNDS))
    local index response host expected actual
    mkdir -p "$output_dir"
    build_query_payload "$KEY_BASE" "$BLOCK_COUNT" "" "$payload" "ghcs_read_stress_$$"
    log "read stress: $total queries, concurrency=$READ_CONCURRENCY"

    export BASE_URL CURL_CONNECT_TIMEOUT CURL_MAX_TIME payload output_dir
    seq 1 "$total" | xargs -n 1 -P "$READ_CONCURRENCY" sh -c '
        index=$1
        response="$output_dir/$index.response.json"
        result=$(curl --silent --show-error \
            --connect-timeout "$CURL_CONNECT_TIMEOUT" \
            --max-time "$CURL_MAX_TIME" \
            --header "Content-Type: application/json" \
            --header "Accept: application/json" \
            --request POST --data-binary "@$payload" \
            --output "$response" --write-out "%{http_code} %{time_total}" \
            "${BASE_URL}/api/getHostCacheState") || exit 20
        printf "%s\n" "$result" >"${response}.meta"
        case "$result" in 2*) ;; *) exit 21 ;; esac
        [ "$(jq -r ".header.status.code // empty" "$response")" = OK ] || exit 22
    ' sh

    for response in "$output_dir"/*.response.json; do
        for ((index = 0; index < HOST_COUNT; ++index)); do
            host="$(host_for_index "$index")"
            actual="$(prefix_for_host "$response" "$host")"
            expected="$BLOCK_COUNT"
            [[ "$actual" == "$expected" ]] || \
                fail "read stress mismatch file=$response host=$host expected=$expected actual=$actual"
        done
    done
    pass "all $total concurrent reads returned complete prefixes"

    jq -Rs 'split("\n") | map(select(length > 0) | split(" ")[1] | tonumber) | sort |
        {count:length,
         p50:.[((length - 1) * 0.50 | floor)],
         p95:.[((length - 1) * 0.95 | floor)],
         p99:.[((length - 1) * 0.99 | floor)],
        max:.[-1]}' "$output_dir"/*.meta | \
        jq -c '{latency_seconds:.}' | sed 's/^/[METRIC] /'
}

single_event() {
    local host="$1"
    local event_type="$2"
    local medium="$3"
    local key="$4"
    local tag="$5"
    local payload="$WORK_DIR/${tag}.json"
    local response="$WORK_DIR/${tag}.response.json"
    build_report_payload "$host" "$event_type" "$medium" "$key" 0 1 "ghcs_${tag}_$$" >"$payload"
    post_and_check /api/reportEvent "$payload" "$response" "$tag"
}

case_prefix() {
    local gaps=()
    local index host gap response="$WORK_DIR/prefix_query.response.json"
    local payload="$WORK_DIR/prefix_query.json"
    gaps+=("$((BLOCK_COUNT / 4))" "$((BLOCK_COUNT / 2))" "$((BLOCK_COUNT * 3 / 4))")
    for ((index = 0; index < 3; ++index)); do
        host="$(host_for_index "$index")"
        gap="${gaps[$index]}"
        single_event "$host" delete "$MEDIUM" "$((KEY_BASE + gap))" "prefix_delete_h${index}"
    done
    build_query_payload "$KEY_BASE" "$BLOCK_COUNT" "" "$payload" "ghcs_prefix_query_$$"
    post_and_check /api/getHostCacheState "$payload" "$response" prefix-query
    for ((index = 0; index < HOST_COUNT; ++index)); do
        host="$(host_for_index "$index")"
        if ((index < 3)); then
            gap="${gaps[$index]}"
        else
            gap="$BLOCK_COUNT"
        fi
        assert_host_prefix "$response" "$host" "$gap" prefix-query
    done
    pass "different per-host delete positions produced exact prefixes"
    for ((index = 0; index < 3; ++index)); do
        host="$(host_for_index "$index")"
        gap="${gaps[$index]}"
        single_event "$host" add "$MEDIUM" "$((KEY_BASE + gap))" "prefix_repair_h${index}"
    done
}

case_idempotent() {
    local host payload response
    host="$(host_for_index 0)"
    payload="$WORK_DIR/idempotent.json"
    response="$WORK_DIR/idempotent.response.json"
    build_report_payload "$host" add "$MEDIUM" "$KEY_BASE" 0 "$BLOCK_COUNT" "ghcs_idempotent_$$" >"$payload"
    post_and_check /api/reportEvent "$payload" "$response" idempotent-add-1
    post_and_check /api/reportEvent "$payload" "${response}.second" idempotent-add-2
    build_query_payload "$KEY_BASE" "$BLOCK_COUNT" "" "$WORK_DIR/idempotent_query.json" "ghcs_idempotent_query_$$"
    post_and_check /api/getHostCacheState "$WORK_DIR/idempotent_query.json" \
        "$WORK_DIR/idempotent_query.response.json" idempotent-query
    assert_all_base_hosts_full "$WORK_DIR/idempotent_query.response.json" idempotent-query
    pass "duplicate BLOCK_ADD remained idempotent"
}

case_medium() {
    local host gap all_response mem_response disk_response
    host="$(host_for_index 0)"
    gap=$((BLOCK_COUNT / 3))
    single_event "$host" add "$SECONDARY_MEDIUM" "$((KEY_BASE + gap))" medium_add_secondary
    single_event "$host" delete "$MEDIUM" "$((KEY_BASE + gap))" medium_delete_primary

    build_query_payload "$KEY_BASE" "$BLOCK_COUNT" "" "$WORK_DIR/medium_all.json" "ghcs_medium_all_$$"
    all_response="$WORK_DIR/medium_all.response.json"
    post_and_check /api/getHostCacheState "$WORK_DIR/medium_all.json" "$all_response" medium-all
    assert_host_prefix "$all_response" "$host" "$BLOCK_COUNT" medium-all

    build_query_payload "$KEY_BASE" "$BLOCK_COUNT" "$MEDIUM" "$WORK_DIR/medium_primary.json" \
        "ghcs_medium_primary_$$"
    mem_response="$WORK_DIR/medium_primary.response.json"
    post_and_check /api/getHostCacheState "$WORK_DIR/medium_primary.json" "$mem_response" medium-primary
    assert_host_prefix "$mem_response" "$host" "$gap" medium-primary

    disk_response="$WORK_DIR/medium_secondary.response.json"
    query_custom_keys "[$((KEY_BASE + gap))]" "$SECONDARY_MEDIUM" "$disk_response" "ghcs_medium_secondary_$$"
    assert_host_prefix "$disk_response" "$host" 1 medium-secondary
    pass "medium union and filtering returned expected prefixes"
    single_event "$host" add "$MEDIUM" "$((KEY_BASE + gap))" medium_repair_primary
}

case_ordering() {
    local host keys response
    host="$(host_for_index 0)"
    keys="$(jq -nc --argjson base "$KEY_BASE" --argjson count "$BLOCK_COUNT" \
        '[range($count - 1; -1; -1) | $base + .]')"
    response="$WORK_DIR/reverse.response.json"
    query_custom_keys "$keys" "" "$response" "ghcs_reverse_$$"
    assert_host_prefix "$response" "$host" "$BLOCK_COUNT" reverse-query

    keys="[$KEY_BASE,$KEY_BASE,$((KEY_BASE + 1)),$((KEY_BASE + 1))]"
    response="$WORK_DIR/duplicate_keys.response.json"
    query_custom_keys "$keys" "" "$response" "ghcs_duplicate_keys_$$"
    assert_host_prefix "$response" "$host" 4 duplicate-key-query
    pass "reversed and duplicate ordered keys were counted consistently"
}

concurrent_reader() {
    local worker="$1"
    local host="$2"
    local payload="$3"
    local result_file="$4"
    local round response prefix previous=0
    : >"$result_file"
    for ((round = 0; round < CONCURRENT_READ_ROUNDS; ++round)); do
        response="$WORK_DIR/concurrent_read_${worker}_${round}.json"
        post_and_check /api/getHostCacheState "$payload" "$response" "concurrent-reader-$worker-$round"
        prefix="$(prefix_for_host "$response" "$host")"
        if ((prefix < previous || prefix > BLOCK_COUNT)); then
            printf '[ERROR] add-only prefix invariant failed worker=%s round=%s previous=%s current=%s\n' \
                "$worker" "$round" "$previous" "$prefix" >&2
            return 1
        fi
        printf '%s\n' "$prefix" >>"$result_file"
        previous="$prefix"
    done
}

case_concurrent() {
    local host_index="$HOST_COUNT"
    local host key_base query_payload writer_dir start end payload response
    local pids=() pid worker result_file
    host="$(host_for_index "$host_index")"
    key_base=$((KEY_BASE + 10000000))
    payload="$WORK_DIR/concurrent_node.json"
    response="$WORK_DIR/concurrent_node.response.json"
    build_node_register_payload "$host" "ghcs_concurrent_node_$$" >"$payload"
    post_and_check /api/reportEvent "$payload" "$response" concurrent-node

    query_payload="$WORK_DIR/concurrent_query.json"
    build_query_payload "$key_base" "$BLOCK_COUNT" "" "$query_payload" "ghcs_concurrent_query_$$"
    log "add-only concurrent case: readers=$READ_CONCURRENCY, rounds=$CONCURRENT_READ_ROUNDS"
    for ((worker = 0; worker < READ_CONCURRENCY; ++worker)); do
        result_file="$WORK_DIR/concurrent_reader_${worker}.prefixes"
        concurrent_reader "$worker" "$host" "$query_payload" "$result_file" &
        pids+=("$!")
    done

    writer_dir="$WORK_DIR/concurrent_writer"
    mkdir -p "$writer_dir"
    for ((start = 0; start < BLOCK_COUNT; start += WRITE_BATCH_SIZE)); do
        end=$((start + WRITE_BATCH_SIZE))
        ((end > BLOCK_COUNT)) && end="$BLOCK_COUNT"
        payload="$writer_dir/${start}.json"
        response="$writer_dir/${start}.response.json"
        build_report_payload "$host" add "$MEDIUM" "$key_base" "$start" "$end" \
            "ghcs_concurrent_add_${start}_$$" >"$payload"
        post_and_check /api/reportEvent "$payload" "$response" "concurrent-add-$start"
    done

    for pid in "${pids[@]}"; do
        wait "$pid"
    done
    post_and_check /api/getHostCacheState "$query_payload" "$WORK_DIR/concurrent_final.response.json" concurrent-final
    assert_host_prefix "$WORK_DIR/concurrent_final.response.json" "$host" "$BLOCK_COUNT" concurrent-final
    pass "ADD-only concurrent reads never regressed and final state is complete"
}

case_race() {
    local host_index="$HOST_COUNT"
    local host key_base race_dir query response prefix repair_dir gap
    host="$(host_for_index "$host_index")"
    key_base=$((KEY_BASE + 10000000))
    build_node_register_payload "$host" "ghcs_race_node_$$" >"$WORK_DIR/race_node.json"
    post_and_check /api/reportEvent "$WORK_DIR/race_node.json" "$WORK_DIR/race_node.response.json" race-node
    race_dir="$WORK_DIR/race"
    prepare_batch_payloads "$race_dir/add" add "$key_base" "$BLOCK_COUNT" "$MEDIUM" "$host_index" "$((host_index + 1))"
    prepare_batch_payloads "$race_dir/delete" delete "$key_base" "$BLOCK_COUNT" "$MEDIUM" "$host_index" "$((host_index + 1))"
    mkdir -p "$race_dir/mixed"
    cp "$race_dir/add"/*.json "$race_dir/mixed/"
    for query in "$race_dir/delete"/*.json; do
        cp "$query" "$race_dir/mixed/delete_$(basename "$query")"
    done
    parallel_post_dir /api/reportEvent "$race_dir/mixed" "$WRITE_CONCURRENCY" "same-key ADD/DELETE race"

    query="$WORK_DIR/race_query.json"
    response="$WORK_DIR/race_query.response.json"
    build_query_payload "$key_base" "$BLOCK_COUNT" "" "$query" "ghcs_race_query_$$"
    post_and_check /api/getHostCacheState "$query" "$response" race-query
    prefix="$(prefix_for_host "$response" "$host")"
    ((prefix >= 0 && prefix <= BLOCK_COUNT)) || fail "race prefix out of range: $prefix"

    repair_dir="$WORK_DIR/race_repair"
    prepare_batch_payloads "$repair_dir" add "$key_base" "$BLOCK_COUNT" "$MEDIUM" "$host_index" "$((host_index + 1))"
    parallel_post_dir /api/reportEvent "$repair_dir" "$WRITE_CONCURRENCY" "post-race deterministic repair"
    post_and_check /api/getHostCacheState "$query" "$WORK_DIR/race_repaired.response.json" race-repaired
    assert_host_prefix "$WORK_DIR/race_repaired.response.json" "$host" "$BLOCK_COUNT" race-repaired

    gap=$((BLOCK_COUNT / 2))
    single_event "$host" delete "$MEDIUM" "$((key_base + gap))" race_deterministic_delete
    post_and_check /api/getHostCacheState "$query" "$WORK_DIR/race_deleted.response.json" race-deleted
    assert_host_prefix "$WORK_DIR/race_deleted.response.json" "$host" "$gap" race-deleted
    single_event "$host" add "$MEDIUM" "$((key_base + gap))" race_final_repair
    pass "ADD/DELETE race stayed bounded and deterministic reconciliation succeeded"
}

case_mamba() {
    local host_index=$((HOST_COUNT + 1))
    local host key_base gap start end payload response query actual
    local mamba_only_counted_as_full=0
    local mamba_add_overwrote_full=0
    local mamba_only_broke_full=0
    host="$(host_for_index "$host_index")"
    key_base=$((KEY_BASE + 20000000))
    gap=$((BLOCK_COUNT / 5))

    build_node_register_payload "$host" "ghcs_mamba_node_$$" >"$WORK_DIR/mamba_node.json"
    post_and_check /api/reportEvent "$WORK_DIR/mamba_node.json" "$WORK_DIR/mamba_node.response.json" mamba-node

    # A Mamba-only state must not be considered a full-attention routing hit.
    mkdir -p "$WORK_DIR/mamba_only_add"
    for ((start = 0; start < BLOCK_COUNT; start += WRITE_BATCH_SIZE)); do
        end=$((start + WRITE_BATCH_SIZE))
        ((end > BLOCK_COUNT)) && end="$BLOCK_COUNT"
        payload="$WORK_DIR/mamba_only_add/${start}.json"
        build_single_spec_add_payload "$host" "$key_base" "$start" "$end" "$MAMBA_SPEC_NAME" \
            "ghcs_mamba_only_add_${start}_$$" >"$payload"
    done
    parallel_post_dir /api/reportEvent "$WORK_DIR/mamba_only_add" "$WRITE_CONCURRENCY" \
        "mamba-only BLOCK_ADD"
    query="$WORK_DIR/mamba_query.json"
    build_query_payload "$key_base" "$BLOCK_COUNT" "" "$query" "ghcs_mamba_query_$$" \
        QT_PREFIX_MATCH_WITH_MAMBA
    post_and_check /api/getHostCacheState "$query" "$WORK_DIR/mamba_only.response.json" mamba-only-query
    actual="$(prefix_for_host "$WORK_DIR/mamba_only.response.json" "$host")"
    if [[ "$actual" != 0 ]]; then
        mamba_only_counted_as_full=1
        printf '[BUG] mamba-only state counted as full-attention cache: host=%s expected=0 actual=%s\n' \
            "$host" "$actual" >&2
    fi

    # Report the two components in separate events and in both orders. A
    # component-level upsert must not replace the other component's state.
    mkdir -p "$WORK_DIR/full_only_add"
    for ((start = 0; start < BLOCK_COUNT; start += WRITE_BATCH_SIZE)); do
        end=$((start + WRITE_BATCH_SIZE))
        ((end > BLOCK_COUNT)) && end="$BLOCK_COUNT"
        payload="$WORK_DIR/full_only_add/${start}.json"
        build_single_spec_add_payload "$host" "$key_base" "$start" "$end" "$FULL_SPEC_NAME" \
            "ghcs_full_only_add_${start}_$$" >"$payload"
    done
    parallel_post_dir /api/reportEvent "$WORK_DIR/full_only_add" "$WRITE_CONCURRENCY" \
        "full-attention-only BLOCK_ADD after mamba"
    post_and_check /api/getHostCacheState "$query" "$WORK_DIR/full_after_mamba.response.json" full-after-mamba
    assert_host_prefix "$WORK_DIR/full_after_mamba.response.json" "$host" "$BLOCK_COUNT" full-after-mamba

    parallel_post_dir /api/reportEvent "$WORK_DIR/mamba_only_add" "$WRITE_CONCURRENCY" \
        "mamba-only BLOCK_ADD after full-attention"
    post_and_check /api/getHostCacheState "$query" "$WORK_DIR/full_after_mamba_reupsert.response.json" \
        full-after-mamba-reupsert
    actual="$(prefix_for_host "$WORK_DIR/full_after_mamba_reupsert.response.json" "$host")"
    if [[ "$actual" != "$BLOCK_COUNT" ]]; then
        mamba_add_overwrote_full=1
        printf '[BUG] mamba component upsert replaced full-attention state: host=%s expected=%s actual=%s\n' \
            "$host" "$BLOCK_COUNT" "$actual" >&2
    fi

    # Upsert both components over the Mamba-only locations.
    mkdir -p "$WORK_DIR/mamba_add"
    for ((start = 0; start < BLOCK_COUNT; start += WRITE_BATCH_SIZE)); do
        end=$((start + WRITE_BATCH_SIZE))
        ((end > BLOCK_COUNT)) && end="$BLOCK_COUNT"
        payload="$WORK_DIR/mamba_add/${start}.json"
        build_hybrid_add_payload "$host" "$key_base" "$start" "$end" "ghcs_mamba_add_${start}_$$" >"$payload"
    done
    parallel_post_dir /api/reportEvent "$WORK_DIR/mamba_add" "$WRITE_CONCURRENCY" \
        "hybrid full-attention + mamba BLOCK_ADD"

    post_and_check /api/getHostCacheState "$query" "$WORK_DIR/mamba_before.response.json" mamba-before
    assert_host_prefix "$WORK_DIR/mamba_before.response.json" "$host" "$BLOCK_COUNT" mamba-before

    payload="$WORK_DIR/mamba_only_delete.json"
    response="$WORK_DIR/mamba_only_delete.response.json"
    build_component_delete_payload "$host" "$((key_base + gap))" "$MAMBA_SPEC_NAME" \
        "ghcs_mamba_only_delete_$$" >"$payload"
    post_and_check /api/reportEvent "$payload" "$response" mamba-only-delete
    post_and_check /api/getHostCacheState "$query" "$WORK_DIR/mamba_after_mamba_delete.response.json" \
        mamba-after-mamba-delete
    actual="$(prefix_for_host "$WORK_DIR/mamba_after_mamba_delete.response.json" "$host")"
    if [[ "$actual" != "$BLOCK_COUNT" ]]; then
        mamba_only_broke_full=1
        printf '[BUG] mamba-only delete removed full-attention state: host=%s gap=%s expected=%s actual=%s\n' \
            "$host" "$gap" "$BLOCK_COUNT" "$actual" >&2
    fi

    # Restore the two-component location, then prove that deleting the full
    # component is the operation that should reduce the routing prefix.
    build_hybrid_add_payload "$host" "$key_base" "$gap" "$((gap + 1))" \
        "ghcs_mamba_repair_$$" >"$WORK_DIR/mamba_repair.json"
    post_and_check /api/reportEvent "$WORK_DIR/mamba_repair.json" "$WORK_DIR/mamba_repair.response.json" mamba-repair
    build_component_delete_payload "$host" "$((key_base + gap))" "$FULL_SPEC_NAME" \
        "ghcs_full_delete_$$" >"$WORK_DIR/full_delete.json"
    post_and_check /api/reportEvent "$WORK_DIR/full_delete.json" "$WORK_DIR/full_delete.response.json" full-delete
    post_and_check /api/getHostCacheState "$query" "$WORK_DIR/mamba_after_full_delete.response.json" \
        mamba-after-full-delete
    assert_host_prefix "$WORK_DIR/mamba_after_full_delete.response.json" "$host" "$gap" mamba-after-full-delete

    if [[ "$mamba_only_counted_as_full" == 1 || "$mamba_add_overwrote_full" == 1 || \
        "$mamba_only_broke_full" == 1 ]]; then
        fail "mamba/full-attention isolation is broken in add, read, and/or delete handling"
    fi
    pass "mamba-only delete preserved full-attention state; full delete reduced the prefix"
}

validate_configuration() {
    local case_name
    local -a selected_cases

    [[ "$HOST_COUNT" =~ ^[0-9]+$ ]] && ((HOST_COUNT >= 3)) || die "HOST_COUNT must be >= 3"
    [[ "$BLOCK_COUNT" =~ ^[0-9]+$ ]] && ((BLOCK_COUNT >= 8)) || die "BLOCK_COUNT must be >= 8"
    [[ "$WRITE_BATCH_SIZE" =~ ^[0-9]+$ ]] && ((WRITE_BATCH_SIZE > 0)) || die "WRITE_BATCH_SIZE must be > 0"
    [[ "$WRITE_CONCURRENCY" =~ ^[0-9]+$ ]] && ((WRITE_CONCURRENCY > 0)) || die "WRITE_CONCURRENCY must be > 0"
    [[ "$READ_CONCURRENCY" =~ ^[0-9]+$ ]] && ((READ_CONCURRENCY > 0)) || die "READ_CONCURRENCY must be > 0"
    [[ "$READ_ROUNDS" =~ ^[0-9]+$ ]] && ((READ_ROUNDS > 0)) || die "READ_ROUNDS must be > 0"
    [[ "$CONCURRENT_READ_ROUNDS" =~ ^[0-9]+$ ]] && \
        ((CONCURRENT_READ_ROUNDS > 0)) || die "CONCURRENT_READ_ROUNDS must be > 0"
    [[ "$KEY_BASE" =~ ^[0-9]+$ ]] || die "KEY_BASE must be a positive int64"
    ((KEY_BASE + 20000000 + BLOCK_COUNT < 9007199254740991)) || \
        die "key range exceeds jq's exact-integer range (2^53 - 1)"
    [[ "$SKIP_REGISTER" != 1 || -n "$EXPLICIT_INSTANCE_ID" ]] || \
        die "SKIP_REGISTER=1 requires an explicitly supplied INSTANCE_ID"
    [[ "$RUN_CASES" == "all" ]] && return
    [[ -n "$RUN_CASES" ]] || die "RUN_CASES must be 'all' or a comma-separated case list"
    IFS=',' read -r -a selected_cases <<<"$RUN_CASES"
    for case_name in "${selected_cases[@]}"; do
        case "$case_name" in
        bulk | read_stress | prefix | idempotent | medium | ordering | concurrent | race | mamba) ;;
        *) die "unknown RUN_CASES entry: $case_name (use --help)" ;;
        esac
    done
}

main() {
    if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
        usage
        exit 0
    fi
    [[ $# -eq 0 ]] || die "unknown argument: $1 (use --help)"

    require_command curl
    require_command jq
    require_command xargs
    require_command awk
    validate_configuration
    WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/kvcm-ghcs-stress.XXXXXX")"
    trap cleanup EXIT

    log "meta=$BASE_URL admin=$ADMIN_URL instance=$INSTANCE_ID cases=$RUN_CASES artifacts=$WORK_DIR"
    if [[ "$BOOTSTRAP_TEST_GROUP" == 1 ]]; then
        check_admin_service
        bootstrap_test_group
    fi
    if [[ "$SKIP_REGISTER" == 0 ]]; then
        register_instance
    else
        log "SKIP_REGISTER=1: using explicitly supplied instance $INSTANCE_ID; cleanup disabled"
        CLEANUP=0
    fi
    register_hosts

    # Every later case assumes the base data set created by bulk.
    case_bulk
    case_enabled idempotent && case_idempotent
    case_enabled read_stress && case_read_stress
    case_enabled prefix && case_prefix
    case_enabled medium && case_medium
    case_enabled ordering && case_ordering
    case_enabled concurrent && case_concurrent
    case_enabled race && case_race
    case_enabled mamba && case_mamba

    log "summary: pass=$PASS_COUNT fail=$FAIL_COUNT"
    ((FAIL_COUNT == 0))
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
