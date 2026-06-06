#!/bin/bash
# =============================================================================
# KVCM Raft Cluster — Comprehensive End-to-End Test
#
# Self-contained: build → deploy → test all APIs → failover → report → cleanup.
# Path-independent. Requires: docker, curl, python3, bash 4+.
#
# Usage:
#   bash test-raft-cluster.sh
#   bash test-raft-cluster.sh --no-build   # skip compile, reuse cached binary
# =============================================================================
set -uo pipefail

# ---------------------------------------------------------------------------
# Phase 0: Environment
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"
BUILD_CONTAINER="th_kvcache"
BUILD_TARGET="//kv_cache_manager:kv_cache_manager_bin"
BINARY_PATH_IN_CONTAINER="bazel-bin/kv_cache_manager/kv_cache_manager_bin"
TMP_BINARY="/tmp/kvcm_test_binary_$$"
VOLUME_NAME="raft-cluster_kvcm-bin"
PROJECT_NAME="raft-cluster"
NO_BUILD=false

for arg in "$@"; do [ "$arg" = "--no-build" ] && NO_BUILD=true; done

ADMIN_PORTS=(6492 6494 6496)
META_PORTS=(6382 6384 6386)
NODE_NAMES=(kvcm-node1 kvcm-node2 kvcm-node3)

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; RESET='\033[0m'

declare -a TEST_NAMES=() TEST_RESULTS=() TEST_TIMES=() TEST_DETAILS=() TEST_GROUPS=()
TESTS_PASSED=0; TESTS_FAILED=0; CURRENT_GROUP=""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log_phase() { echo -e "\n${BOLD}${CYAN}══════════════════════════════════════════════════════════════${RESET}"; echo -e "${BOLD}${CYAN}  $1${RESET}"; echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════════════${RESET}"; }
log_group() { CURRENT_GROUP="$1"; echo -e "\n  ${BOLD}${YELLOW}▶ $1${RESET}"; }
log_info()  { echo -e "    ${DIM}$1${RESET}"; }
log_ok()    { echo -e "    ${GREEN}✓${RESET} $1"; }
log_fail()  { echo -e "    ${RED}✗${RESET} $1"; }

cleanup() {
    echo -e "\n${DIM}Cleaning up containers and volumes...${RESET}"
    cd "$SCRIPT_DIR"
    docker compose -p "$PROJECT_NAME" -f "$COMPOSE_FILE" down -v --remove-orphans 2>/dev/null
    rm -f "$TMP_BINARY"
    docker volume rm "$VOLUME_NAME" 2>/dev/null || true
}
trap cleanup EXIT

now_ms() { echo $(( $(date +%s%N) / 1000000 )); }

wait_for_leader() {
    local exclude_idx="${1:--1}" max_wait="${2:-20}"
    local start_time=$(date +%s%N)
    for _ in $(seq 1 $((max_wait * 5))); do
        sleep 0.2
        for i in 0 1 2; do
            [ "$i" = "$exclude_idx" ] && continue
            local result is_leader
            result=$(curl -s --connect-timeout 1 http://localhost:${ADMIN_PORTS[$i]}/api/checkHealth -d '{}' 2>/dev/null) || continue
            is_leader=$(echo "$result" | python3 -c "import sys,json;print(json.load(sys.stdin).get('is_leader',''))" 2>/dev/null)
            if [ "$is_leader" = "True" ]; then
                local end_time=$(date +%s%N)
                echo "$i $(( (end_time - start_time) / 1000000 ))"
                return 0
            fi
        done
    done
    return 1
}

api_call() { curl -s --connect-timeout 3 "http://localhost:${1}${2}" -d "$3" 2>/dev/null; }

api_code() {
    api_call "$1" "$2" "$3" | python3 -c "import sys,json;print(json.load(sys.stdin)['header']['status']['code'])" 2>/dev/null
}

record_test() {
    local name="$1" result="$2" elapsed="$3" detail="${4:-}"
    TEST_NAMES+=("$name"); TEST_RESULTS+=("$result"); TEST_TIMES+=("$elapsed")
    TEST_DETAILS+=("$detail"); TEST_GROUPS+=("$CURRENT_GROUP")
    if [ "$result" = "PASS" ]; then
        ((TESTS_PASSED++)); log_ok "$name ${DIM}(${elapsed}ms)${RESET}"
    else
        ((TESTS_FAILED++)); log_fail "$name ${DIM}(${elapsed}ms)${RESET} — $detail"
    fi
}

assert_api() {
    local name="$1" expected="$2" port="$3" path="$4" data="$5"
    local t0=$(now_ms)
    local actual=$(api_code "$port" "$path" "$data")
    local t1=$(now_ms)
    if [ "$actual" = "$expected" ]; then
        record_test "$name" "PASS" "$((t1 - t0))"
    else
        record_test "$name" "FAIL" "$((t1 - t0))" "expected=$expected got=$actual"
    fi
}

assert_count() {
    local name="$1" expected="$2" port="$3" path="$4" data="$5" jq_expr="$6"
    local t0=$(now_ms)
    local actual=$(api_call "$port" "$path" "$data" | python3 -c "$jq_expr" 2>/dev/null)
    local t1=$(now_ms)
    if [ "$actual" = "$expected" ]; then
        record_test "$name" "PASS" "$((t1 - t0))"
    else
        record_test "$name" "FAIL" "$((t1 - t0))" "expected=$expected got=$actual"
    fi
}

# ---------------------------------------------------------------------------
# Phase 1: Build
# ---------------------------------------------------------------------------
log_phase "Phase 1: Build"

if [ "$NO_BUILD" = true ]; then
    log_info "Skipping build (--no-build flag)"
    docker cp "$BUILD_CONTAINER:$(docker exec $BUILD_CONTAINER bash -c \
        'cd /home/sili.th/KVCacheManager/github-opensource && readlink -f bazel-bin/kv_cache_manager/kv_cache_manager_bin' \
        2>/dev/null)" "$TMP_BINARY" 2>/dev/null || true
else
    if [ "$(docker inspect -f '{{.State.Running}}' "$BUILD_CONTAINER" 2>/dev/null)" != "true" ]; then
        echo -e "${RED}ERROR: Container '$BUILD_CONTAINER' not running.${RESET}"; exit 1
    fi
    SRC_IN_CONTAINER="/home/sili.th/KVCacheManager/github-opensource"
    log_info "Building $BUILD_TARGET ..."
    BUILD_T0=$(now_ms)
    if ! docker exec "$BUILD_CONTAINER" bash -c "cd $SRC_IN_CONTAINER && bazelisk build $BUILD_TARGET 2>&1" | tail -3; then
        echo -e "${RED}ERROR: Build failed.${RESET}"; exit 1
    fi
    log_ok "Build complete ($(($(now_ms) - BUILD_T0))ms)"
    docker cp "$BUILD_CONTAINER:$SRC_IN_CONTAINER/$BINARY_PATH_IN_CONTAINER" "$TMP_BINARY"
    chmod +x "$TMP_BINARY"
fi

if [ ! -f "$TMP_BINARY" ]; then
    echo -e "${RED}ERROR: No binary available.${RESET}"; exit 1
fi
log_info "Binary: $(du -h "$TMP_BINARY" | cut -f1)"

# ---------------------------------------------------------------------------
# Phase 2: Deploy Fresh Cluster
# ---------------------------------------------------------------------------
log_phase "Phase 2: Deploy"
cd "$SCRIPT_DIR"

docker compose -p "$PROJECT_NAME" -f "$COMPOSE_FILE" down -v --remove-orphans 2>/dev/null
docker volume rm "$VOLUME_NAME" 2>/dev/null || true

docker volume create "$VOLUME_NAME" >/dev/null
docker run --rm -v "$VOLUME_NAME":/opt/kvcm -v "$TMP_BINARY":/tmp/bin:ro \
    alpine sh -c "cp /tmp/bin /opt/kvcm/kv_cache_manager_bin && chmod +x /opt/kvcm/kv_cache_manager_bin"

log_info "Starting 3-node cluster..."
docker compose -p "$PROJECT_NAME" -f "$COMPOSE_FILE" up -d 2>&1 | grep -v "^$"

log_info "Waiting for leader election and cluster stabilization..."
LEADER_INFO=$(wait_for_leader -1 30) || { echo -e "${RED}ERROR: No leader elected in 30s${RESET}"; exit 1; }
ELECT_MS=$(echo "$LEADER_INFO" | awk '{print $2}')

# Wait for stable leadership: same single leader across 5 consecutive checks
stable_count=0; stable_idx=-1; prev_leader=-1
for _ in $(seq 1 60); do
    sleep 0.5
    leaders=0; current_leader=-1
    for i in 0 1 2; do
        r=$(curl -s --connect-timeout 1 http://localhost:${ADMIN_PORTS[$i]}/api/checkHealth -d '{}' 2>/dev/null) || continue
        l=$(echo "$r" | python3 -c "import sys,json;print(json.load(sys.stdin).get('is_leader',''))" 2>/dev/null)
        [ "$l" = "True" ] && { ((leaders++)); current_leader=$i; }
    done
    if [ "$leaders" -eq 1 ] && [ "$current_leader" = "$prev_leader" ]; then
        ((stable_count++))
    else
        stable_count=0
    fi
    prev_leader=$current_leader
    [ "$stable_count" -ge 5 ] && { stable_idx=$current_leader; break; }
done
[ "$stable_idx" = "-1" ] && { echo -e "${RED}ERROR: No stable leader after 30s${RESET}"; exit 1; }
LEADER_IDX=$stable_idx
log_ok "Leader: ${NODE_NAMES[$LEADER_IDX]} (election: ${ELECT_MS}ms)"

LA=${ADMIN_PORTS[$LEADER_IDX]}
LM=${META_PORTS[$LEADER_IDX]}

# Wait for startup_config to be fully loaded (nfs_01 storage must exist)
log_info "Waiting for leader API readiness (startup_config)..."
for _ in $(seq 1 60); do
    resp=$(api_call "$LA" "/api/listStorage" '{}')
    code=$(echo "$resp" | python3 -c "import sys,json;print(json.load(sys.stdin).get('header',{}).get('status',{}).get('code',''))" 2>/dev/null)
    count=$(echo "$resp" | python3 -c "import sys,json;print(len(json.load(sys.stdin).get('storage',[])))" 2>/dev/null)
    [ "$code" = "OK" ] && [ "${count:-0}" -ge 1 ] && break
    sleep 0.5
done
[ "$code" != "OK" ] || [ "${count:-0}" -lt 1 ] && { echo -e "${RED}ERROR: Leader API not ready (code=$code count=$count)${RESET}"; exit 1; }
log_ok "Leader API ready (storage count=$count)"

# ---------------------------------------------------------------------------
# Phase 3: Tests
# ---------------------------------------------------------------------------
log_phase "Phase 3: Tests"

# ===== Cluster Health =====
log_group "Cluster Health"
t0=$(now_ms)
healthy=0; leaders=0
for i in 0 1 2; do
    r=$(curl -s --connect-timeout 2 http://localhost:${ADMIN_PORTS[$i]}/api/checkHealth -d '{}' 2>/dev/null)
    h=$(echo "$r" | python3 -c "import sys,json;print(json.load(sys.stdin).get('is_health',''))" 2>/dev/null)
    l=$(echo "$r" | python3 -c "import sys,json;print(json.load(sys.stdin).get('is_leader',''))" 2>/dev/null)
    [ "$h" = "True" ] && ((healthy++))
    [ "$l" = "True" ] && ((leaders++))
done
t1=$(now_ms)
[ "$healthy" -eq 3 ] && [ "$leaders" -eq 1 ] \
    && record_test "3 nodes healthy, 1 leader" "PASS" "$((t1-t0))" \
    || record_test "3 nodes healthy, 1 leader" "FAIL" "$((t1-t0))" "healthy=$healthy leaders=$leaders"

# ===== Storage CRUD =====
log_group "Storage CRUD"

# nfs_01 already created by startup_config, verify it exists
assert_count "nfs_01 from startup_config exists" "1" "$LA" "/api/listStorage" '{}' \
    "import sys,json;print(len([s for s in json.load(sys.stdin).get('storage',[]) if s['global_unique_name']=='nfs_01']))"

# Add a second storage for CRUD testing
assert_api "addStorage(test_nfs)" "OK" "$LA" "/api/addStorage" \
    '{"storage":{"global_unique_name":"test_nfs","nfs":{"root_path":"/data/nfs_test/","key_count_per_file":16}}}'
assert_count "listStorage = 2 (nfs_01 + test_nfs)" "2" "$LA" "/api/listStorage" '{}' \
    "import sys,json;print(len(json.load(sys.stdin).get('storage',[])))"
assert_api "disableStorage(test_nfs)" "OK" "$LA" "/api/disableStorage" \
    '{"storage_unique_name":"test_nfs"}'
assert_api "enableStorage(test_nfs)" "OK" "$LA" "/api/enableStorage" \
    '{"storage_unique_name":"test_nfs"}'
assert_api "removeStorage(test_nfs)" "OK" "$LA" "/api/removeStorage" \
    '{"storage_unique_name":"test_nfs"}'
assert_count "after remove, listStorage = 1" "1" "$LA" "/api/listStorage" '{}' \
    "import sys,json;print(len(json.load(sys.stdin).get('storage',[])))"

# ===== Instance Group CRUD =====
log_group "Instance Group CRUD"

# "default" already created by startup_config, verify it
assert_api "getInstanceGroup(default) from startup" "OK" "$LA" "/api/getInstanceGroup" '{"name":"default"}'

# Create a test group for CRUD
assert_api "createInstanceGroup(test_group)" "OK" "$LA" "/api/createInstanceGroup" \
    '{"instance_group":{"name":"test_group","storage_candidates":["nfs_01"],"global_quota_group_name":"test_quota","max_instance_count":50,"quota":{"capacity":5000000000},"cache_config":{"reclaim_strategy":{"storage_unique_name":"nfs_01","reclaim_policy":1,"trigger_strategy":{"used_percentage":0.9},"delay_before_delete_ms":2000},"data_storage_strategy":2,"meta_indexer_config":{"max_key_count":500000,"mutex_shard_num":8,"batch_key_size":16,"meta_storage_backend_config":{"storage_type":"local"},"meta_cache_policy_config":{"type":"LRU","capacity":5000}}}}}'
assert_count "listInstanceGroup = 2 (default + test)" "2" "$LA" "/api/listInstanceGroup" '{}' \
    "import sys,json;print(len(json.load(sys.stdin).get('instance_group',[])))"
assert_api "updateInstanceGroup(test_group v0→v1)" "OK" "$LA" "/api/updateInstanceGroup" \
    '{"instance_group":{"name":"test_group","storage_candidates":["nfs_01"],"global_quota_group_name":"test_quota","max_instance_count":200,"quota":{"capacity":8000000000},"cache_config":{"reclaim_strategy":{"storage_unique_name":"nfs_01","reclaim_policy":1,"trigger_strategy":{"used_percentage":0.9},"delay_before_delete_ms":2000},"data_storage_strategy":2,"meta_indexer_config":{"max_key_count":500000,"mutex_shard_num":8,"batch_key_size":16,"meta_storage_backend_config":{"storage_type":"local"},"meta_cache_policy_config":{"type":"LRU","capacity":5000}}},"version":1},"current_version":0}'
assert_api "removeInstanceGroup(test_group)" "OK" "$LA" "/api/removeInstanceGroup" '{"name":"test_group"}'
assert_count "after remove, listInstanceGroup = 1" "1" "$LA" "/api/listInstanceGroup" '{}' \
    "import sys,json;print(len(json.load(sys.stdin).get('instance_group',[])))"

# ===== Instance CRUD =====
log_group "Instance CRUD"

assert_api "registerInstance(inst-1) via meta port" "OK" "$LM" "/api/registerInstance" \
    '{"instance_group":"default","instance_id":"inst-1","block_size":16,"location_spec_infos":[{"name":"TP0","size":1024}],"model_deployment":{"model_name":"llama-7b","dtype":"FP16","tp_size":1,"dp_size":1}}'
assert_api "registerInstance(inst-2) via meta port" "OK" "$LM" "/api/registerInstance" \
    '{"instance_group":"default","instance_id":"inst-2","block_size":32,"location_spec_infos":[{"name":"TP0","size":2048},{"name":"TP1","size":2048}],"model_deployment":{"model_name":"llama-13b","dtype":"BF16","tp_size":2,"dp_size":1}}'
assert_api "getInstanceInfo(inst-1)" "OK" "$LM" "/api/getInstanceInfo" '{"instance_id":"inst-1"}'
assert_api "getInstanceInfo(inst-2)" "OK" "$LM" "/api/getInstanceInfo" '{"instance_id":"inst-2"}'
assert_count "listInstanceInfo(default) = 2" "2" "$LA" "/api/listInstanceInfo" '{"instance_group_name":"default"}' \
    "import sys,json;print(len(json.load(sys.stdin).get('instance_info',[])))"
assert_api "removeInstance(inst-2)" "OK" "$LA" "/api/removeInstance" \
    '{"instance_group":"default","instance_id":"inst-2"}'
# Verify inst-2 is gone (should return error)
t0=$(now_ms)
s=$(api_code "$LM" "/api/getInstanceInfo" '{"instance_id":"inst-2"}')
t1=$(now_ms)
[ "$s" != "OK" ] \
    && record_test "inst-2 removed (not found)" "PASS" "$((t1-t0))" \
    || record_test "inst-2 removed (not found)" "FAIL" "$((t1-t0))" "still exists"

# Re-register inst-2 for failover verification later
api_call "$LM" "/api/registerInstance" \
    '{"instance_group":"default","instance_id":"inst-2","block_size":32,"location_spec_infos":[{"name":"TP0","size":2048},{"name":"TP1","size":2048}],"model_deployment":{"model_name":"llama-13b","dtype":"BF16","tp_size":2,"dp_size":1}}' >/dev/null

# ===== Account CRUD =====
log_group "Account CRUD"

assert_api "addAccount(admin, ROLE_ADMIN)" "OK" "$LA" "/api/addAccount" \
    '{"user_name":"admin","password":"secret123","role":1}'
assert_api "addAccount(reader, ROLE_USER)" "OK" "$LA" "/api/addAccount" \
    '{"user_name":"reader","password":"read456","role":0}'
assert_count "listAccount = 2" "2" "$LA" "/api/listAccount" '{}' \
    "import sys,json;print(len(json.load(sys.stdin).get('accounts',[])))"
assert_api "deleteAccount(reader)" "OK" "$LA" "/api/deleteAccount" '{"user_name":"reader"}'
assert_count "after delete, listAccount = 1" "1" "$LA" "/api/listAccount" '{}' \
    "import sys,json;print(len(json.load(sys.stdin).get('accounts',[])))"

# ===== Cache Operations (Meta Service) =====
log_group "Cache Operations"

# startWriteCache allocates write locations
t0=$(now_ms)
resp=$(api_call "$LM" "/api/startWriteCache" '{"instance_id":"inst-1","block_keys":[1001,1002,1003]}')
code=$(echo "$resp" | python3 -c "import sys,json;print(json.load(sys.stdin)['header']['status']['code'])" 2>/dev/null)
session=$(echo "$resp" | python3 -c "import sys,json;print(json.load(sys.stdin).get('write_session_id',''))" 2>/dev/null)
t1=$(now_ms)
[ "$code" = "OK" ] \
    && record_test "startWriteCache(3 keys)" "PASS" "$((t1-t0))" \
    || record_test "startWriteCache(3 keys)" "FAIL" "$((t1-t0))" "code=$code"

# finishWriteCache (mark all blocks successful)
if [ -n "$session" ] && [ "$code" = "OK" ]; then
    assert_api "finishWriteCache(session)" "OK" "$LM" "/api/finishWriteCache" \
        "{\"instance_id\":\"inst-1\",\"write_session_id\":\"$session\",\"success_blocks\":{\"offset\":0}}"
fi

# getCacheMeta
assert_api "getCacheMeta(inst-1, keys 1001-1002)" "OK" "$LM" "/api/getCacheMeta" \
    '{"instance_id":"inst-1","block_keys":[1001,1002]}'

# getCacheLocation
assert_api "getCacheLocation(inst-1, key 1001)" "OK" "$LM" "/api/getCacheLocation" \
    '{"instance_id":"inst-1","block_keys":[1001],"query_type":1}'

# removeCache
assert_api "removeCache(inst-1, key 1003)" "OK" "$LM" "/api/removeCache" \
    '{"instance_id":"inst-1","block_keys":[1003]}'

# ===== First Failover =====
log_group "First Failover"

log_info "Killing leader ${NODE_NAMES[$LEADER_IDX]}..."
docker compose -p "$PROJECT_NAME" -f "$COMPOSE_FILE" stop "${NODE_NAMES[$LEADER_IDX]}" 2>/dev/null

NEW_LEADER_INFO=$(wait_for_leader "$LEADER_IDX" 15) || NEW_LEADER_INFO=""
if [ -n "$NEW_LEADER_INFO" ]; then
    NEW_IDX=$(echo "$NEW_LEADER_INFO" | awk '{print $1}')
    FO_MS=$(echo "$NEW_LEADER_INFO" | awk '{print $2}')
    record_test "New leader elected" "PASS" "$FO_MS" "${NODE_NAMES[$NEW_IDX]}"
    NA=${ADMIN_PORTS[$NEW_IDX]}; NM=${META_PORTS[$NEW_IDX]}
else
    record_test "New leader elected" "FAIL" "15000" "timeout"
    NA=""; NM=""; NEW_IDX=-1
fi

# ===== Post-Failover Verification =====
log_group "Post-Failover: Data Integrity"

if [ -n "$NA" ]; then
    # Wait for new leader's API to be ready (storage data must be visible)
    for _ in $(seq 1 40); do
        resp=$(api_call "$NA" "/api/listStorage" '{}')
        code=$(echo "$resp" | python3 -c "import sys,json;print(json.load(sys.stdin).get('header',{}).get('status',{}).get('code',''))" 2>/dev/null)
        cnt=$(echo "$resp" | python3 -c "import sys,json;print(len(json.load(sys.stdin).get('storage',[])))" 2>/dev/null)
        [ "$code" = "OK" ] && [ "${cnt:-0}" -ge 1 ] && break
        sleep 0.5
    done

    assert_count "storage preserved (nfs_01)" "1" "$NA" "/api/listStorage" '{}' \
        "import sys,json;print(len(json.load(sys.stdin).get('storage',[])))"
    assert_count "instance group preserved (default)" "1" "$NA" "/api/listInstanceGroup" '{}' \
        "import sys,json;print(len(json.load(sys.stdin).get('instance_group',[])))"
    assert_api "getInstanceInfo(inst-1) on new leader" "OK" "$NM" "/api/getInstanceInfo" \
        '{"instance_id":"inst-1"}'
    assert_api "getInstanceInfo(inst-2) on new leader" "OK" "$NM" "/api/getInstanceInfo" \
        '{"instance_id":"inst-2"}'
    assert_count "account preserved (admin)" "1" "$NA" "/api/listAccount" '{}' \
        "import sys,json;print(len(json.load(sys.stdin).get('accounts',[])))"

    # Write still works on new leader
    assert_api "write on new leader (addAccount)" "OK" "$NA" "/api/addAccount" \
        '{"user_name":"post-fo-user","password":"pw","role":0}'
    assert_api "cache read on new leader" "OK" "$NM" "/api/getCacheMeta" \
        '{"instance_id":"inst-1","block_keys":[1001]}'
else
    for desc in "storage preserved" "group preserved" "inst-1 on new" "inst-2 on new" "account preserved" "write new leader" "cache read new leader"; do
        record_test "$desc" "FAIL" "0" "no leader"
    done
fi

# ===== Node Rejoin =====
log_group "Node Rejoin"

log_info "Restarting ${NODE_NAMES[$LEADER_IDX]}..."
docker compose -p "$PROJECT_NAME" -f "$COMPOSE_FILE" start "${NODE_NAMES[$LEADER_IDX]}" 2>/dev/null
sleep 3
t0=$(now_ms)
rh=$(curl -s --connect-timeout 2 http://localhost:${ADMIN_PORTS[$LEADER_IDX]}/api/checkHealth -d '{}' \
    | python3 -c "import sys,json;print(json.load(sys.stdin).get('is_health',''))" 2>/dev/null)
t1=$(now_ms)
[ "$rh" = "True" ] \
    && record_test "Killed node rejoins healthy" "PASS" "$((t1-t0))" \
    || record_test "Killed node rejoins healthy" "FAIL" "$((t1-t0))" "health=$rh"

# ===== Second Failover =====
log_group "Second Failover (commit callback验证)"

if [ "${NEW_IDX:--1}" != "-1" ]; then
    log_info "Killing second leader ${NODE_NAMES[$NEW_IDX]}..."
    docker compose -p "$PROJECT_NAME" -f "$COMPOSE_FILE" stop "${NODE_NAMES[$NEW_IDX]}" 2>/dev/null

    THIRD_INFO=$(wait_for_leader "$NEW_IDX" 15) || THIRD_INFO=""
    if [ -n "$THIRD_INFO" ]; then
        THIRD_IDX=$(echo "$THIRD_INFO" | awk '{print $1}')
        THIRD_MS=$(echo "$THIRD_INFO" | awk '{print $2}')
        TA=${ADMIN_PORTS[$THIRD_IDX]}; TM=${META_PORTS[$THIRD_IDX]}
        record_test "Third leader elected" "PASS" "$THIRD_MS" "${NODE_NAMES[$THIRD_IDX]}"
        # Wait for 3rd leader API readiness (storage data must be visible)
        for _ in $(seq 1 40); do
            resp=$(api_call "$TA" "/api/listStorage" '{}')
            code=$(echo "$resp" | python3 -c "import sys,json;print(json.load(sys.stdin).get('header',{}).get('status',{}).get('code',''))" 2>/dev/null)
            cnt=$(echo "$resp" | python3 -c "import sys,json;print(len(json.load(sys.stdin).get('storage',[])))" 2>/dev/null)
            [ "$code" = "OK" ] && [ "${cnt:-0}" -ge 1 ] && break
            sleep 0.5
        done

        assert_api "inst-1 on 3rd leader" "OK" "$TM" "/api/getInstanceInfo" '{"instance_id":"inst-1"}'
        assert_api "inst-2 on 3rd leader" "OK" "$TM" "/api/getInstanceInfo" '{"instance_id":"inst-2"}'
        assert_count "all accounts on 3rd leader" "2" "$TA" "/api/listAccount" '{}' \
            "import sys,json;print(len(json.load(sys.stdin).get('accounts',[])))"
        assert_count "storage on 3rd leader" "1" "$TA" "/api/listStorage" '{}' \
            "import sys,json;print(len(json.load(sys.stdin).get('storage',[])))"
        assert_api "cache meta on 3rd leader" "OK" "$TM" "/api/getCacheMeta" \
            '{"instance_id":"inst-1","block_keys":[1001]}'
    else
        record_test "Third leader elected" "FAIL" "15000" "timeout"
    fi
else
    record_test "Third leader elected" "FAIL" "0" "skipped (no 2nd leader)"
fi

# ---------------------------------------------------------------------------
# Phase 4: Report
# ---------------------------------------------------------------------------
log_phase "Phase 4: Report"
echo ""
TOTAL=$((TESTS_PASSED + TESTS_FAILED))
printf "  ${BOLD}%-60s %-6s %7s${RESET}\n" "TEST" "RESULT" "TIME"
printf "  %s %s %s\n" "$(printf '%0.s─' {1..60})" "──────" "───────"

PREV_GRP=""
for i in "${!TEST_NAMES[@]}"; do
    if [ "${TEST_GROUPS[$i]}" != "$PREV_GRP" ]; then
        PREV_GRP="${TEST_GROUPS[$i]}"
        printf "\n  ${YELLOW}%s${RESET}\n" "$PREV_GRP"
    fi
    [ "${TEST_RESULTS[$i]}" = "PASS" ] && st="${GREEN}PASS${RESET}" || st="${RED}FAIL${RESET}"
    printf "    %-56s ${st}  %5sms\n" "${TEST_NAMES[$i]}" "${TEST_TIMES[$i]}"
    if [ -n "${TEST_DETAILS[$i]}" ] && [ "${TEST_RESULTS[$i]}" = "FAIL" ]; then
        printf "      ${DIM}↳ %s${RESET}\n" "${TEST_DETAILS[$i]}"
    fi
done

echo ""
printf "  %s\n" "$(printf '%0.s─' {1..75})"
if [ "$TESTS_FAILED" -eq 0 ]; then
    printf "  ${GREEN}${BOLD}ALL %d TESTS PASSED${RESET}\n\n" "$TOTAL"
else
    printf "  ${RED}${BOLD}%d/%d FAILED${RESET} | ${GREEN}%d passed${RESET}\n\n" "$TESTS_FAILED" "$TOTAL" "$TESTS_PASSED"
fi

# ---------------------------------------------------------------------------
# Phase 5: Cleanup (trap)
# ---------------------------------------------------------------------------
log_phase "Phase 5: Cleanup"
[ "$TESTS_FAILED" -eq 0 ]
