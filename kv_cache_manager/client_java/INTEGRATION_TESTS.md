# Java MetaClient Integration Tests

## Overview

Integration tests for the Java MetaClient that verify correct behavior against a real KVCM server. These tests cover the 3 CacheAware RPCs that were not covered by the existing Python integration tests:

- **GetCacheLocationsByBackend** - Query cache locations by backend type
- **GetCacheLocationLen** - Query cache hit count
- **GetCacheMeta** - Query cache metadata with status

## Prerequisites

1. **KVCM Server Binary**: Must be built and available at a known path
   ```bash
   # Build the server binary (if not already built)
   cd /path/to/tair-kvcache
   bazelisk build //kv_cache_manager:kv_cache_manager_bin
   
   # Binary will be at: bazel-bin/kv_cache_manager/kv_cache_manager_bin
   ```

2. **Set Environment Variable**: Point to the server binary
   ```bash
   export KVCM_BIN=/path/to/tair-kvcache/bazel-bin/kv_cache_manager/kv_cache_manager_bin
   ```

## Running Tests

### Run Integration Tests Only
```bash
cd kv_cache_manager/client_java
./gradlew integrationTest
```

### Run Unit Tests Only
```bash
./gradlew test
```

### Run All Tests (Unit + Integration)
```bash
./gradlew check
```

## Test Structure

### Test Classes

1. **CacheAwareTestBase** (abstract base class)
   - Shared test logic for all CacheAware RPCs
   - Setup helpers: `registerInstance`, `startWriteCache`, `finishWriteCache`
   - 15 shared test methods covering 3 RPCs

2. **CacheAwareGrpcTest** (15 tests)
   - Tests all 3 CacheAware RPCs via gRPC
   - Covers normal, error, and boundary scenarios
   - Includes `GetCacheLocationLen` tests

3. **CacheAwareHttpTest** (15 tests)
   - Tests all 3 CacheAware RPCs via HTTP
   - Covers the same scenarios as the gRPC tests

### Test Coverage

#### GetCacheLocationsByBackend (6 scenarios)
- Basic query with NFS backend selector
- Partial key match (some keys exist, some don't)
- Instance does not exist (error handling)
- Empty backend selectors (error handling)
- Location spec names filtering
- Token IDs to block keys conversion

#### GetCacheLocationLen (5 scenarios)
- QT_PREFIX_MATCH with prefix break
- QT_PREFIX_MATCH with no matches
- QT_BATCH_GET with existing keys
- QT_BATCH_GET with no matches
- Consistency with GetCacheLocation

#### GetCacheMeta (4 scenarios)
- CLS_SERVING status after write
- CLS_NOT_FOUND status for missing keys
- Instance does not exist (error handling)
- JSON structure validation

## Test Infrastructure

### Server Lifecycle

Each test class:
1. Starts a KVCM server instance in `@BeforeAll`
2. Allocates 4 random ports (rpc, http, admin_rpc, admin_http)
3. Performs health check via `getClusterInfo` polling
4. Runs all tests with isolated instance IDs
5. Stops server and cleans up in `@AfterAll`

### Test Isolation

- Each test method gets a unique `instanceId` (format: `test_<methodName>_<timestamp>`)
- Tests can run in parallel (different ports)
- No shared state between tests

## Known Limitations

1. **Token IDs conversion behavior**
   - The exact conversion logic (token_id → block_key) needs verification
   - Tests verify the query doesn't throw exceptions
   - TODO: Document expected conversion behavior

## Troubleshooting

### Server Fails to Start

Check the server logs:
```bash
# Logs are in the test work directory
cat /tmp/kvcm-integration-test-*/stderr.log
cat /tmp/kvcm-integration-test-*/stdout.log
```

### Port Conflicts

Tests use `ServerSocket(0)` to allocate random ports. If you see port conflicts:
- Check for zombie server processes: `pgrep -f kv_cache_manager_bin`
- Kill them: `pkill -f kv_cache_manager_bin`

### KVCM_BIN Not Found

```bash
# Verify the binary exists and is executable
ls -lh $KVCM_BIN
file $KVCM_BIN
```

## CI/CD Integration

Add to your CI pipeline:

```yaml
- name: Run Java Integration Tests
  env:
    KVCM_BIN: ${{ github.workspace }}/bazel-bin/kv_cache_manager/kv_cache_manager_bin
  run: |
    cd kv_cache_manager/client_java
    ./gradlew integrationTest
```

## Architecture

```
IntegrationTestBase
    ├── @BeforeAll: start KVCM server
    ├── @AfterAll: stop server & cleanup
    ├── @BeforeEach: generate unique instanceId
    └── Helper methods: registerInstance, startWriteCache, finishWriteCache

CacheAwareTestBase extends IntegrationTestBase
    ├── Abstract method: getClient()
    └── 15 shared test methods (RPC logic)

CacheAwareGrpcTest extends CacheAwareTestBase
    ├── getClient() → grpcClient
    └── 15 test methods (gRPC protocol)

CacheAwareHttpTest extends CacheAwareTestBase
    ├── getClient() → httpClient
    └── 15 test methods (HTTP protocol)

KvcmServerManager
    ├── Port allocation (ServerSocket)
    ├── Server startup (ProcessBuilder + daemon mode)
    ├── Health check (getClusterInfo polling)
    ├── Graceful shutdown (SIGTERM → SIGKILL)
    └── Work directory management
```

## Test Results

Total: **63 tests**
- Unit tests: 33 passed
- Integration tests: 30 passed (15 gRPC + 15 HTTP)

