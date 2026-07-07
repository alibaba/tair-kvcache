package com.alibaba.tair.kvcm.client;

import kv_cache_manager.proto.meta.MetaServiceOuterClass.*;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration tests for CacheAware RPCs using HTTP transport.
 * <p>
 * Covers:
 * - GetCacheLocationsByBackend (6 scenarios)
 * - GetCacheMeta (4 scenarios)
 * <p>
 * Note: GetCacheLocationLen is excluded because it has no HTTP endpoint.
 * See task 5.6 for documentation of this known limitation.
 */
public class CacheAwareHttpTest extends IntegrationTestBase {

    // === GetCacheLocationsByBackend Tests ===

    @Test
    void testGetCacheLocationsByBackend_basicQuery() {
        // Setup: register and write cache for keys [1, 2, 3]
        registerInstance(instanceId);
        String sessionId = startWriteCache(instanceId, 1L, 2L, 3L);
        finishWriteCache(instanceId, sessionId, 3);

        // Query with NFS backend selector
        GetCacheLocationsByBackendRequest request = GetCacheLocationsByBackendRequest.newBuilder()
                .setTraceId("test-trace")
                .setInstanceId(instanceId)
                .setQueryType(QueryType.QT_BATCH_GET)
                .addBlockKeys(1L)
                .addBlockKeys(2L)
                .addBlockKeys(3L)
                .addBackendSelectors(BackendLocationSelector.newBuilder()
                        .setBackendType(StorageType.ST_NFS)
                        .setStrategy(LocationSelectStrategy.LSS_WEIGHTED_RANDOM)
                        .build())
                .build();

        GetCacheLocationsByBackendResponse response = httpClient.getCacheLocationsByBackend(request);

        // Verify response
        assertEquals(3, response.getKeyLocationsCount());
        for (int i = 0; i < 3; i++) {
            CacheLocationVector vector = response.getKeyLocations(i);
            assertEquals(1, vector.getLocationsCount());
            assertEquals(StorageType.ST_NFS, vector.getLocations(0).getType());
            assertFalse(vector.getLocations(0).getLocationSpecsList().isEmpty());
            assertFalse(vector.getLocations(0).getLocationSpecs(0).getUri().isEmpty());
        }
    }

    @Test
    void testGetCacheLocationsByBackend_partialKeyMatch() {
        // Setup: write keys [100, 300] but not 200
        registerInstance(instanceId);
        String sessionId = startWriteCache(instanceId, 100L, 300L);
        finishWriteCache(instanceId, sessionId, 2);

        // Query for keys [100, 200, 300]
        GetCacheLocationsByBackendRequest request = GetCacheLocationsByBackendRequest.newBuilder()
                .setTraceId("test-trace")
                .setInstanceId(instanceId)
                .setQueryType(QueryType.QT_BATCH_GET)
                .addBlockKeys(100L)
                .addBlockKeys(200L)
                .addBlockKeys(300L)
                .addBackendSelectors(BackendLocationSelector.newBuilder()
                        .setBackendType(StorageType.ST_NFS)
                        .setStrategy(LocationSelectStrategy.LSS_WEIGHTED_RANDOM)
                        .build())
                .build();

        GetCacheLocationsByBackendResponse response = httpClient.getCacheLocationsByBackend(request);

        // Verify: keys 100 and 300 have locations, key 200 does not
        assertEquals(3, response.getKeyLocationsCount());
        assertFalse(response.getKeyLocations(0).getLocationsList().isEmpty()); // key 100
        assertTrue(response.getKeyLocations(1).getLocationsList().isEmpty()); // key 200
        assertFalse(response.getKeyLocations(2).getLocationsList().isEmpty()); // key 300
    }

    @Test
    void testGetCacheLocationsByBackend_instanceNotExist() {
        GetCacheLocationsByBackendRequest request = GetCacheLocationsByBackendRequest.newBuilder()
                .setTraceId("test-trace")
                .setInstanceId("non_existent_instance")
                .setQueryType(QueryType.QT_BATCH_GET)
                .addBlockKeys(1L)
                .addBackendSelectors(BackendLocationSelector.newBuilder()
                        .setBackendType(StorageType.ST_NFS)
                        .setStrategy(LocationSelectStrategy.LSS_WEIGHTED_RANDOM)
                        .build())
                .build();

        assertThrows(Exception.class, () -> httpClient.getCacheLocationsByBackend(request));
    }

    @Test
    void testGetCacheLocationsByBackend_emptyBackendSelectors() {
        registerInstance(instanceId);

        GetCacheLocationsByBackendRequest request = GetCacheLocationsByBackendRequest.newBuilder()
                .setTraceId("test-trace")
                .setInstanceId(instanceId)
                .setQueryType(QueryType.QT_BATCH_GET)
                .addBlockKeys(1L)
                // No backend_selectors
                .build();

        assertThrows(Exception.class, () -> httpClient.getCacheLocationsByBackend(request));
    }

    @Test
    void testGetCacheLocationsByBackend_locationSpecNamesFilter() {
        // Setup: register with two location specs (tp0 and tp1)
        ModelDeployment deployment = ModelDeployment.newBuilder()
                .setModelName("test_model")
                .setDtype("FP8")
                .setUseMla(false)
                .setTpSize(1)
                .setDpSize(1)
                .setPpSize(1)
                .build();

        RegisterInstanceRequest regRequest = RegisterInstanceRequest.newBuilder()
                .setTraceId("test-trace")
                .setInstanceGroup("default")
                .setInstanceId(instanceId)
                .setBlockSize(128)
                .setModelDeployment(deployment)
                .addLocationSpecInfos(LocationSpecInfo.newBuilder().setName("tp0").setSize(1024).build())
                .addLocationSpecInfos(LocationSpecInfo.newBuilder().setName("tp1").setSize(1024).build())
                .build();

        grpcClient.registerInstance(regRequest);

        // Write cache
        String sessionId = startWriteCache(instanceId, 1L);
        finishWriteCache(instanceId, sessionId, 1);

        // Query with location_spec_names filter
        GetCacheLocationsByBackendRequest request = GetCacheLocationsByBackendRequest.newBuilder()
                .setTraceId("test-trace")
                .setInstanceId(instanceId)
                .setQueryType(QueryType.QT_BATCH_GET)
                .addBlockKeys(1L)
                .addLocationSpecNames("tp0")
                .addBackendSelectors(BackendLocationSelector.newBuilder()
                        .setBackendType(StorageType.ST_NFS)
                        .setStrategy(LocationSelectStrategy.LSS_WEIGHTED_RANDOM)
                        .build())
                .build();

        GetCacheLocationsByBackendResponse response = httpClient.getCacheLocationsByBackend(request);

        // Verify: only tp0 spec should be returned
        assertEquals(1, response.getKeyLocationsCount());
        CacheLocation location = response.getKeyLocations(0).getLocations(0);
        assertEquals(1, location.getLocationSpecsCount());
        assertEquals("tp0", location.getLocationSpecs(0).getName());
    }

    @Test
    void testGetCacheLocationsByBackend_tokenIdsConversion() {
        // Setup: register with block_size=128
        registerInstance(instanceId, 128);

        // Write cache using token_ids (not block_keys) so both paths use the same hash
        // Need 3 blocks * 128 tokens = 384 token_ids
        long[] tokenIds = new long[384];
        for (int i = 0; i < 384; i++) {
            tokenIds[i] = i + 1000; // deterministic token values
        }

        StartWriteCacheRequest.Builder writeBuilder = StartWriteCacheRequest.newBuilder()
                .setTraceId("test-trace")
                .setInstanceId(instanceId)
                .setWriteTimeoutSeconds(30);
        for (long t : tokenIds) {
            writeBuilder.addTokenIds(t);
        }
        StartWriteCacheResponse writeResp = grpcClient.startWriteCache(writeBuilder.build());
        String sessionId = writeResp.getWriteSessionId();
        finishWriteCache(instanceId, sessionId, 3);

        // Query with the same token_ids — server applies GenKeyVector() to get block_keys
        GetCacheLocationsByBackendRequest request = GetCacheLocationsByBackendRequest.newBuilder()
                .setTraceId("test-trace")
                .setInstanceId(instanceId)
                .setQueryType(QueryType.QT_BATCH_GET)
                .addAllTokenIds(java.util.Arrays.stream(tokenIds).boxed().collect(java.util.stream.Collectors.toList()))
                .addBackendSelectors(BackendLocationSelector.newBuilder()
                        .setBackendType(StorageType.ST_NFS)
                        .setStrategy(LocationSelectStrategy.LSS_WEIGHTED_RANDOM)
                        .build())
                .build();

        GetCacheLocationsByBackendResponse response = httpClient.getCacheLocationsByBackend(request);

        // Verify: 3 blocks should be returned (384 tokens / 128 block_size = 3)
        assertEquals(3, response.getKeyLocationsCount());
    }

    // === GetCacheMeta Tests ===

    @Test
    void testGetCacheMeta_servingStatus() {
        // Setup: write and finish cache
        registerInstance(instanceId);
        String sessionId = startWriteCache(instanceId, 1L, 2L, 3L);
        finishWriteCache(instanceId, sessionId, 3);

        // Query GetCacheMeta
        GetCacheMetaRequest request = GetCacheMetaRequest.newBuilder()
                .setTraceId("test-trace")
                .setInstanceId(instanceId)
                .addBlockKeys(1L)
                .addBlockKeys(2L)
                .addBlockKeys(3L)
                .build();

        GetCacheMetaResponse response = httpClient.getCacheMeta(request);

        // Verify: all keys should have CLS_SERVING status
        assertEquals(3, response.getMetasCount());
        for (int i = 0; i < 3; i++) {
            String meta = response.getMetas(i);
            assertTrue(meta.contains("CLS_SERVING"));
            assertTrue(meta.contains("id"));
        }
    }

    @Test
    void testGetCacheMeta_notFoundStatus() {
        // Setup: write keys [1, 2, 3]
        registerInstance(instanceId);
        String sessionId = startWriteCache(instanceId, 1L, 2L, 3L);
        finishWriteCache(instanceId, sessionId, 3);

        // Query for keys [1, 2, 3, 111111]
        GetCacheMetaRequest request = GetCacheMetaRequest.newBuilder()
                .setTraceId("test-trace")
                .setInstanceId(instanceId)
                .addBlockKeys(1L)
                .addBlockKeys(2L)
                .addBlockKeys(3L)
                .addBlockKeys(111111L)
                .build();

        GetCacheMetaResponse response = httpClient.getCacheMeta(request);

        // Verify: first 3 keys have CLS_SERVING, last key has CLS_NOT_FOUND
        assertEquals(4, response.getMetasCount());
        assertTrue(response.getMetas(0).contains("CLS_SERVING"));
        assertTrue(response.getMetas(1).contains("CLS_SERVING"));
        assertTrue(response.getMetas(2).contains("CLS_SERVING"));
        assertTrue(response.getMetas(3).contains("CLS_NOT_FOUND"));
    }

    @Test
    void testGetCacheMeta_instanceNotExist() {
        GetCacheMetaRequest request = GetCacheMetaRequest.newBuilder()
                .setTraceId("test-trace")
                .setInstanceId("non_existent_instance")
                .addBlockKeys(1L)
                .build();

        assertThrows(Exception.class, () -> httpClient.getCacheMeta(request));
    }

    @Test
    void testGetCacheMeta_validateJsonStructure() {
        // Setup: write cache
        registerInstance(instanceId);
        String sessionId = startWriteCache(instanceId, 1L);
        finishWriteCache(instanceId, sessionId, 1);

        // Query GetCacheMeta
        GetCacheMetaRequest request = GetCacheMetaRequest.newBuilder()
                .setTraceId("test-trace")
                .setInstanceId(instanceId)
                .addBlockKeys(1L)
                .build();

        GetCacheMetaResponse response = httpClient.getCacheMeta(request);

        // Verify: meta JSON structure
        assertEquals(1, response.getMetasCount());
        String meta = response.getMetas(0);

        // Parse JSON (simple check for required fields)
        assertTrue(meta.contains("\"status\""));
        assertTrue(meta.contains("\"id\""));
        assertTrue(meta.contains("CLS_SERVING") || meta.contains("CLS_NOT_FOUND") || meta.contains("CLS_WRITING"));
    }

    // === Known Limitation Tests ===

    @Test
    void testGetCacheLocationLen_noHttpEndpoint() {
        // This test documents that GetCacheLocationLen has no HTTP endpoint
        // and verifies that attempting to call it via HTTP returns 404

        registerInstance(instanceId);

        GetCacheLocationLenRequest request = GetCacheLocationLenRequest.newBuilder()
                .setTraceId("test-trace")
                .setInstanceId(instanceId)
                .setQueryType(QueryType.QT_BATCH_GET)
                .addBlockKeys(1L)
                .build();

        // Should throw exception because HTTP endpoint doesn't exist (404)
        Exception exception = assertThrows(Exception.class, () -> {
            httpClient.getCacheLocationLen(request);
        });

        // Verify it's an HTTP 404 error
        String message = exception.getMessage();
        assertTrue(message.contains("404") || message.contains("Not Found") || message.contains("not found"),
                "Expected 404 error but got: " + message);
    }
}
