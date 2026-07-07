package com.alibaba.tair.kvcm.client;

import kv_cache_manager.proto.meta.MetaServiceOuterClass.*;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.TestInfo;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Shared test logic for CacheAware RPCs.
 * <p>
 * Subclasses provide the MetaClient instance (gRPC or HTTP) to test.
 * Setup operations (registerInstance, startWriteCache, finishWriteCache) use gRPC
 * as the primary transport to avoid code duplication.
 */
public abstract class CacheAwareTestBase extends IntegrationTestBase {

    private static final Logger LOG = LoggerFactory.getLogger(CacheAwareTestBase.class);

    private String instanceId;

    @BeforeEach
    void setUpTestIsolation(TestInfo testInfo) {
        String methodName = testInfo.getTestMethod()
                .map(m -> m.getName())
                .orElse("unknown");
        instanceId = "test_" + methodName + "_" + System.currentTimeMillis();
    }

    /**
     * Returns the MetaClient to test (gRPC or HTTP).
     */
    protected abstract MetaClient getClient();

    // === GetCacheLocationsByBackend Tests ===

    void testGetCacheLocationsByBackend_basicQuery() {
        MetaClient client = getClient();
        registerInstance(instanceId);
        String sessionId = startWriteCache(instanceId, 1L, 2L, 3L);
        finishWriteCache(instanceId, sessionId, 3);

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

        GetCacheLocationsByBackendResponse response = client.getCacheLocationsByBackend(request);

        assertEquals(3, response.getKeyLocationsCount());
        for (int i = 0; i < 3; i++) {
            CacheLocationVector vector = response.getKeyLocations(i);
            assertEquals(1, vector.getLocationsCount());
            assertEquals(StorageType.ST_NFS, vector.getLocations(0).getType());
            assertFalse(vector.getLocations(0).getLocationSpecsList().isEmpty());
            assertFalse(vector.getLocations(0).getLocationSpecs(0).getUri().isEmpty());
        }
    }

    void testGetCacheLocationsByBackend_partialKeyMatch() {
        MetaClient client = getClient();
        registerInstance(instanceId);
        String sessionId = startWriteCache(instanceId, 100L, 300L);
        finishWriteCache(instanceId, sessionId, 2);

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

        GetCacheLocationsByBackendResponse response = client.getCacheLocationsByBackend(request);

        assertEquals(3, response.getKeyLocationsCount());
        assertFalse(response.getKeyLocations(0).getLocationsList().isEmpty());
        assertTrue(response.getKeyLocations(1).getLocationsList().isEmpty());
        assertFalse(response.getKeyLocations(2).getLocationsList().isEmpty());
    }

    void testGetCacheLocationsByBackend_instanceNotExist() {
        MetaClient client = getClient();
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

        assertThrows(Exception.class, () -> client.getCacheLocationsByBackend(request));
    }

    void testGetCacheLocationsByBackend_emptyBackendSelectors() {
        MetaClient client = getClient();
        registerInstance(instanceId);

        GetCacheLocationsByBackendRequest request = GetCacheLocationsByBackendRequest.newBuilder()
                .setTraceId("test-trace")
                .setInstanceId(instanceId)
                .setQueryType(QueryType.QT_BATCH_GET)
                .addBlockKeys(1L)
                .build();

        assertThrows(Exception.class, () -> client.getCacheLocationsByBackend(request));
    }

    void testGetCacheLocationsByBackend_locationSpecNamesFilter() {
        MetaClient client = getClient();
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

        String sessionId = startWriteCache(instanceId, 1L);
        finishWriteCache(instanceId, sessionId, 1);

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

        GetCacheLocationsByBackendResponse response = client.getCacheLocationsByBackend(request);

        assertEquals(1, response.getKeyLocationsCount());
        CacheLocation location = response.getKeyLocations(0).getLocations(0);
        assertEquals(1, location.getLocationSpecsCount());
        assertEquals("tp0", location.getLocationSpecs(0).getName());
    }

    void testGetCacheLocationsByBackend_tokenIdsConversion() {
        MetaClient client = getClient();
        registerInstance(instanceId, 128);

        long[] tokenIds = new long[384];
        for (int i = 0; i < 384; i++) {
            tokenIds[i] = i + 1000;
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

        GetCacheLocationsByBackendResponse response = client.getCacheLocationsByBackend(request);

        assertEquals(3, response.getKeyLocationsCount());
    }

    // === GetCacheLocationLen Tests (gRPC only) ===

    void testGetCacheLocationLen_gRpcOnly() {
        MetaClient client = getClient();
        registerInstance(instanceId);
        String sessionId = startWriteCache(instanceId, 1L, 2L, 3L);
        finishWriteCache(instanceId, sessionId, 3);

        GetCacheLocationLenRequest request = GetCacheLocationLenRequest.newBuilder()
                .setTraceId("test-trace")
                .setInstanceId(instanceId)
                .setQueryType(QueryType.QT_PREFIX_MATCH)
                .addBlockKeys(1L)
                .addBlockKeys(2L)
                .addBlockKeys(3L)
                .addBlockKeys(8L)
                .addBlockKeys(5L)
                .addBlockKeys(6L)
                .build();

        GetCacheLocationLenResponse response = client.getCacheLocationLen(request);
        assertEquals(3, response.getCacheLocationLen());
    }

    void testGetCacheLocationLen_noMatches() {
        MetaClient client = getClient();
        registerInstance(instanceId);

        GetCacheLocationLenRequest request = GetCacheLocationLenRequest.newBuilder()
                .setTraceId("test-trace")
                .setInstanceId(instanceId)
                .setQueryType(QueryType.QT_PREFIX_MATCH)
                .addBlockKeys(999L)
                .addBlockKeys(1L)
                .addBlockKeys(2L)
                .build();

        GetCacheLocationLenResponse response = client.getCacheLocationLen(request);
        assertEquals(0, response.getCacheLocationLen());
    }

    void testGetCacheLocationLen_batchGet() {
        MetaClient client = getClient();
        registerInstance(instanceId);
        String sessionId = startWriteCache(instanceId, 1L, 2L, 3L, 4L, 5L, 6L, 7L);
        finishWriteCacheWithOffset(instanceId, sessionId, 5);

        GetCacheLocationLenRequest request = GetCacheLocationLenRequest.newBuilder()
                .setTraceId("test-trace")
                .setInstanceId(instanceId)
                .setQueryType(QueryType.QT_BATCH_GET)
                .addBlockKeys(1L)
                .addBlockKeys(2L)
                .addBlockKeys(8L)
                .addBlockKeys(4L)
                .addBlockKeys(5L)
                .addBlockKeys(9L)
                .addBlockKeys(6L)
                .build();

        GetCacheLocationLenResponse response = client.getCacheLocationLen(request);
        assertEquals(4, response.getCacheLocationLen());
    }

    void testGetCacheLocationLen_batchGetNoMatches() {
        MetaClient client = getClient();
        registerInstance(instanceId);

        GetCacheLocationLenRequest request = GetCacheLocationLenRequest.newBuilder()
                .setTraceId("test-trace")
                .setInstanceId(instanceId)
                .setQueryType(QueryType.QT_BATCH_GET)
                .addBlockKeys(1L)
                .addBlockKeys(2L)
                .addBlockKeys(3L)
                .build();

        GetCacheLocationLenResponse response = client.getCacheLocationLen(request);
        assertEquals(0, response.getCacheLocationLen());
    }

    void testGetCacheLocationLen_consistency() {
        MetaClient client = getClient();
        registerInstance(instanceId);
        String sessionId = startWriteCache(instanceId, 1L, 2L, 3L, 4L, 5L);
        finishWriteCache(instanceId, sessionId, 5);

        GetCacheLocationRequest locationRequest = GetCacheLocationRequest.newBuilder()
                .setTraceId("test-trace")
                .setInstanceId(instanceId)
                .setQueryType(QueryType.QT_BATCH_GET)
                .addBlockKeys(1L)
                .addBlockKeys(2L)
                .addBlockKeys(3L)
                .addBlockKeys(4L)
                .addBlockKeys(5L)
                .build();

        GetCacheLocationResponse locationResponse = client.getCacheLocation(locationRequest);

        GetCacheLocationLenRequest lenRequest = GetCacheLocationLenRequest.newBuilder()
                .setTraceId("test-trace")
                .setInstanceId(instanceId)
                .setQueryType(QueryType.QT_BATCH_GET)
                .addBlockKeys(1L)
                .addBlockKeys(2L)
                .addBlockKeys(3L)
                .addBlockKeys(4L)
                .addBlockKeys(5L)
                .build();

        GetCacheLocationLenResponse lenResponse = client.getCacheLocationLen(lenRequest);

        assertEquals(locationResponse.getLocationsCount(), lenResponse.getCacheLocationLen());
    }

    // === GetCacheMeta Tests ===

    void testGetCacheMeta_servingStatus() {
        MetaClient client = getClient();
        registerInstance(instanceId);
        String sessionId = startWriteCache(instanceId, 1L, 2L, 3L);
        finishWriteCache(instanceId, sessionId, 3);

        GetCacheMetaRequest request = GetCacheMetaRequest.newBuilder()
                .setTraceId("test-trace")
                .setInstanceId(instanceId)
                .addBlockKeys(1L)
                .addBlockKeys(2L)
                .addBlockKeys(3L)
                .build();

        GetCacheMetaResponse response = client.getCacheMeta(request);

        assertEquals(3, response.getMetasCount());
        for (int i = 0; i < 3; i++) {
            String meta = response.getMetas(i);
            assertTrue(meta.contains("CLS_SERVING"));
            assertTrue(meta.contains("id"));
        }
    }

    void testGetCacheMeta_notFoundStatus() {
        MetaClient client = getClient();
        registerInstance(instanceId);
        String sessionId = startWriteCache(instanceId, 1L, 2L, 3L);
        finishWriteCache(instanceId, sessionId, 3);

        GetCacheMetaRequest request = GetCacheMetaRequest.newBuilder()
                .setTraceId("test-trace")
                .setInstanceId(instanceId)
                .addBlockKeys(1L)
                .addBlockKeys(2L)
                .addBlockKeys(3L)
                .addBlockKeys(111111L)
                .build();

        GetCacheMetaResponse response = client.getCacheMeta(request);

        assertEquals(4, response.getMetasCount());
        assertTrue(response.getMetas(0).contains("CLS_SERVING"));
        assertTrue(response.getMetas(1).contains("CLS_SERVING"));
        assertTrue(response.getMetas(2).contains("CLS_SERVING"));
        assertTrue(response.getMetas(3).contains("CLS_NOT_FOUND"));
    }

    void testGetCacheMeta_instanceNotExist() {
        MetaClient client = getClient();
        GetCacheMetaRequest request = GetCacheMetaRequest.newBuilder()
                .setTraceId("test-trace")
                .setInstanceId("non_existent_instance")
                .addBlockKeys(1L)
                .build();

        assertThrows(Exception.class, () -> client.getCacheMeta(request));
    }

    void testGetCacheMeta_validateJsonStructure() {
        MetaClient client = getClient();
        registerInstance(instanceId);
        String sessionId = startWriteCache(instanceId, 1L);
        finishWriteCache(instanceId, sessionId, 1);

        GetCacheMetaRequest request = GetCacheMetaRequest.newBuilder()
                .setTraceId("test-trace")
                .setInstanceId(instanceId)
                .addBlockKeys(1L)
                .build();

        GetCacheMetaResponse response = client.getCacheMeta(request);

        assertEquals(1, response.getMetasCount());
        String meta = response.getMetas(0);
        assertTrue(meta.contains("\"status\""));
        assertTrue(meta.contains("\"id\""));
        assertTrue(meta.contains("CLS_SERVING") || meta.contains("CLS_NOT_FOUND") || meta.contains("CLS_WRITING"));
    }
}
