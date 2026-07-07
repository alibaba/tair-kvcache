package com.alibaba.tair.kvcm.client;

import kv_cache_manager.proto.meta.MetaServiceOuterClass.*;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration tests for CacheAware RPCs using HTTP transport.
 * <p>
 * Note: GetCacheLocationLen has no HTTP endpoint, so those tests are skipped.
 */
public class CacheAwareHttpTest extends CacheAwareTestBase {

    @Override
    protected MetaClient getClient() {
        return httpClient;
    }

    @Test
    void testGetCacheLocationsByBackend_basicQuery() {
        super.testGetCacheLocationsByBackend_basicQuery();
    }

    @Test
    void testGetCacheLocationsByBackend_partialKeyMatch() {
        super.testGetCacheLocationsByBackend_partialKeyMatch();
    }

    @Test
    void testGetCacheLocationsByBackend_instanceNotExist() {
        super.testGetCacheLocationsByBackend_instanceNotExist();
    }

    @Test
    void testGetCacheLocationsByBackend_emptyBackendSelectors() {
        super.testGetCacheLocationsByBackend_emptyBackendSelectors();
    }

    @Test
    void testGetCacheLocationsByBackend_locationSpecNamesFilter() {
        super.testGetCacheLocationsByBackend_locationSpecNamesFilter();
    }

    @Test
    void testGetCacheLocationsByBackend_tokenIdsConversion() {
        super.testGetCacheLocationsByBackend_tokenIdsConversion();
    }

    @Test
    void testGetCacheMeta_servingStatus() {
        super.testGetCacheMeta_servingStatus();
    }

    @Test
    void testGetCacheMeta_notFoundStatus() {
        super.testGetCacheMeta_notFoundStatus();
    }

    @Test
    void testGetCacheMeta_instanceNotExist() {
        super.testGetCacheMeta_instanceNotExist();
    }

    @Test
    void testGetCacheMeta_validateJsonStructure() {
        super.testGetCacheMeta_validateJsonStructure();
    }

    /**
     * HTTP-specific test: verify that GetCacheLocationLen returns 404 via HTTP.
     * This RPC has no HTTP endpoint registered in the server.
     */
    @Test
    void testGetCacheLocationLen_noHttpEndpoint() {
        MetaClient client = getClient();
        registerInstance(instanceId);

        GetCacheLocationLenRequest request = GetCacheLocationLenRequest.newBuilder()
                .setTraceId("test-trace")
                .setInstanceId(instanceId)
                .setQueryType(QueryType.QT_BATCH_GET)
                .addBlockKeys(1L)
                .build();

        Exception exception = assertThrows(Exception.class, () -> {
            client.getCacheLocationLen(request);
        });

        String message = exception.getMessage();
        assertTrue(message.contains("404") || message.contains("Not Found") || message.contains("not found"),
                "Expected 404 error but got: " + message);
    }
}
