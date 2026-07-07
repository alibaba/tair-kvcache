package com.alibaba.tair.kvcm.client;

import org.junit.jupiter.api.Test;

/**
 * Integration tests for CacheAware RPCs using gRPC transport.
 */
public class CacheAwareGrpcTest extends CacheAwareTestBase {

    @Override
    protected MetaClient getClient() {
        return grpcClient;
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
    void testGetCacheLocationLen_gRpcOnly() {
        super.testGetCacheLocationLen_gRpcOnly();
    }

    @Test
    void testGetCacheLocationLen_noMatches() {
        super.testGetCacheLocationLen_noMatches();
    }

    @Test
    void testGetCacheLocationLen_batchGet() {
        super.testGetCacheLocationLen_batchGet();
    }

    @Test
    void testGetCacheLocationLen_batchGetNoMatches() {
        super.testGetCacheLocationLen_batchGetNoMatches();
    }

    @Test
    void testGetCacheLocationLen_consistency() {
        super.testGetCacheLocationLen_consistency();
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
}
