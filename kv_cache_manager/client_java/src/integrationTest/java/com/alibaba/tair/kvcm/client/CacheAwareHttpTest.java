package com.alibaba.tair.kvcm.client;

import org.junit.jupiter.api.Test;

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
}
