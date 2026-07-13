package com.alibaba.tair.kvcm.client;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for MetaClientConfig URL scheme support.
 */
class MetaClientConfigTest {

    // ======================== normalizeToDiscoveryUrl ========================

    @Test
    void normalize_staticUrl() {
        assertEquals("static://10.0.0.1:6381",
                MetaClientConfig.normalizeToDiscoveryUrl("static://10.0.0.1:6381", 6381));
    }

    @Test
    void normalize_staticUrl_withQueryParams() {
        String result = MetaClientConfig.normalizeToDiscoveryUrl(
                "static://10.0.0.1:6381,10.0.0.2:6381?foo=bar", 6381);
        assertTrue(result.startsWith("static://"));
        assertTrue(result.contains("foo=bar"));
    }

    @Test
    void normalize_spectrumUrl() {
        assertEquals("spectrum://v-xxxx",
                MetaClientConfig.normalizeToDiscoveryUrl("spectrum://v-xxxx", 6381));
    }

    @Test
    void normalize_plainHost_portProvided() {
        // Builder uses grpcPort as default
        assertEquals("static://10.0.0.1:6381",
                MetaClientConfig.normalizeToDiscoveryUrl("10.0.0.1", 6381));
    }

    @Test
    void normalize_plainHostWithPort() {
        assertEquals("static://10.0.0.1:8080",
                MetaClientConfig.normalizeToDiscoveryUrl("10.0.0.1:8080", 6381));
    }

    @Test
    void normalize_emptyBody_throws() {
        assertThrows(IllegalArgumentException.class,
                () -> MetaClientConfig.normalizeToDiscoveryUrl("static://", 6381));
    }

    @Test
    void normalize_malformedScheme_throws() {
        assertThrows(IllegalArgumentException.class,
                () -> MetaClientConfig.normalizeToDiscoveryUrl("://host:port", 6381));
    }

    // ======================== Builder ========================

    @Test
    void builder_staticUrlConfig() {
        MetaClientConfig config = MetaClientConfig.builder("static://10.0.0.1:6381")
                .instanceId("test-instance")
                .build();
        assertEquals("static://10.0.0.1:6381", config.getSeedAddress());
        assertEquals("static://10.0.0.1:6381", config.getServiceDiscoveryUrl());
        assertEquals("test-instance", config.getInstanceId());
    }

    @Test
    void builder_plainHost_autoConvertToStatic() {
        MetaClientConfig config = MetaClientConfig.builder("10.0.0.1")
                .grpcPort(8080)
                .build();
        assertEquals("10.0.0.1", config.getSeedAddress());
        assertEquals("static://10.0.0.1:8080", config.getServiceDiscoveryUrl());
    }

    @Test
    void builder_nullSeedAddress_throws() {
        assertThrows(IllegalArgumentException.class,
                () -> MetaClientConfig.builder(null));
    }

    @Test
    void builder_emptySeedAddress_throws() {
        assertThrows(IllegalArgumentException.class,
                () -> MetaClientConfig.builder(""));
    }
}
