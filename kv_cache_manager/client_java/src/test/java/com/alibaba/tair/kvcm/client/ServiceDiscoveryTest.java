package com.alibaba.tair.kvcm.client;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for StaticServiceDiscovery and ServiceDiscoveryFactory.
 */
class ServiceDiscoveryTest {

    // ======================== StaticServiceDiscovery ========================

    @Test
    void staticDiscovery_singleEndpoint() {
        StaticServiceDiscovery d = new StaticServiceDiscovery("10.0.0.1:6381");
        List<ServiceEndpoint> all = d.getAllEndpoints();
        assertEquals(1, all.size());
        assertEquals("10.0.0.1", all.get(0).getHost());
        assertEquals(6381, all.get(0).getPort());
        assertEquals("Static", d.getType());
        assertTrue(d.refresh());
    }

    @Test
    void staticDiscovery_multipleEndpoints_roundRobin() {
        StaticServiceDiscovery d = new StaticServiceDiscovery("10.0.0.1:6381,10.0.0.2:6382,10.0.0.3:6383");
        assertEquals(3, d.getAllEndpoints().size());

        // Round-robin: A, B, C, A, B, C
        assertEquals("10.0.0.1", d.getOneEndpoint().getHost());
        assertEquals("10.0.0.2", d.getOneEndpoint().getHost());
        assertEquals("10.0.0.3", d.getOneEndpoint().getHost());
        assertEquals("10.0.0.1", d.getOneEndpoint().getHost());
    }

    @Test
    void staticDiscovery_concurrentAccess() throws Exception {
        StaticServiceDiscovery d = new StaticServiceDiscovery("10.0.0.1:6381,10.0.0.2:6382");
        int threadCount = 10;
        int callsPerThread = 100;
        ExecutorService exec = Executors.newFixedThreadPool(threadCount);
        CountDownLatch latch = new CountDownLatch(threadCount);
        ConcurrentMap<String, Integer> counts = new ConcurrentHashMap<>();

        for (int t = 0; t < threadCount; t++) {
            exec.submit(() -> {
                try {
                    for (int i = 0; i < callsPerThread; i++) {
                        ServiceEndpoint ep = d.getOneEndpoint();
                        assertNotNull(ep);
                        counts.merge(ep.getHost(), 1, Integer::sum);
                    }
                } finally {
                    latch.countDown();
                }
            });
        }

        latch.await(5, TimeUnit.SECONDS);
        exec.shutdown();

        // Each host should be selected roughly equally (500 each)
        assertEquals(threadCount * callsPerThread,
                counts.values().stream().mapToInt(Integer::intValue).sum());
        assertTrue(counts.getOrDefault("10.0.0.1", 0) > 0);
        assertTrue(counts.getOrDefault("10.0.0.2", 0) > 0);
    }

    @Test
    void staticDiscovery_invalidPort_throws() {
        assertThrows(IllegalArgumentException.class,
                () -> new StaticServiceDiscovery("10.0.0.1:99999"));
    }

    @Test
    void staticDiscovery_missingPort_throws() {
        assertThrows(IllegalArgumentException.class,
                () -> new StaticServiceDiscovery("10.0.0.1"));
    }

    @Test
    void staticDiscovery_emptyList_throws() {
        assertThrows(IllegalArgumentException.class,
                () -> new StaticServiceDiscovery(""));
    }

    @Test
    void staticDiscovery_nullInput_throws() {
        assertThrows(IllegalArgumentException.class,
                () -> new StaticServiceDiscovery((String) null));
    }

    @Test
    void staticDiscovery_getAllEndpoints_defensiveCopy() {
        StaticServiceDiscovery d = new StaticServiceDiscovery("10.0.0.1:6381,10.0.0.2:6382");
        List<ServiceEndpoint> copy = d.getAllEndpoints();
        copy.clear();
        // Internal state should be unaffected
        assertEquals(2, d.getAllEndpoints().size());
    }

    // ======================== ServiceDiscoveryFactory ========================

    @Test
    void factory_staticUrl() {
        ServiceDiscovery d = ServiceDiscoveryFactory.create("static://10.0.0.1:6381");
        assertNotNull(d);
        assertInstanceOf(StaticServiceDiscovery.class, d);
        assertEquals("10.0.0.1", d.getOneEndpoint().getHost());
    }

    @Test
    void factory_staticUrl_multipleEndpoints() {
        ServiceDiscovery d = ServiceDiscoveryFactory.create("static://10.0.0.1:6381,10.0.0.2:6382");
        assertNotNull(d);
        assertEquals(2, d.getAllEndpoints().size());
    }

    @Test
    void factory_staticUrl_withQueryParams() {
        // Static ignores query params, but parsing should succeed
        ServiceDiscovery d = ServiceDiscoveryFactory.create("static://10.0.0.1:6381?foo=bar");
        assertNotNull(d);
        assertEquals(1, d.getAllEndpoints().size());
    }

    @Test
    void factory_unknownScheme_throws() {
        assertThrows(IllegalArgumentException.class,
                () -> ServiceDiscoveryFactory.create("unknown://something"));
    }

    @Test
    void factory_nullUrl_throws() {
        assertThrows(IllegalArgumentException.class,
                () -> ServiceDiscoveryFactory.create(null));
    }

    @Test
    void factory_emptyUrl_throws() {
        assertThrows(IllegalArgumentException.class,
                () -> ServiceDiscoveryFactory.create(""));
    }

    @Test
    void factory_missingScheme_throws() {
        assertThrows(IllegalArgumentException.class,
                () -> ServiceDiscoveryFactory.create("no-scheme-here"));
    }

    @Test
    void factory_emptyBody_throws() {
        assertThrows(IllegalArgumentException.class,
                () -> ServiceDiscoveryFactory.create("static://"));
    }

    @Test
    void factory_tryCreate_returnsNullOnFailure() {
        assertNull(ServiceDiscoveryFactory.tryCreate("unknown://something"));
        assertNull(ServiceDiscoveryFactory.tryCreate(null));
    }

    @Test
    void factory_tryCreate_returnsInstanceOnSuccess() {
        ServiceDiscovery d = ServiceDiscoveryFactory.tryCreate("static://10.0.0.1:6381");
        assertNotNull(d);
    }

    // ======================== ServiceEndpoint ========================

    @Test
    void endpoint_defaults() {
        ServiceEndpoint ep = new ServiceEndpoint("10.0.0.1", 6381);
        assertEquals("10.0.0.1", ep.getHost());
        assertEquals(6381, ep.getPort());
        assertEquals(100, ep.getWeight());
        assertTrue(ep.isHealthy());
        assertEquals("10.0.0.1:6381", ep.getAddress());
    }

    @Test
    void endpoint_equalsAndHashCode() {
        ServiceEndpoint a = new ServiceEndpoint("10.0.0.1", 6381);
        ServiceEndpoint b = new ServiceEndpoint("10.0.0.1", 6381);
        ServiceEndpoint c = new ServiceEndpoint("10.0.0.2", 6381);
        assertEquals(a, b);
        assertEquals(a.hashCode(), b.hashCode());
        assertNotEquals(a, c);
    }

    @Test
    void endpoint_invalidPort_throws() {
        assertThrows(IllegalArgumentException.class, () -> new ServiceEndpoint("host", 0));
        assertThrows(IllegalArgumentException.class, () -> new ServiceEndpoint("host", -1));
        assertThrows(IllegalArgumentException.class, () -> new ServiceEndpoint("host", 70000));
    }

    @Test
    void endpoint_emptyHost_throws() {
        assertThrows(IllegalArgumentException.class, () -> new ServiceEndpoint("", 6381));
        assertThrows(IllegalArgumentException.class, () -> new ServiceEndpoint(null, 6381));
    }

    // ======================== URL query parsing ========================

    @Test
    void parseQueryString_basic() {
        java.util.Map<String, String> params = ServiceDiscoveryFactory.parseQueryString("a=1&b=hello&c=");
        assertEquals("1", params.get("a"));
        assertEquals("hello", params.get("b"));
        assertEquals("", params.get("c"));
    }

    @Test
    void parseQueryString_empty() {
        assertTrue(ServiceDiscoveryFactory.parseQueryString("").isEmpty());
        assertTrue(ServiceDiscoveryFactory.parseQueryString(null).isEmpty());
    }
}
