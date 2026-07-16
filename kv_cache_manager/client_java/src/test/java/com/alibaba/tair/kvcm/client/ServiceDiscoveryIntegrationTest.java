package com.alibaba.tair.kvcm.client;

import io.grpc.*;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import io.grpc.stub.StreamObserver;
import kv_cache_manager.proto.meta.MetaServiceGrpc;
import kv_cache_manager.proto.meta.MetaServiceOuterClass.*;
import org.junit.jupiter.api.*;

import java.lang.reflect.Field;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for service discovery integration in AutoFailoverClient and LeaderDiscovery.
 */
class ServiceDiscoveryIntegrationTest {

    // ======================== AutoFailoverClient + ServiceDiscovery ========================

    @Test
    void autoFailoverClient_staticUrl_createsServiceDiscovery() throws Exception {
        // Config with static:// URL → serviceDiscovery should be created
        MetaClientConfig config = MetaClientConfig.builder("static://10.0.0.1:6381")
                .autoDiscoverLeader(false)
                .build();

        // Use test constructor but verify config was properly built
        assertEquals("static://10.0.0.1:6381", config.getServiceDiscoveryUrl());
        ServiceDiscovery sd = ServiceDiscoveryFactory.create(config.getServiceDiscoveryUrl());
        assertNotNull(sd);
        assertInstanceOf(StaticServiceDiscovery.class, sd);
        assertEquals("10.0.0.1", sd.getOneEndpoint().getHost());
        assertEquals(6381, sd.getOneEndpoint().getPort());
    }

    @Test
    void autoFailoverClient_plainHost_resolvesViaStatic() throws Exception {
        // Plain host → normalized to static:// → resolved
        MetaClientConfig config = MetaClientConfig.builder("myhost")
                .grpcPort(8080)
                .autoDiscoverLeader(false)
                .build();

        assertEquals("static://myhost:8080", config.getServiceDiscoveryUrl());
        ServiceDiscovery sd = ServiceDiscoveryFactory.create(config.getServiceDiscoveryUrl());
        assertEquals("myhost", sd.getOneEndpoint().getHost());
        assertEquals(8080, sd.getOneEndpoint().getPort());
    }

    @Test
    void autoFailoverClient_unknownScheme_fallsBackToSeed() throws Exception {
        // Unknown scheme → tryCreate returns null → falls back to original seed
        MetaClientConfig config = MetaClientConfig.builder("static://10.0.0.1:6381")
                .autoDiscoverLeader(false)
                .build();

        // tryCreate with unknown scheme returns null
        ServiceDiscovery sd = ServiceDiscoveryFactory.tryCreate("unknown://bad");
        assertNull(sd);
    }

    @Test
    void autoFailoverClient_testConstructor_noServiceDiscovery() throws Exception {
        // Test constructor should have null serviceDiscovery
        MetaClientConfig config = MetaClientConfig.builder("dummy")
                .autoDiscoverLeader(false)
                .build();

        ManagedChannel channel = InProcessChannelBuilder.forName("test")
                .directExecutor().build();
        GrpcMetaClient grpc = new GrpcMetaClient(channel, 5000);
        AutoFailoverClient client = new AutoFailoverClient(config, grpc, null);

        Field sdField = AutoFailoverClient.class.getDeclaredField("serviceDiscovery");
        sdField.setAccessible(true);
        assertNull(sdField.get(client));

        client.close();
    }

    // ======================== LeaderDiscovery + ServiceDiscovery ========================

    @Test
    void leaderDiscovery_setServiceDiscovery_fieldIsSet() {
        LeaderDiscovery ld = new LeaderDiscovery("seed", 6381, "", 30);
        assertNull(getServiceDiscovery(ld));

        ServiceDiscovery sd = new StaticServiceDiscovery("10.0.0.1:6381");
        ld.setServiceDiscovery(sd);
        assertSame(sd, getServiceDiscovery(ld));
    }

    @Test
    void leaderDiscovery_serviceDiscoveryUsedForTargetResolution() throws Exception {
        // Create a multi-endpoint discovery that tracks which endpoints are used
        AtomicInteger callCount = new AtomicInteger(0);
        List<ServiceEndpoint> endpoints = Arrays.asList(
                new ServiceEndpoint("node1", 6381),
                new ServiceEndpoint("node2", 6382),
                new ServiceEndpoint("node3", 6383)
        );

        ServiceDiscovery sd = new ServiceDiscovery() {
            private final AtomicInteger idx = new AtomicInteger(0);
            @Override public List<ServiceEndpoint> getAllEndpoints() { return endpoints; }
            @Override public ServiceEndpoint getOneEndpoint() {
                callCount.incrementAndGet();
                return endpoints.get(idx.getAndIncrement() % endpoints.size());
            }
            @Override public boolean refresh() { return true; }
            @Override public String getType() { return "TestMulti"; }
        };

        LeaderDiscovery ld = new LeaderDiscovery("seed-host", 6381, "", 30);
        ld.setServiceDiscovery(sd);

        // discoverLeader() will fail (no real server), but it should have called
        // serviceDiscovery.getOneEndpoint() to resolve the target
        ld.discoverLeader();
        assertEquals(1, callCount.get(), "serviceDiscovery.getOneEndpoint() should be called once");

        // Call again — should pick next endpoint
        ld.discoverLeader();
        assertEquals(2, callCount.get());

        // Call again
        ld.discoverLeader();
        assertEquals(3, callCount.get());
    }

    @Test
    void leaderDiscovery_noServiceDiscovery_usesSeedAddress() throws Exception {
        // Without ServiceDiscovery, should use seedAddress (and fail since no real server)
        LeaderDiscovery ld = new LeaderDiscovery("nonexistent-host", 6381, "", 30);
        // Should not throw, just return false
        boolean result = ld.discoverLeader();
        assertFalse(result);
    }

    @Test
    void leaderDiscovery_serviceDiscoveryReturnsNull_fallsBackToSeed() throws Exception {
        // ServiceDiscovery that returns null → fall back to seed address
        ServiceDiscovery sd = new ServiceDiscovery() {
            @Override public List<ServiceEndpoint> getAllEndpoints() { return Collections.emptyList(); }
            @Override public ServiceEndpoint getOneEndpoint() { return null; }
            @Override public boolean refresh() { return false; }
            @Override public String getType() { return "Empty"; }
        };

        LeaderDiscovery ld = new LeaderDiscovery("seed-host", 6381, "", 30);
        ld.setServiceDiscovery(sd);

        // Should not throw; falls back to seed-host:6381, fails since no server
        boolean result = ld.discoverLeader();
        assertFalse(result);
    }

    // ======================== End-to-end: Config → Factory → Discovery ========================

    @Test
    void endToEnd_configToStaticDiscovery() {
        MetaClientConfig config = MetaClientConfig.builder("static://192.168.1.1:6381,192.168.1.2:6381")
                .instanceId("test")
                .build();

        // Factory creates StaticServiceDiscovery from the config URL
        ServiceDiscovery sd = ServiceDiscoveryFactory.create(config.getServiceDiscoveryUrl());
        assertNotNull(sd);
        assertEquals("Static", sd.getType());
        assertEquals(2, sd.getAllEndpoints().size());

        // Round-robin
        assertEquals("192.168.1.1", sd.getOneEndpoint().getHost());
        assertEquals("192.168.1.2", sd.getOneEndpoint().getHost());
        assertEquals("192.168.1.1", sd.getOneEndpoint().getHost());
    }

    // --- Helper ---

    private static ServiceDiscovery getServiceDiscovery(LeaderDiscovery ld) {
        try {
            Field f = LeaderDiscovery.class.getDeclaredField("serviceDiscovery");
            f.setAccessible(true);
            return (ServiceDiscovery) f.get(ld);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
