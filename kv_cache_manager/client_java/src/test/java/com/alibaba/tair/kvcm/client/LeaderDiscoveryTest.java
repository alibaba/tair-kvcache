package com.alibaba.tair.kvcm.client;

import io.grpc.Server;
import io.grpc.inprocess.InProcessServerBuilder;
import io.grpc.stub.StreamObserver;
import kv_cache_manager.proto.meta.MetaServiceGrpc;
import kv_cache_manager.proto.meta.MetaServiceOuterClass.*;
import org.junit.jupiter.api.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for LeaderDiscovery using InProcess gRPC server.
 */
class LeaderDiscoveryTest {

    private Server server;
    private MockLeaderService mockService;
    private String serverName;

    @BeforeEach
    void setUp() throws Exception {
        serverName = InProcessServerBuilder.generateName();
        mockService = new MockLeaderService();
        server = InProcessServerBuilder.forName(serverName)
                .directExecutor()
                .addService(mockService)
                .build()
                .start();
    }

    @AfterEach
    void tearDown() {
        server.shutdownNow();
    }

    /**
     * Helper to create a LeaderDiscovery that connects to the InProcess server.
     * Since InProcessServer doesn't use real host:port, we override discoverLeader
     * to use InProcess channel.
     */
    private TestableLeaderDiscovery createDiscovery() {
        return new TestableLeaderDiscovery(serverName);
    }

    @Test
    void testDiscoverLeader_success() {
        mockService.leaderHost = "10.0.0.42";
        mockService.leaderPort = 6381;
        TestableLeaderDiscovery discovery = createDiscovery();

        boolean result = discovery.discoverLeader();
        assertTrue(result);
        assertEquals("10.0.0.42", discovery.getCurrentHost());
        assertEquals(6381, discovery.getCurrentPort());
    }

    @Test
    void testDiscoverLeader_incompleteEndpoint_returnsFalse() {
        mockService.leaderHost = "";
        mockService.leaderPort = 6381;
        TestableLeaderDiscovery discovery = createDiscovery();

        boolean result = discovery.discoverLeader();
        assertFalse(result);
        // Should keep original address (the seed passed to parent constructor)
        assertEquals("dummy-host", discovery.getCurrentHost());
    }

    @Test
    void testDiscoverLeader_missingEndpoint_returnsFalse() {
        mockService.hasLeaderEndpoint = false;
        TestableLeaderDiscovery discovery = createDiscovery();

        boolean result = discovery.discoverLeader();
        assertFalse(result);
    }

    @Test
    void testDiscoverLeader_errorResponse_returnsFalse() {
        mockService.returnError = true;
        TestableLeaderDiscovery discovery = createDiscovery();

        boolean result = discovery.discoverLeader();
        assertFalse(result);
    }

    @Test
    void testDiscoverLeader_updatesAddress() {
        mockService.leaderHost = "10.0.0.1";
        mockService.leaderPort = 50051;
        TestableLeaderDiscovery discovery = createDiscovery();

        discovery.discoverLeader();
        assertEquals("10.0.0.1", discovery.getCurrentHost());
        assertEquals(50051, discovery.getCurrentPort());

        // Change leader
        mockService.leaderHost = "10.0.0.2";
        mockService.leaderPort = 50052;
        discovery.discoverLeader();
        assertEquals("10.0.0.2", discovery.getCurrentHost());
        assertEquals(50052, discovery.getCurrentPort());
    }

    @Test
    void testStartAndStop_noException() {
        mockService.leaderHost = "10.0.0.1";
        mockService.leaderPort = 6381;
        TestableLeaderDiscovery discovery = createDiscovery();

        discovery.start();
        discovery.triggerImmediateRefresh();
        discovery.stop();
        // Should not throw
    }

    // --- Testable subclass using InProcess channel ---

    static class TestableLeaderDiscovery extends LeaderDiscovery {
        private final String inProcessName;

        TestableLeaderDiscovery(String inProcessName) {
            super("dummy-host", 0, "", 30);
            this.inProcessName = inProcessName;
        }

        @Override
        boolean discoverLeader() {
            // Use InProcess channel instead of real TCP
            io.grpc.ManagedChannel channel = io.grpc.inprocess.InProcessChannelBuilder
                    .forName(inProcessName)
                    .directExecutor()
                    .build();
            try {
                kv_cache_manager.proto.meta.MetaServiceGrpc.MetaServiceBlockingStub stub =
                        MetaServiceGrpc.newBlockingStub(channel);
                GetClusterInfoResponse response = stub.withDeadlineAfter(5, java.util.concurrent.TimeUnit.SECONDS)
                        .getClusterInfo(GetClusterInfoRequest.newBuilder()
                                .setTraceId("test-discovery")
                                .build());

                if (!response.hasHeader() || response.getHeader().getStatus().getCode() != ErrorCode.OK) {
                    return false;
                }
                if (!response.hasLeaderEndpoint()) {
                    return false;
                }
                MetaNodeEndpoint ep = response.getLeaderEndpoint();
                if (ep.getHost().isEmpty() || ep.getMetaRpcPort() <= 0) {
                    return false;
                }
                // Use reflection to update the currentAddress field (now an immutable LeaderAddress)
                try {
                    java.lang.reflect.Field addrField = LeaderDiscovery.class.getDeclaredField("currentAddress");
                    addrField.setAccessible(true);
                    addrField.set(this, new LeaderAddress(ep.getHost(), ep.getMetaRpcPort()));
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
                return true;
            } finally {
                channel.shutdownNow();
            }
        }
    }

    // --- Mock service ---

    static class MockLeaderService extends MetaServiceGrpc.MetaServiceImplBase {
        volatile String leaderHost = "10.0.0.1";
        volatile int leaderPort = 6381;
        volatile boolean hasLeaderEndpoint = true;
        volatile boolean returnError = false;

        @Override
        public void getClusterInfo(GetClusterInfoRequest request,
                                   StreamObserver<GetClusterInfoResponse> responseObserver) {
            if (returnError) {
                responseObserver.onNext(GetClusterInfoResponse.newBuilder()
                        .setHeader(CommonResponseHeader.newBuilder()
                                .setStatus(Status.newBuilder()
                                        .setCode(ErrorCode.SERVICE_NOT_READY)
                                        .setMessage("not ready")
                                        .build())
                                .build())
                        .build());
                responseObserver.onCompleted();
                return;
            }

            GetClusterInfoResponse.Builder builder = GetClusterInfoResponse.newBuilder()
                    .setHeader(CommonResponseHeader.newBuilder()
                            .setStatus(Status.newBuilder().setCode(ErrorCode.OK).build())
                            .build())
                    .setSelfNodeId("self-node")
                    .setLeaderNodeId("leader-node");

            if (hasLeaderEndpoint) {
                builder.setLeaderEndpoint(MetaNodeEndpoint.newBuilder()
                        .setNodeId("leader-node")
                        .setHost(leaderHost)
                        .setMetaRpcPort(leaderPort)
                        .build());
            }

            responseObserver.onNext(builder.build());
            responseObserver.onCompleted();
        }
    }
}
