package com.kvcm.client;

import com.kvcm.client.exception.KvcmException;
import com.kvcm.client.exception.ServerNotLeaderException;
import io.grpc.*;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import io.grpc.stub.StreamObserver;
import kv_cache_manager.proto.meta.MetaServiceGrpc;
import kv_cache_manager.proto.meta.MetaServiceOuterClass.*;
import kv_cache_manager.proto.meta.MetaServiceOuterClass.Status;
import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import org.junit.jupiter.api.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * End-to-end tests for AutoFailoverClient failover logic.
 */
class AutoFailoverClientTest {

    private Server grpcServer;
    private MockWebServer httpServer;
    private String grpcServerName;
    private FailoverMockService mockGrpcService;

    @BeforeEach
    void setUp() throws Exception {
        grpcServerName = InProcessServerBuilder.generateName();
        mockGrpcService = new FailoverMockService();
        grpcServer = InProcessServerBuilder.forName(grpcServerName)
                .directExecutor()
                .addService(mockGrpcService)
                .build()
                .start();

        httpServer = new MockWebServer();
        httpServer.start();
    }

    @AfterEach
    void tearDown() throws Exception {
        grpcServer.shutdownNow();
        httpServer.shutdown();
    }

    // --- Helper: build AutoFailoverClient with InProcess gRPC ---

    private AutoFailoverClient buildClient() throws Exception {
        ManagedChannel channel = InProcessChannelBuilder.forName(grpcServerName)
                .directExecutor()
                .build();
        GrpcMetaClient grpcClient = new GrpcMetaClient(channel, 5000);
        HttpMetaClient httpClient = new HttpMetaClient(httpServer.getHostName(), httpServer.getPort(), 5000);

        MetaClientConfig config = MetaClientConfig.builder("dummy")
                .autoDiscoverLeader(false)
                .httpPort(httpServer.getPort())
                .build();
        return new AutoFailoverClient(config, grpcClient, httpClient);
    }

    private AutoFailoverClient buildClientGrpcOnly() throws Exception {
        ManagedChannel channel = InProcessChannelBuilder.forName(grpcServerName)
                .directExecutor()
                .build();
        GrpcMetaClient grpcClient = new GrpcMetaClient(channel, 5000);

        MetaClientConfig config = MetaClientConfig.builder("dummy")
                .autoDiscoverLeader(false)
                .build();
        return new AutoFailoverClient(config, grpcClient, null);
    }

    // --- Tests ---

    @Test
    void testGrpcSuccess_noFailover() throws Exception {
        AutoFailoverClient client = buildClient();

        GetCacheLocationResponse resp = client.getCacheLocation(
                GetCacheLocationRequest.newBuilder()
                        .setTraceId("test")
                        .setInstanceId("inst1")
                        .build());
        assertNotNull(resp);
        assertEquals(1, resp.getLocationsCount());
        assertEquals(StorageType.ST_3FS, resp.getLocations(0).getType());

        client.close();
    }

    @Test
    void testServerNotLeader_throwsServerNotLeaderException() throws Exception {
        // With leaderRetryCount=0, no retry → immediately throws
        ManagedChannel channel = InProcessChannelBuilder.forName(grpcServerName)
                .directExecutor()
                .build();
        GrpcMetaClient grpcClient = new GrpcMetaClient(channel, 5000);
        MetaClientConfig config = MetaClientConfig.builder("dummy")
                .autoDiscoverLeader(false)
                .leaderRetryCount(0)
                .build();
        AutoFailoverClient client = new AutoFailoverClient(config, grpcClient, null);
        mockGrpcService.notLeaderOnce = true;

        assertThrows(ServerNotLeaderException.class, () ->
                client.getCacheLocation(GetCacheLocationRequest.newBuilder()
                        .setTraceId("not-leader")
                        .setInstanceId("inst1")
                        .build()));

        client.close();
    }

    @Test
    void testGrpcFails_httpFallbackSucceeds() throws Exception {
        // gRPC will fail with IO_ERROR (simulate by shutting down server mid-call)
        // Instead, use a separate gRPC server that's already shut down
        String deadServerName = InProcessServerBuilder.generateName();
        Server deadServer = InProcessServerBuilder.forName(deadServerName)
                .directExecutor()
                .addService(mockGrpcService)
                .build()
                .start();
        deadServer.shutdownNow(); // kill it immediately

        ManagedChannel deadChannel = InProcessChannelBuilder.forName(deadServerName)
                .directExecutor()
                .build();
        GrpcMetaClient deadGrpc = new GrpcMetaClient(deadChannel, 1000); // 1s timeout

        // HTTP succeeds
        String okJson = "{\"header\":{\"status\":{\"code\":\"OK\"},\"request_id\":\"\",\"tracer_result\":\"\"},"
                + "\"locations\":[{\"type\":\"ST_3FS\",\"spec_size\":0,\"location_specs\":[{\"name\":\"tp0\",\"uri\":\"3fs://fallback/path\"}]}]}";
        httpServer.enqueue(new MockResponse().setBody(okJson).setHeader("Content-Type", "application/json"));

        HttpMetaClient httpClient = new HttpMetaClient(httpServer.getHostName(), httpServer.getPort(), 5000);

        MetaClientConfig config = MetaClientConfig.builder("dummy")
                .autoDiscoverLeader(false)
                .httpPort(httpServer.getPort())
                .build();
        AutoFailoverClient client = new AutoFailoverClient(config, deadGrpc, httpClient);

        // Should fall back to HTTP
        GetCacheLocationResponse resp = client.getCacheLocation(
                GetCacheLocationRequest.newBuilder()
                        .setTraceId("fallback-test")
                        .setInstanceId("inst1")
                        .build());
        assertNotNull(resp);
        assertEquals(1, resp.getLocationsCount());
        assertEquals("3fs://fallback/path", resp.getLocations(0).getLocationSpecs(0).getUri());

        client.close();
    }

    @Test
    void testBothTransportsFail_throwsCombinedException() throws Exception {
        // gRPC fails (dead server)
        String deadServerName = InProcessServerBuilder.generateName();
        Server deadServer = InProcessServerBuilder.forName(deadServerName)
                .directExecutor()
                .addService(mockGrpcService)
                .build()
                .start();
        deadServer.shutdownNow();

        ManagedChannel deadChannel = InProcessChannelBuilder.forName(deadServerName)
                .directExecutor()
                .build();
        GrpcMetaClient deadGrpc = new GrpcMetaClient(deadChannel, 1000);

        // HTTP also fails
        httpServer.enqueue(new MockResponse().setResponseCode(500).setBody("Internal Server Error"));
        HttpMetaClient httpClient = new HttpMetaClient(httpServer.getHostName(), httpServer.getPort(), 5000);

        MetaClientConfig config = MetaClientConfig.builder("dummy")
                .autoDiscoverLeader(false)
                .httpPort(httpServer.getPort())
                .build();
        AutoFailoverClient client = new AutoFailoverClient(config, deadGrpc, httpClient);

        KvcmException ex = assertThrows(KvcmException.class, () ->
                client.getCacheLocation(GetCacheLocationRequest.newBuilder()
                        .setTraceId("both-fail")
                        .setInstanceId("inst1")
                        .build()));

        assertEquals(ErrorCode.IO_ERROR, ex.getErrorCode());
        assertTrue(ex.getMessage().contains("All transports failed"));

        client.close();
    }

    @Test
    void testClosedClient_throwsIllegalStateException() throws Exception {
        AutoFailoverClient client = buildClient();
        client.close();

        IllegalStateException ex = assertThrows(IllegalStateException.class, () ->
                client.getCacheLocation(GetCacheLocationRequest.newBuilder()
                        .setTraceId("closed")
                        .setInstanceId("inst1")
                        .build()));
        assertTrue(ex.getMessage().contains("closed"));
    }

    @Test
    void testDoubleClose_noException() throws Exception {
        AutoFailoverClient client = buildClient();
        client.close();
        client.close(); // should not throw
    }

    @Test
    void testRetryAfterServerNotLeader() throws Exception {
        // With default leaderRetryCount=1: first call fails, retry succeeds
        AutoFailoverClient client = buildClientGrpcOnly();
        mockGrpcService.notLeaderOnce = true;

        // First call: notLeaderOnce returns SERVER_NOT_LEADER, retriesLeft=1 → retry
        // Second call: notLeaderOnce consumed → OK
        // Result: should succeed on retry
        GetCacheLocationResponse resp = client.getCacheLocation(
                GetCacheLocationRequest.newBuilder()
                        .setTraceId("retry-test")
                        .setInstanceId("inst1")
                        .build());
        assertNotNull(resp);
        assertEquals(1, resp.getLocationsCount());

        client.close();
    }

    @Test
    void testAllRpcMethods_routeCorrectly() throws Exception {
        AutoFailoverClient client = buildClient();

        // Test each RPC method goes through withFailover correctly
        assertNotNull(client.registerInstance(RegisterInstanceRequest.newBuilder().setTraceId("t").build()));
        assertNotNull(client.getInstanceInfo(GetInstanceInfoRequest.newBuilder().setTraceId("t").build()));
        assertNotNull(client.getCacheLocation(GetCacheLocationRequest.newBuilder().setTraceId("t").build()));
        assertNotNull(client.getCacheLocationsByBackend(GetCacheLocationsByBackendRequest.newBuilder().setTraceId("t").build()));
        assertNotNull(client.getCacheLocationLen(GetCacheLocationLenRequest.newBuilder().setTraceId("t").build()));
        assertNotNull(client.getCacheMeta(GetCacheMetaRequest.newBuilder().setTraceId("t").build()));
        assertNotNull(client.startWriteCache(StartWriteCacheRequest.newBuilder().setTraceId("t").build()));
        assertNotNull(client.finishWriteCache(FinishWriteCacheRequest.newBuilder().setTraceId("t").build()));
        assertNotNull(client.removeCache(RemoveCacheRequest.newBuilder().setTraceId("t").build()));
        assertNotNull(client.trimCache(TrimCacheRequest.newBuilder().setTraceId("t").build()));
        assertNotNull(client.reportEvent(ReportEventRequest.newBuilder().setTraceId("t").build()));
        assertNotNull(client.getClusterInfo(GetClusterInfoRequest.newBuilder().setTraceId("t").build()));

        client.close();
    }

    // --- Mock service ---

    static class FailoverMockService extends MetaServiceGrpc.MetaServiceImplBase {
        volatile boolean notLeaderOnce = false;

        private CommonResponseHeader okHeader() {
            return CommonResponseHeader.newBuilder()
                    .setStatus(Status.newBuilder().setCode(ErrorCode.OK).build())
                    .build();
        }

        private CommonResponseHeader notLeaderHeader() {
            return CommonResponseHeader.newBuilder()
                    .setStatus(Status.newBuilder()
                            .setCode(ErrorCode.SERVER_NOT_LEADER)
                            .setMessage("not leader")
                            .build())
                    .build();
        }

        @Override
        public void registerInstance(RegisterInstanceRequest request, StreamObserver<RegisterInstanceResponse> obs) {
            obs.onNext(RegisterInstanceResponse.newBuilder().setHeader(okHeader()).setStorageConfigs("cfg").build());
            obs.onCompleted();
        }

        @Override
        public void getInstanceInfo(GetInstanceInfoRequest request, StreamObserver<GetInstanceInfoResponse> obs) {
            obs.onNext(GetInstanceInfoResponse.newBuilder().setHeader(okHeader()).build());
            obs.onCompleted();
        }

        @Override
        public void getCacheLocation(GetCacheLocationRequest request, StreamObserver<GetCacheLocationResponse> obs) {
            if (notLeaderOnce) {
                notLeaderOnce = false;
                obs.onNext(GetCacheLocationResponse.newBuilder().setHeader(notLeaderHeader()).build());
                obs.onCompleted();
                return;
            }
            obs.onNext(GetCacheLocationResponse.newBuilder()
                    .setHeader(okHeader())
                    .addLocations(CacheLocation.newBuilder()
                            .setType(StorageType.ST_3FS)
                            .addLocationSpecs(LocationSpec.newBuilder().setName("tp0").setUri("3fs://test/path").build())
                            .build())
                    .build());
            obs.onCompleted();
        }

        @Override
        public void getCacheLocationsByBackend(GetCacheLocationsByBackendRequest request, StreamObserver<GetCacheLocationsByBackendResponse> obs) {
            obs.onNext(GetCacheLocationsByBackendResponse.newBuilder().setHeader(okHeader()).build());
            obs.onCompleted();
        }

        @Override
        public void getCacheLocationLen(GetCacheLocationLenRequest request, StreamObserver<GetCacheLocationLenResponse> obs) {
            obs.onNext(GetCacheLocationLenResponse.newBuilder().setHeader(okHeader()).setCacheLocationLen(42).build());
            obs.onCompleted();
        }

        @Override
        public void getCacheMeta(GetCacheMetaRequest request, StreamObserver<GetCacheMetaResponse> obs) {
            obs.onNext(GetCacheMetaResponse.newBuilder().setHeader(okHeader()).build());
            obs.onCompleted();
        }

        @Override
        public void startWriteCache(StartWriteCacheRequest request, StreamObserver<StartWriteCacheResponse> obs) {
            obs.onNext(StartWriteCacheResponse.newBuilder().setHeader(okHeader()).setWriteSessionId("s1").build());
            obs.onCompleted();
        }

        @Override
        public void finishWriteCache(FinishWriteCacheRequest request, StreamObserver<CommonResponse> obs) {
            obs.onNext(CommonResponse.newBuilder().setHeader(okHeader()).build());
            obs.onCompleted();
        }

        @Override
        public void removeCache(RemoveCacheRequest request, StreamObserver<CommonResponse> obs) {
            obs.onNext(CommonResponse.newBuilder().setHeader(okHeader()).build());
            obs.onCompleted();
        }

        @Override
        public void trimCache(TrimCacheRequest request, StreamObserver<CommonResponse> obs) {
            obs.onNext(CommonResponse.newBuilder().setHeader(okHeader()).build());
            obs.onCompleted();
        }

        @Override
        public void reportEvent(ReportEventRequest request, StreamObserver<ReportEventResponse> obs) {
            obs.onNext(ReportEventResponse.newBuilder().setHeader(okHeader()).build());
            obs.onCompleted();
        }

        @Override
        public void getClusterInfo(GetClusterInfoRequest request, StreamObserver<GetClusterInfoResponse> obs) {
            obs.onNext(GetClusterInfoResponse.newBuilder()
                    .setHeader(okHeader())
                    .setSelfNodeId("self")
                    .setLeaderNodeId("leader")
                    .setLeaderEndpoint(MetaNodeEndpoint.newBuilder()
                            .setHost("10.0.0.1")
                            .setMetaRpcPort(6381)
                            .build())
                    .build());
            obs.onCompleted();
        }
    }
}
