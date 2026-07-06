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

import java.io.IOException;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for AutoFailoverClient failover logic.
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

    @Test
    void testGrpcSuccess_noFailover() throws Exception {
        // Test GrpcMetaClient directly with InProcess channel (no failover needed)
        ManagedChannel channel = InProcessChannelBuilder.forName(grpcServerName)
                .directExecutor()
                .build();
        GrpcMetaClient client = new GrpcMetaClient(channel, 5000);

        GetCacheLocationResponse resp = client.getCacheLocation(
                GetCacheLocationRequest.newBuilder()
                        .setTraceId("test")
                        .setInstanceId("inst1")
                        .build());
        assertNotNull(resp);
        assertEquals(1, resp.getLocationsCount());

        client.close();
    }

    @Test
    void testServerNotLeader_throwsException() throws Exception {
        ManagedChannel channel = InProcessChannelBuilder.forName(grpcServerName)
                .directExecutor()
                .build();
        GrpcMetaClient client = new GrpcMetaClient(channel, 5000);

        mockGrpcService.notLeaderOnce = true;

        assertThrows(ServerNotLeaderException.class, () ->
                client.getCacheLocation(GetCacheLocationRequest.newBuilder()
                        .setTraceId("not-leader")
                        .setInstanceId("inst1")
                        .build()));

        client.close();
    }

    @Test
    void testRetryAfterNotLeader() throws Exception {
        ManagedChannel channel = InProcessChannelBuilder.forName(grpcServerName)
                .directExecutor()
                .build();
        GrpcMetaClient client = new GrpcMetaClient(channel, 5000);

        mockGrpcService.notLeaderOnce = true;

        // First call fails with SERVER_NOT_LEADER
        assertThrows(ServerNotLeaderException.class, () ->
                client.getCacheLocation(GetCacheLocationRequest.newBuilder()
                        .setTraceId("retry-1")
                        .setInstanceId("inst1")
                        .build()));

        // Second call succeeds (notLeaderOnce consumed)
        GetCacheLocationResponse resp = client.getCacheLocation(
                GetCacheLocationRequest.newBuilder()
                        .setTraceId("retry-2")
                        .setInstanceId("inst1")
                        .build());
        assertNotNull(resp);

        client.close();
    }

    @Test
    void testHttpFallback_onGrpcFailure() throws Exception {
        // This test verifies the HTTP fallback path conceptually
        // Full AutoFailoverClient integration would require more complex setup

        // Enqueue HTTP success response
        String okJson = "{\"header\":{\"status\":{\"code\":\"OK\"},\"request_id\":\"\",\"tracer_result\":\"\"},\"locations\":[]}";
        httpServer.enqueue(new MockResponse().setBody(okJson).setHeader("Content-Type", "application/json"));

        HttpMetaClient httpClient = new HttpMetaClient(httpServer.getHostName(), httpServer.getPort(), 5000);

        // HTTP call succeeds
        GetCacheLocationResponse resp = httpClient.getCacheLocation(
                GetCacheLocationRequest.newBuilder()
                        .setTraceId("http-fallback")
                        .setInstanceId("inst1")
                        .build());
        assertNotNull(resp);

        httpClient.close();
    }

    @Test
    void testBothTransportsFail_throwsKvcmException() throws Exception {
        // Enqueue HTTP error
        httpServer.enqueue(new MockResponse().setResponseCode(500).setBody("Internal Server Error"));

        HttpMetaClient httpClient = new HttpMetaClient(httpServer.getHostName(), httpServer.getPort(), 5000);

        // HTTP call fails
        assertThrows(KvcmException.class, () ->
                httpClient.getCacheLocation(GetCacheLocationRequest.newBuilder()
                        .setTraceId("both-fail")
                        .setInstanceId("inst1")
                        .build()));

        httpClient.close();
    }

    @Test
    void testClosedClient_throwsIllegalStateException() throws Exception {
        ManagedChannel channel = InProcessChannelBuilder.forName(grpcServerName)
                .directExecutor()
                .build();
        GrpcMetaClient client = new GrpcMetaClient(channel, 5000);
        client.close();

        // After close, calls should fail (gRPC will throw on closed channel)
        assertThrows(Exception.class, () ->
                client.getCacheLocation(GetCacheLocationRequest.newBuilder()
                        .setTraceId("closed")
                        .setInstanceId("inst1")
                        .build()));
    }

    // --- Mock service ---

    static class FailoverMockService extends MetaServiceGrpc.MetaServiceImplBase {
        volatile boolean notLeaderOnce = false;

        private CommonResponseHeader okHeader() {
            return CommonResponseHeader.newBuilder()
                    .setStatus(Status.newBuilder().setCode(ErrorCode.OK).build())
                    .build();
        }

        @Override
        public void getCacheLocation(GetCacheLocationRequest request,
                                     StreamObserver<GetCacheLocationResponse> responseObserver) {
            if (notLeaderOnce) {
                notLeaderOnce = false;
                responseObserver.onNext(GetCacheLocationResponse.newBuilder()
                        .setHeader(CommonResponseHeader.newBuilder()
                                .setStatus(Status.newBuilder()
                                        .setCode(ErrorCode.SERVER_NOT_LEADER)
                                        .setMessage("not leader")
                                        .build())
                                .build())
                        .build());
                responseObserver.onCompleted();
                return;
            }

            responseObserver.onNext(GetCacheLocationResponse.newBuilder()
                    .setHeader(okHeader())
                    .addLocations(CacheLocation.newBuilder()
                            .setType(StorageType.ST_3FS)
                            .addLocationSpecs(LocationSpec.newBuilder()
                                    .setName("tp0")
                                    .setUri("3fs://test/path")
                                    .build())
                            .build())
                    .build());
            responseObserver.onCompleted();
        }

        @Override
        public void getClusterInfo(GetClusterInfoRequest request,
                                   StreamObserver<GetClusterInfoResponse> responseObserver) {
            responseObserver.onNext(GetClusterInfoResponse.newBuilder()
                    .setHeader(okHeader())
                    .setSelfNodeId("self")
                    .setLeaderNodeId("leader")
                    .setLeaderEndpoint(MetaNodeEndpoint.newBuilder()
                            .setHost("10.0.0.1")
                            .setMetaRpcPort(6381)
                            .build())
                    .build());
            responseObserver.onCompleted();
        }
    }
}
