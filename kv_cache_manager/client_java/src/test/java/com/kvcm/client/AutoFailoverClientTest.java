package com.kvcm.client;

import com.kvcm.client.exception.ServerNotLeaderException;
import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import io.grpc.stub.StreamObserver;
import kv_cache_manager.proto.meta.MetaServiceGrpc;
import kv_cache_manager.proto.meta.MetaServiceOuterClass.*;
import org.junit.jupiter.api.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for AutoFailoverClient failover logic.
 * Uses InProcess gRPC server to test SERVER_NOT_LEADER retry behavior.
 */
class AutoFailoverClientTest {

    private Server server;
    private FailoverMockService mockService;
    private String serverName;

    @BeforeEach
    void setUp() throws Exception {
        serverName = InProcessServerBuilder.generateName();
        mockService = new FailoverMockService();
        server = InProcessServerBuilder.forName(serverName)
                .directExecutor()
                .addService(mockService)
                .build()
                .start();

        // We test GrpcMetaClient directly with InProcess channel for failover scenarios
    }

    @AfterEach
    void tearDown() {
        server.shutdownNow();
    }

    @Test
    void testGrpcDirect_success() throws Exception {
        ManagedChannel channel = InProcessChannelBuilder
                .forName(serverName)
                .directExecutor()
                .build();
        GrpcMetaClient client = new GrpcMetaClient(channel, 5000);

        GetCacheLocationResponse resp = client.getCacheLocation(
                GetCacheLocationRequest.newBuilder()
                        .setTraceId("test")
                        .setInstanceId("inst1")
                        .addBlockKeys(100L)
                        .build());
        assertNotNull(resp);
        assertEquals(1, resp.getLocationsCount());

        client.close();
    }

    @Test
    void testServerNotLeader_throwsCorrectException() throws Exception {
        ManagedChannel channel = InProcessChannelBuilder
                .forName(serverName)
                .directExecutor()
                .build();
        GrpcMetaClient client = new GrpcMetaClient(channel, 5000);

        mockService.notLeaderOnce = true;

        assertThrows(ServerNotLeaderException.class, () ->
                client.getCacheLocation(GetCacheLocationRequest.newBuilder()
                        .setTraceId("not-leader-test")
                        .setInstanceId("inst1")
                        .build()));

        client.close();
    }

    @Test
    void testRetryAfterNotLeader() throws Exception {
        ManagedChannel channel = InProcessChannelBuilder
                .forName(serverName)
                .directExecutor()
                .build();
        GrpcMetaClient client = new GrpcMetaClient(channel, 5000);

        // First call returns SERVER_NOT_LEADER, second call succeeds
        mockService.notLeaderOnce = true;

        // First call fails
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
