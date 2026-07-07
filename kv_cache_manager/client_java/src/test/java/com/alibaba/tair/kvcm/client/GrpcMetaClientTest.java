package com.alibaba.tair.kvcm.client;

import com.alibaba.tair.kvcm.client.exception.KvcmException;
import com.alibaba.tair.kvcm.client.exception.ServerNotLeaderException;
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
 * Tests for GrpcMetaClient using InProcess gRPC server.
 */
class GrpcMetaClientTest {

    private Server server;
    private GrpcMetaClient client;
    private MockMetaServiceImpl mockService;

    @BeforeEach
    void setUp() throws Exception {
        String serverName = InProcessServerBuilder.generateName();
        mockService = new MockMetaServiceImpl();
        server = InProcessServerBuilder.forName(serverName)
                .directExecutor()
                .addService(mockService)
                .build()
                .start();

        ManagedChannel channel = InProcessChannelBuilder.forName(serverName)
                .directExecutor()
                .build();
        client = new GrpcMetaClient(channel, 5000);
    }

    @AfterEach
    void tearDown() throws Exception {
        client.close();
        server.shutdownNow();
    }

    @Test
    void testGetCacheLocation_success() {
        GetCacheLocationRequest req = GetCacheLocationRequest.newBuilder()
                .setTraceId("test")
                .setInstanceId("inst1")
                .addBlockKeys(123L)
                .build();
        GetCacheLocationResponse resp = client.getCacheLocation(req);
        assertNotNull(resp);
        assertTrue(mockService.lastTraceId.equals("test"));
    }

    @Test
    void testGetCacheLocationsByBackend_success() {
        GetCacheLocationsByBackendRequest req = GetCacheLocationsByBackendRequest.newBuilder()
                .setTraceId("test-backend")
                .setInstanceId("inst1")
                .build();
        GetCacheLocationsByBackendResponse resp = client.getCacheLocationsByBackend(req);
        assertNotNull(resp);
    }

    @Test
    void testGetCacheLocationLen_success() {
        GetCacheLocationLenRequest req = GetCacheLocationLenRequest.newBuilder()
                .setTraceId("test-len")
                .setInstanceId("inst1")
                .build();
        GetCacheLocationLenResponse resp = client.getCacheLocationLen(req);
        assertNotNull(resp);
        assertEquals(42L, resp.getCacheLocationLen());
    }

    @Test
    void testGetCacheMeta_success() {
        GetCacheMetaRequest req = GetCacheMetaRequest.newBuilder()
                .setTraceId("test-meta")
                .setInstanceId("inst1")
                .build();
        GetCacheMetaResponse resp = client.getCacheMeta(req);
        assertNotNull(resp);
    }

    @Test
    void testRegisterInstance_success() {
        RegisterInstanceRequest req = RegisterInstanceRequest.newBuilder()
                .setTraceId("test-reg")
                .setInstanceId("inst1")
                .setInstanceGroup("group1")
                .build();
        RegisterInstanceResponse resp = client.registerInstance(req);
        assertNotNull(resp);
        assertEquals("test-storage-config", resp.getStorageConfigs());
    }

    @Test
    void testGetInstanceInfo_success() {
        GetInstanceInfoRequest req = GetInstanceInfoRequest.newBuilder()
                .setTraceId("test-info")
                .setInstanceId("inst1")
                .build();
        GetInstanceInfoResponse resp = client.getInstanceInfo(req);
        assertNotNull(resp);
    }

    @Test
    void testStartWriteCache_success() {
        StartWriteCacheRequest req = StartWriteCacheRequest.newBuilder()
                .setTraceId("test-start")
                .setInstanceId("inst1")
                .build();
        StartWriteCacheResponse resp = client.startWriteCache(req);
        assertNotNull(resp);
    }

    @Test
    void testFinishWriteCache_success() {
        FinishWriteCacheRequest req = FinishWriteCacheRequest.newBuilder()
                .setTraceId("test-finish")
                .setInstanceId("inst1")
                .setWriteSessionId("session1")
                .build();
        CommonResponse resp = client.finishWriteCache(req);
        assertNotNull(resp);
    }

    @Test
    void testRemoveCache_success() {
        RemoveCacheRequest req = RemoveCacheRequest.newBuilder()
                .setTraceId("test-remove")
                .setInstanceId("inst1")
                .build();
        CommonResponse resp = client.removeCache(req);
        assertNotNull(resp);
    }

    @Test
    void testTrimCache_success() {
        TrimCacheRequest req = TrimCacheRequest.newBuilder()
                .setTraceId("test-trim")
                .setInstanceId("inst1")
                .build();
        CommonResponse resp = client.trimCache(req);
        assertNotNull(resp);
    }

    @Test
    void testReportEvent_success() {
        ReportEventRequest req = ReportEventRequest.newBuilder()
                .setTraceId("test-report")
                .setInstanceId("inst1")
                .setHostIpPort("10.0.0.1:8080")
                .build();
        ReportEventResponse resp = client.reportEvent(req);
        assertNotNull(resp);
    }

    @Test
    void testGetClusterInfo_success() {
        GetClusterInfoRequest req = GetClusterInfoRequest.newBuilder()
                .setTraceId("test-cluster")
                .setInstanceId("inst1")
                .build();
        GetClusterInfoResponse resp = client.getClusterInfo(req);
        assertNotNull(resp);
        assertEquals("leader-node", resp.getLeaderNodeId());
    }

    @Test
    void testServerNotLeader_throwsServerNotLeaderException() {
        mockService.nextError = ErrorCode.SERVER_NOT_LEADER;
        GetCacheLocationRequest req = GetCacheLocationRequest.newBuilder()
                .setTraceId("test-not-leader")
                .setInstanceId("inst1")
                .build();
        ServerNotLeaderException ex = assertThrows(ServerNotLeaderException.class,
                () -> client.getCacheLocation(req));
        assertEquals(ErrorCode.SERVER_NOT_LEADER, ex.getErrorCode());
    }

    @Test
    void testInstanceNotExist_throwsKvcmException() {
        mockService.nextError = ErrorCode.INSTANCE_NOT_EXIST;
        GetInstanceInfoRequest req = GetInstanceInfoRequest.newBuilder()
                .setTraceId("test-not-exist")
                .setInstanceId("nonexistent")
                .build();
        KvcmException ex = assertThrows(KvcmException.class,
                () -> client.getInstanceInfo(req));
        assertEquals(ErrorCode.INSTANCE_NOT_EXIST, ex.getErrorCode());
    }

    // --- Mock service ---

    static class MockMetaServiceImpl extends MetaServiceGrpc.MetaServiceImplBase {
        volatile ErrorCode nextError = null;
        volatile String lastTraceId = "";

        private CommonResponseHeader okHeader() {
            return CommonResponseHeader.newBuilder()
                    .setStatus(Status.newBuilder().setCode(ErrorCode.OK).build())
                    .build();
        }

        private CommonResponseHeader errorHeader(ErrorCode code) {
            return CommonResponseHeader.newBuilder()
                    .setStatus(Status.newBuilder().setCode(code).setMessage("error: " + code).build())
                    .build();
        }

        private CommonResponseHeader pickHeader(String traceId) {
            lastTraceId = traceId;
            if (nextError != null) {
                ErrorCode err = nextError;
                nextError = null;
                return errorHeader(err);
            }
            return okHeader();
        }

        @Override
        public void getCacheLocation(GetCacheLocationRequest request,
                                     StreamObserver<GetCacheLocationResponse> responseObserver) {
            responseObserver.onNext(GetCacheLocationResponse.newBuilder()
                    .setHeader(pickHeader(request.getTraceId())).build());
            responseObserver.onCompleted();
        }

        @Override
        public void getCacheLocationsByBackend(GetCacheLocationsByBackendRequest request,
                                               StreamObserver<GetCacheLocationsByBackendResponse> responseObserver) {
            responseObserver.onNext(GetCacheLocationsByBackendResponse.newBuilder()
                    .setHeader(pickHeader(request.getTraceId())).build());
            responseObserver.onCompleted();
        }

        @Override
        public void getCacheLocationLen(GetCacheLocationLenRequest request,
                                        StreamObserver<GetCacheLocationLenResponse> responseObserver) {
            responseObserver.onNext(GetCacheLocationLenResponse.newBuilder()
                    .setHeader(pickHeader(request.getTraceId()))
                    .setCacheLocationLen(42L)
                    .build());
            responseObserver.onCompleted();
        }

        @Override
        public void getCacheMeta(GetCacheMetaRequest request,
                                 StreamObserver<GetCacheMetaResponse> responseObserver) {
            responseObserver.onNext(GetCacheMetaResponse.newBuilder()
                    .setHeader(pickHeader(request.getTraceId())).build());
            responseObserver.onCompleted();
        }

        @Override
        public void registerInstance(RegisterInstanceRequest request,
                                     StreamObserver<RegisterInstanceResponse> responseObserver) {
            responseObserver.onNext(RegisterInstanceResponse.newBuilder()
                    .setHeader(pickHeader(request.getTraceId()))
                    .setStorageConfigs("test-storage-config")
                    .build());
            responseObserver.onCompleted();
        }

        @Override
        public void getInstanceInfo(GetInstanceInfoRequest request,
                                    StreamObserver<GetInstanceInfoResponse> responseObserver) {
            responseObserver.onNext(GetInstanceInfoResponse.newBuilder()
                    .setHeader(pickHeader(request.getTraceId()))
                    .setInstanceGroup("test-group")
                    .build());
            responseObserver.onCompleted();
        }

        @Override
        public void startWriteCache(StartWriteCacheRequest request,
                                    StreamObserver<StartWriteCacheResponse> responseObserver) {
            responseObserver.onNext(StartWriteCacheResponse.newBuilder()
                    .setHeader(pickHeader(request.getTraceId()))
                    .setWriteSessionId("session-1")
                    .build());
            responseObserver.onCompleted();
        }

        @Override
        public void finishWriteCache(FinishWriteCacheRequest request,
                                     StreamObserver<CommonResponse> responseObserver) {
            responseObserver.onNext(CommonResponse.newBuilder()
                    .setHeader(pickHeader(request.getTraceId())).build());
            responseObserver.onCompleted();
        }

        @Override
        public void removeCache(RemoveCacheRequest request,
                                StreamObserver<CommonResponse> responseObserver) {
            responseObserver.onNext(CommonResponse.newBuilder()
                    .setHeader(pickHeader(request.getTraceId())).build());
            responseObserver.onCompleted();
        }

        @Override
        public void trimCache(TrimCacheRequest request,
                              StreamObserver<CommonResponse> responseObserver) {
            responseObserver.onNext(CommonResponse.newBuilder()
                    .setHeader(pickHeader(request.getTraceId())).build());
            responseObserver.onCompleted();
        }

        @Override
        public void reportEvent(ReportEventRequest request,
                                StreamObserver<ReportEventResponse> responseObserver) {
            responseObserver.onNext(ReportEventResponse.newBuilder()
                    .setHeader(pickHeader(request.getTraceId())).build());
            responseObserver.onCompleted();
        }

        @Override
        public void getClusterInfo(GetClusterInfoRequest request,
                                   StreamObserver<GetClusterInfoResponse> responseObserver) {
            responseObserver.onNext(GetClusterInfoResponse.newBuilder()
                    .setHeader(pickHeader(request.getTraceId()))
                    .setLeaderNodeId("leader-node")
                    .setSelfNodeId("self-node")
                    .build());
            responseObserver.onCompleted();
        }
    }
}
