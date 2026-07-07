package com.alibaba.tair.kvcm.client;

import com.alibaba.tair.kvcm.client.exception.KvcmException;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.StatusRuntimeException;
import kv_cache_manager.proto.meta.MetaServiceGrpc;
import kv_cache_manager.proto.meta.MetaServiceOuterClass.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.TimeUnit;

/**
 * gRPC-based MetaClient implementation.
 * Uses blocking stubs with per-call deadlines.
 */
public class GrpcMetaClient implements MetaClient {

    private static final Logger LOG = LoggerFactory.getLogger(GrpcMetaClient.class);

    private final ManagedChannel channel;
    private final MetaServiceGrpc.MetaServiceBlockingStub stub;
    private final int callTimeoutMs;

    public GrpcMetaClient(String host, int port, int callTimeoutMs) {
        this.callTimeoutMs = callTimeoutMs;
        // C3 fix: Add gRPC-level retry policy matching C++ client behavior
        java.util.Map<String, Object> retryPolicy = new java.util.HashMap<>();
        // gRPC service config expects JSON number types as Java Double, not Integer
        retryPolicy.put("maxAttempts", 3.0);
        retryPolicy.put("initialBackoff", "0.1s");
        retryPolicy.put("maxBackoff", "1s");
        retryPolicy.put("backoffMultiplier", 1.5);
        retryPolicy.put("retryableStatusCodes", java.util.Arrays.asList("UNAVAILABLE"));

        java.util.Map<String, Object> name = new java.util.HashMap<>();
        name.put("service", "kv_cache_manager.proto.meta.MetaService");

        java.util.Map<String, Object> methodConfig = new java.util.HashMap<>();
        methodConfig.put("name", java.util.Arrays.asList(name));
        methodConfig.put("waitForReady", true);
        methodConfig.put("timeout", (callTimeoutMs / 1000.0) + "s");
        methodConfig.put("retryPolicy", retryPolicy);

        java.util.Map<String, Object> serviceConfig = new java.util.HashMap<>();
        serviceConfig.put("methodConfig", java.util.Arrays.asList(methodConfig));

        this.channel = ManagedChannelBuilder.forAddress(host, port)
                .usePlaintext()
                .maxInboundMessageSize(Integer.MAX_VALUE)
                .defaultServiceConfig(serviceConfig)
                .enableRetry()
                .keepAliveTime(10, TimeUnit.SECONDS)
                .keepAliveTimeout(10, TimeUnit.SECONDS)
                .keepAliveWithoutCalls(true)
                .build();
        this.stub = MetaServiceGrpc.newBlockingStub(channel);
    }

    GrpcMetaClient(MetaClientConfig config) {
        this(config.getSeedAddress(), config.getGrpcPort(), config.getCallTimeoutMs());
    }

    /** Package-private: for testing with InProcess channels. */
    GrpcMetaClient(ManagedChannel channel, int callTimeoutMs) {
        this.channel = channel;
        this.callTimeoutMs = callTimeoutMs;
        this.stub = MetaServiceGrpc.newBlockingStub(channel);
    }

    private MetaServiceGrpc.MetaServiceBlockingStub withTimeout() {
        return stub.withDeadlineAfter(callTimeoutMs, TimeUnit.MILLISECONDS);
    }

    // --- Instance management ---

    @Override
    public RegisterInstanceResponse registerInstance(RegisterInstanceRequest request) {
        RegisterInstanceResponse response = callGrpc("registerInstance",
                () -> withTimeout().registerInstance(request));
        ResponseChecker.check(response);
        return response;
    }

    @Override
    public GetInstanceInfoResponse getInstanceInfo(GetInstanceInfoRequest request) {
        GetInstanceInfoResponse response = callGrpc("getInstanceInfo",
                () -> withTimeout().getInstanceInfo(request));
        ResponseChecker.check(response);
        return response;
    }

    // --- CacheAware queries ---

    @Override
    public GetCacheLocationResponse getCacheLocation(GetCacheLocationRequest request) {
        GetCacheLocationResponse response = callGrpc("getCacheLocation",
                () -> withTimeout().getCacheLocation(request));
        ResponseChecker.check(response);
        return response;
    }

    @Override
    public GetCacheLocationsByBackendResponse getCacheLocationsByBackend(GetCacheLocationsByBackendRequest request) {
        GetCacheLocationsByBackendResponse response = callGrpc("getCacheLocationsByBackend",
                () -> withTimeout().getCacheLocationsByBackend(request));
        ResponseChecker.check(response);
        return response;
    }

    @Override
    public GetCacheLocationLenResponse getCacheLocationLen(GetCacheLocationLenRequest request) {
        GetCacheLocationLenResponse response = callGrpc("getCacheLocationLen",
                () -> withTimeout().getCacheLocationLen(request));
        ResponseChecker.check(response);
        return response;
    }

    @Override
    public GetCacheMetaResponse getCacheMeta(GetCacheMetaRequest request) {
        GetCacheMetaResponse response = callGrpc("getCacheMeta",
                () -> withTimeout().getCacheMeta(request));
        ResponseChecker.check(response);
        return response;
    }

    // --- Write flow ---

    @Override
    public StartWriteCacheResponse startWriteCache(StartWriteCacheRequest request) {
        StartWriteCacheResponse response = callGrpc("startWriteCache",
                () -> withTimeout().startWriteCache(request));
        ResponseChecker.check(response);
        return response;
    }

    @Override
    public CommonResponse finishWriteCache(FinishWriteCacheRequest request) {
        CommonResponse response = callGrpc("finishWriteCache",
                () -> withTimeout().finishWriteCache(request));
        ResponseChecker.check(response);
        return response;
    }

    // --- Delete / trim ---

    @Override
    public CommonResponse removeCache(RemoveCacheRequest request) {
        CommonResponse response = callGrpc("removeCache",
                () -> withTimeout().removeCache(request));
        ResponseChecker.check(response);
        return response;
    }

    @Override
    public CommonResponse trimCache(TrimCacheRequest request) {
        CommonResponse response = callGrpc("trimCache",
                () -> withTimeout().trimCache(request));
        ResponseChecker.check(response);
        return response;
    }

    // --- Reporting ---

    @Override
    public ReportEventResponse reportEvent(ReportEventRequest request) {
        ReportEventResponse response = callGrpc("reportEvent",
                () -> withTimeout().reportEvent(request));
        ResponseChecker.check(response);
        return response;
    }

    // --- Cluster info ---

    @Override
    public GetClusterInfoResponse getClusterInfo(GetClusterInfoRequest request) {
        GetClusterInfoResponse response = callGrpc("getClusterInfo",
                () -> withTimeout().getClusterInfo(request));
        ResponseChecker.check(response);
        return response;
    }

    // --- Lifecycle ---

    @Override
    public void close() throws Exception {
        channel.shutdown();
        if (!channel.awaitTermination(5, TimeUnit.SECONDS)) {
            channel.shutdownNow();
        }
    }

    // --- Internal ---

    @FunctionalInterface
    private interface GrpcCall<T> {
        T execute();
    }

    private <T> T callGrpc(String rpcName, GrpcCall<T> call) {
        try {
            return call.execute();
        } catch (StatusRuntimeException e) {
            // M4 fix: Better error mapping - distinguish timeout from transient failures
            ErrorCode errorCode;
            String message;
            io.grpc.Status.Code grpcCode = e.getStatus().getCode();

            if (grpcCode == io.grpc.Status.Code.DEADLINE_EXCEEDED) {
                errorCode = ErrorCode.IO_ERROR;
                message = "gRPC call " + rpcName + " timed out: " + e.getStatus();
            } else if (grpcCode == io.grpc.Status.Code.UNAVAILABLE) {
                errorCode = ErrorCode.IO_ERROR;
                message = "gRPC call " + rpcName + " unavailable (server may be down): " + e.getStatus();
            } else if (grpcCode == io.grpc.Status.Code.INVALID_ARGUMENT) {
                errorCode = ErrorCode.INVALID_ARGUMENT;
                message = "gRPC call " + rpcName + " invalid argument: " + e.getStatus();
            } else {
                errorCode = ErrorCode.IO_ERROR;
                message = "gRPC call " + rpcName + " failed: " + e.getStatus();
            }
            throw new KvcmException(errorCode, message, e);
        }
    }
}
