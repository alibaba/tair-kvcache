package com.kvcm.client;

import com.kvcm.client.exception.KvcmException;
import com.kvcm.client.exception.ServerNotLeaderException;
import kv_cache_manager.proto.meta.MetaServiceOuterClass.ErrorCode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Auto-failover MetaClient: gRPC primary → HTTP fallback → leader re-discovery + retry.
 * <p>
 * This is the recommended entry point for production use.
 * Create via {@link MetaClientFactory#create(MetaClientConfig)}.
 */
public class AutoFailoverClient implements MetaClient {

    private static final Logger LOG = LoggerFactory.getLogger(AutoFailoverClient.class);
    private static final long BASE_BACKOFF_MS = 5;

    private final MetaClientConfig config;
    private volatile GrpcMetaClient grpcClient;
    private final HttpMetaClient httpClient; // null if HTTP disabled
    private final LeaderDiscovery leaderDiscovery; // null if auto-discover disabled
    private final Object reconnectLock = new Object();

    AutoFailoverClient(MetaClientConfig config) {
        this.config = config;
        this.grpcClient = new GrpcMetaClient(config.getSeedAddress(), config.getGrpcPort(),
                config.getCallTimeoutMs());

        if (config.isHttpEnabled()) {
            this.httpClient = new HttpMetaClient(config.getSeedAddress(), config.getHttpPort(),
                    config.getCallTimeoutMs());
        } else {
            this.httpClient = null;
        }

        if (config.isAutoDiscoverLeader()) {
            this.leaderDiscovery = new LeaderDiscovery(
                    config.getSeedAddress(), config.getGrpcPort(),
                    config.getInstanceId(), config.getLeaderRefreshIntervalSeconds());
            leaderDiscovery.start();
            // If leader was discovered, reconnect to leader
            reconnectIfNeeded();
        } else {
            this.leaderDiscovery = null;
        }
    }

    // --- Instance management ---

    @Override
    public kv_cache_manager.proto.meta.MetaServiceOuterClass.RegisterInstanceResponse registerInstance(
            kv_cache_manager.proto.meta.MetaServiceOuterClass.RegisterInstanceRequest req) {
        return withFailover(c -> c.registerInstance(req));
    }

    @Override
    public kv_cache_manager.proto.meta.MetaServiceOuterClass.GetInstanceInfoResponse getInstanceInfo(
            kv_cache_manager.proto.meta.MetaServiceOuterClass.GetInstanceInfoRequest req) {
        return withFailover(c -> c.getInstanceInfo(req));
    }

    // --- CacheAware queries ---

    @Override
    public kv_cache_manager.proto.meta.MetaServiceOuterClass.GetCacheLocationResponse getCacheLocation(
            kv_cache_manager.proto.meta.MetaServiceOuterClass.GetCacheLocationRequest req) {
        return withFailover(c -> c.getCacheLocation(req));
    }

    @Override
    public kv_cache_manager.proto.meta.MetaServiceOuterClass.GetCacheLocationsByBackendResponse getCacheLocationsByBackend(
            kv_cache_manager.proto.meta.MetaServiceOuterClass.GetCacheLocationsByBackendRequest req) {
        return withFailover(c -> c.getCacheLocationsByBackend(req));
    }

    @Override
    public kv_cache_manager.proto.meta.MetaServiceOuterClass.GetCacheLocationLenResponse getCacheLocationLen(
            kv_cache_manager.proto.meta.MetaServiceOuterClass.GetCacheLocationLenRequest req) {
        return withFailover(c -> c.getCacheLocationLen(req));
    }

    @Override
    public kv_cache_manager.proto.meta.MetaServiceOuterClass.GetCacheMetaResponse getCacheMeta(
            kv_cache_manager.proto.meta.MetaServiceOuterClass.GetCacheMetaRequest req) {
        return withFailover(c -> c.getCacheMeta(req));
    }

    // --- Write flow ---

    @Override
    public kv_cache_manager.proto.meta.MetaServiceOuterClass.StartWriteCacheResponse startWriteCache(
            kv_cache_manager.proto.meta.MetaServiceOuterClass.StartWriteCacheRequest req) {
        return withFailover(c -> c.startWriteCache(req));
    }

    @Override
    public kv_cache_manager.proto.meta.MetaServiceOuterClass.CommonResponse finishWriteCache(
            kv_cache_manager.proto.meta.MetaServiceOuterClass.FinishWriteCacheRequest req) {
        return withFailover(c -> c.finishWriteCache(req));
    }

    // --- Delete / trim ---

    @Override
    public kv_cache_manager.proto.meta.MetaServiceOuterClass.CommonResponse removeCache(
            kv_cache_manager.proto.meta.MetaServiceOuterClass.RemoveCacheRequest req) {
        return withFailover(c -> c.removeCache(req));
    }

    @Override
    public kv_cache_manager.proto.meta.MetaServiceOuterClass.CommonResponse trimCache(
            kv_cache_manager.proto.meta.MetaServiceOuterClass.TrimCacheRequest req) {
        return withFailover(c -> c.trimCache(req));
    }

    // --- Reporting ---

    @Override
    public kv_cache_manager.proto.meta.MetaServiceOuterClass.ReportEventResponse reportEvent(
            kv_cache_manager.proto.meta.MetaServiceOuterClass.ReportEventRequest req) {
        return withFailover(c -> c.reportEvent(req));
    }

    // --- Cluster info ---

    @Override
    public kv_cache_manager.proto.meta.MetaServiceOuterClass.GetClusterInfoResponse getClusterInfo(
            kv_cache_manager.proto.meta.MetaServiceOuterClass.GetClusterInfoRequest req) {
        return withFailover(c -> c.getClusterInfo(req));
    }

    // --- Lifecycle ---

    @Override
    public void close() throws Exception {
        if (leaderDiscovery != null) {
            leaderDiscovery.stop();
        }
        grpcClient.close();
        if (httpClient != null) {
            httpClient.close();
        }
    }

    // --- Internal failover logic ---

    @FunctionalInterface
    private interface RpcCall<T> {
        T execute(MetaClient client);
    }

    /**
     * Core failover template:
     * 1. Try gRPC
     * 2. On SERVER_NOT_LEADER → discover leader → rebuild channel → retry (up to leaderRetryCount)
     * 3. On gRPC IO error → fallback to HTTP if available
     */
    private <T> T withFailover(RpcCall<T> rpc) {
        // Try gRPC with leader-not-leader retry
        int retriesLeft = config.getLeaderRetryCount();
        while (true) {
            try {
                return rpc.execute(grpcClient);
            } catch (ServerNotLeaderException e) {
                LOG.warn("Server is not leader, triggering re-discovery");
                if (leaderDiscovery != null) {
                    leaderDiscovery.triggerImmediateRefresh();
                    if (leaderDiscovery.discoverLeader()) {
                        reconnectIfNeeded();
                    }
                }
                if (retriesLeft <= 0) {
                    throw e;
                }
                retriesLeft--;
                int attempt = config.getLeaderRetryCount() - retriesLeft;
                long backoffMs = BASE_BACKOFF_MS * attempt + (long) (Math.random() * BASE_BACKOFF_MS);
                try {
                    Thread.sleep(backoffMs);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    throw new KvcmException(ErrorCode.IO_ERROR, "Interrupted during failover backoff", ie);
                }
            } catch (KvcmException e) {
                if (e.getErrorCode() == ErrorCode.IO_ERROR && httpClient != null) {
                    LOG.warn("gRPC call failed with IO error, falling back to HTTP: {}", e.getMessage());
                    return rpc.execute(httpClient);
                }
                throw e;
            }
        }
    }

    private void reconnectIfNeeded() {
        if (leaderDiscovery == null) {
            return;
        }
        String leaderHost = leaderDiscovery.getCurrentHost();
        int leaderPort = leaderDiscovery.getCurrentPort();
        synchronized (reconnectLock) {
            // Check if already connected to this address
            if (leaderHost.equals(config.getSeedAddress()) && leaderPort == config.getGrpcPort()) {
                return; // still on seed, no reconnect needed
            }
            LOG.info("Reconnecting gRPC channel to {}:{}", leaderHost, leaderPort);
            GrpcMetaClient old = grpcClient;
            grpcClient = new GrpcMetaClient(leaderHost, leaderPort, config.getCallTimeoutMs());
            try {
                old.close();
            } catch (Exception e) {
                LOG.warn("Error closing old gRPC channel during reconnect", e);
            }
        }
    }
}
