package com.alibaba.tair.kvcm.client;

import com.alibaba.tair.kvcm.client.exception.KvcmException;
import com.alibaba.tair.kvcm.client.exception.ServerNotLeaderException;
import kv_cache_manager.proto.meta.MetaServiceOuterClass.ErrorCode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Auto-failover MetaClient: gRPC primary → HTTP fallback → leader re-discovery + retry.
 * <p>
 * This is the recommended entry point for production use.
 * Create via {@link MetaClientFactory#create(MetaClientConfig)}.
 */
public class AutoFailoverClient implements MetaClient {

    private static final Logger LOG = LoggerFactory.getLogger(AutoFailoverClient.class);
    private static final long BASE_BACKOFF_MS = 100;

    // Shared scheduler for delayed channel close
    private static final ScheduledExecutorService CLOSE_SCHEDULER =
            Executors.newSingleThreadScheduledExecutor(r -> {
                Thread t = new Thread(r, "kvcm-channel-close");
                t.setDaemon(true);
                return t;
            });

    private final MetaClientConfig config;
    private volatile GrpcMetaClient grpcClient;
    private volatile LeaderAddress currentAddress;
    private volatile HttpMetaClient httpClient; // volatile: recreated on leader change
    private final LeaderDiscovery leaderDiscovery; // null if auto-discover disabled
    private final Object reconnectLock = new Object();
    private final AtomicBoolean closed = new AtomicBoolean(false);

    AutoFailoverClient(MetaClientConfig config) {
        this.config = config;
        this.currentAddress = new LeaderAddress(config.getSeedAddress(), config.getGrpcPort());
        this.grpcClient = new GrpcMetaClient(currentAddress.host, currentAddress.port,
                config.getCallTimeoutMs());

        if (config.isHttpEnabled()) {
            this.httpClient = new HttpMetaClient(currentAddress.host, config.getHttpPort(),
                    config.getCallTimeoutMs());
        }

        if (config.isAutoDiscoverLeader()) {
            this.leaderDiscovery = new LeaderDiscovery(
                    config.getSeedAddress(), config.getGrpcPort(),
                    config.getInstanceId(), config.getLeaderRefreshIntervalSeconds());
            leaderDiscovery.start();
            reconnectIfNeeded();
        } else {
            this.leaderDiscovery = null;
        }
    }

    /** Package-private: for testing with pre-built clients (e.g., InProcess gRPC). */
    AutoFailoverClient(MetaClientConfig config, GrpcMetaClient grpcClient, HttpMetaClient httpClient) {
        this.config = config;
        this.currentAddress = new LeaderAddress(config.getSeedAddress(), config.getGrpcPort());
        this.grpcClient = grpcClient;
        this.httpClient = httpClient;
        this.leaderDiscovery = null; // no leader discovery in test mode
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
        if (!closed.compareAndSet(false, true)) {
            return;
        }
        if (leaderDiscovery != null) {
            leaderDiscovery.stop();
        }
        // Acquire lock to prevent race with reconnectIfNeeded
        synchronized (reconnectLock) {
            grpcClient.close();
            HttpMetaClient http = httpClient;
            if (http != null) {
                http.close();
            }
        }
    }

    // --- Internal failover logic ---

    @FunctionalInterface
    private interface RpcCall<T> {
        T execute(MetaClient client);
    }

    private void ensureOpen() {
        if (closed.get()) {
            throw new IllegalStateException("Client is closed");
        }
    }

    /**
     * Core failover template:
     * 1. Try gRPC
     * 2. On SERVER_NOT_LEADER → discover leader → rebuild channel → retry (up to leaderRetryCount)
     * 3. On gRPC IO error → fallback to HTTP if available (propagate SERVER_NOT_LEADER for re-discovery)
     */
    private <T> T withFailover(RpcCall<T> rpc) {
        ensureOpen();
        int retriesLeft = config.getLeaderRetryCount();
        while (true) {
            GrpcMetaClient client;
            synchronized (reconnectLock) {
                client = grpcClient;
            }
            try {
                return rpc.execute(client);
            } catch (ServerNotLeaderException e) {
                LOG.warn("Server is not leader, triggering re-discovery");
                if (leaderDiscovery != null) {
                    // Synchronous discovery only — no redundant triggerImmediateRefresh
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
                HttpMetaClient http = httpClient;
                if (e.getErrorCode() == ErrorCode.IO_ERROR && http != null) {
                    LOG.warn("gRPC call failed with IO error, falling back to HTTP: {}", e.getMessage());
                    try {
                        return rpc.execute(http);
                    } catch (ServerNotLeaderException httpNotLeader) {
                        // Propagate SERVER_NOT_LEADER from HTTP path for re-discovery
                        LOG.warn("HTTP fallback also returned SERVER_NOT_LEADER, triggering re-discovery");
                        if (leaderDiscovery != null) {
                            leaderDiscovery.triggerImmediateRefresh();
                        }
                        throw httpNotLeader;
                    } catch (Exception httpEx) {
                        LOG.error("Both gRPC and HTTP transports failed", httpEx);
                        throw new KvcmException(ErrorCode.IO_ERROR,
                                "All transports failed. gRPC: " + e.getMessage()
                                        + ", HTTP: " + httpEx.getMessage(),
                                httpEx);
                    }
                }
                throw e;
            }
        }
    }

    private void reconnectIfNeeded() {
        if (leaderDiscovery == null) {
            return;
        }
        LeaderAddress leaderAddr = leaderDiscovery.getCurrentAddress();
        synchronized (reconnectLock) {
            if (leaderAddr.equals(currentAddress)) {
                return;
            }
            LOG.info("Reconnecting from {}:{} to {}:{}",
                    currentAddress.host, currentAddress.port, leaderAddr.host, leaderAddr.port);

            // Replace gRPC client
            GrpcMetaClient oldGrpc = grpcClient;
            grpcClient = new GrpcMetaClient(leaderAddr.host, leaderAddr.port, config.getCallTimeoutMs());

            // Recreate HTTP client to follow leader (if HTTP enabled)
            HttpMetaClient oldHttp = httpClient;
            if (config.isHttpEnabled()) {
                httpClient = new HttpMetaClient(leaderAddr.host, config.getHttpPort(),
                        config.getCallTimeoutMs());
            }

            currentAddress = leaderAddr;

            // Delayed close of old clients via shared scheduler
            scheduleClose(oldGrpc);
            if (oldHttp != null) {
                scheduleClose(oldHttp);
            }
        }
    }

    private static void scheduleClose(GrpcMetaClient client) {
        CLOSE_SCHEDULER.schedule(() -> {
            try {
                client.close();
            } catch (Exception e) {
                LOG.warn("Error in delayed close of gRPC channel", e);
            }
        }, 100, TimeUnit.MILLISECONDS);
    }

    private static void scheduleClose(HttpMetaClient client) {
        CLOSE_SCHEDULER.schedule(() -> {
            try {
                client.close();
            } catch (Exception e) {
                LOG.warn("Error in delayed close of HTTP client", e);
            }
        }, 100, TimeUnit.MILLISECONDS);
    }
}
