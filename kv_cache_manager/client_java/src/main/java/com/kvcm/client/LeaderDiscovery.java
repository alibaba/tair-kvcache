package com.kvcm.client;

import com.kvcm.client.exception.KvcmException;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import kv_cache_manager.proto.meta.MetaServiceGrpc;
import kv_cache_manager.proto.meta.MetaServiceOuterClass.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Discovers and tracks the KVCM leader node via GetClusterInfo.
 * <p>
 * Background thread periodically refreshes the leader address.
 * Call {@link #triggerImmediateRefresh()} to force an immediate refresh
 * (e.g., on SERVER_NOT_LEADER or connection failure).
 */
class LeaderDiscovery {

    private static final Logger LOG = LoggerFactory.getLogger(LeaderDiscovery.class);
    private static final long MIN_DISCOVER_INTERVAL_MS = 1000;
    private static final int DISCOVERY_TIMEOUT_MS = 5000;

    private final String seedAddress;
    private final int grpcPort;
    private final String instanceId;
    private final int refreshIntervalSeconds;
    private final AtomicLong lastDiscoverTimeMs = new AtomicLong(0);
    private final AtomicBoolean running = new AtomicBoolean(false);
    private final AtomicBoolean immediateRefresh = new AtomicBoolean(false);

    private volatile String currentHost;
    private volatile int currentPort;

    private ScheduledExecutorService scheduler;
    // M1 fix: Reuse channel for discovery instead of creating per attempt
    private volatile ManagedChannel discoveryChannel;

    LeaderDiscovery(String seedAddress, int grpcPort, String instanceId, int refreshIntervalSeconds) {
        this.seedAddress = seedAddress;
        this.grpcPort = grpcPort;
        this.instanceId = instanceId;
        this.refreshIntervalSeconds = refreshIntervalSeconds;
        this.currentHost = seedAddress;
        this.currentPort = grpcPort;
    }

    /**
     * Start leader discovery: immediate attempt + background refresh thread.
     */
    void start() {
        if (!running.compareAndSet(false, true)) {
            return;
        }
        // Immediate discovery
        try {
            discoverLeader();
        } catch (Exception e) {
            LOG.warn("Initial leader discovery failed, keeping seed address {}: {}", seedAddress, e.getMessage());
        }

        scheduler = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "kvcm-leader-refresh");
            t.setDaemon(true);
            return t;
        });
        scheduler.scheduleWithFixedDelay(this::refreshLoop, refreshIntervalSeconds,
                refreshIntervalSeconds, TimeUnit.SECONDS);
    }

    /**
     * Request an immediate leader refresh. The background thread will pick it up
     * on its next cycle (or sooner if currently sleeping).
     */
    void triggerImmediateRefresh() {
        immediateRefresh.set(true);
        if (scheduler != null) {
            // Cancel and reschedule to wake up sooner
            // Not critical if we miss - the next cycle will pick it up
            scheduler.schedule(this::refreshLoop, 0, TimeUnit.MILLISECONDS);
        }
    }

    /**
     * Attempt to discover the leader and update current address.
     *
     * @return true if leader was discovered and address updated
     */
    boolean discoverLeader() {
        try {
            // M1 fix: Reuse channel, only rebuild if null or shutdown
            if (discoveryChannel == null || discoveryChannel.isShutdown()) {
                discoveryChannel = ManagedChannelBuilder.forAddress(seedAddress, grpcPort)
                        .usePlaintext()
                        .build();
            }
            MetaServiceGrpc.MetaServiceBlockingStub stub = MetaServiceGrpc.newBlockingStub(discoveryChannel);

            GetClusterInfoRequest request = GetClusterInfoRequest.newBuilder()
                    .setTraceId("leader_discovery_" + System.nanoTime())
                    .setInstanceId(instanceId)
                    .build();

            GetClusterInfoResponse response = stub.withDeadlineAfter(DISCOVERY_TIMEOUT_MS, TimeUnit.MILLISECONDS)
                    .getClusterInfo(request);

            if (!response.hasHeader() || response.getHeader().getStatus().getCode() != ErrorCode.OK) {
                String msg = response.hasHeader() ? response.getHeader().getStatus().getMessage() : "no header";
                LOG.warn("Leader discovery from {} returned error: {}", seedAddress, msg);
                return false;
            }

            if (!response.hasLeaderEndpoint()) {
                LOG.warn("Leader discovery from {}: leader_endpoint missing", seedAddress);
                return false;
            }

            MetaNodeEndpoint endpoint = response.getLeaderEndpoint();
            if (endpoint.getHost().isEmpty() || endpoint.getMetaRpcPort() <= 0) {
                LOG.warn("Leader discovery from {}: leader_endpoint incomplete (host={}, port={})",
                        seedAddress, endpoint.getHost(), endpoint.getMetaRpcPort());
                return false;
            }

            String newHost = endpoint.getHost();
            int newPort = endpoint.getMetaRpcPort();

            if (!newHost.equals(currentHost) || newPort != currentPort) {
                LOG.info("Leader discovered: switching from {}:{} to {}:{}",
                        currentHost, currentPort, newHost, newPort);
                currentHost = newHost;
                currentPort = newPort;
            }
            return true;
        } catch (Exception e) {
            LOG.warn("Leader discovery from {} failed: {}", seedAddress, e.getMessage());
            // Channel may be broken, force rebuild next time
            if (discoveryChannel != null) {
                discoveryChannel.shutdownNow();
                discoveryChannel = null;
            }
            return false;
        } finally {
            lastDiscoverTimeMs.set(System.currentTimeMillis());
        }
    }

    String getCurrentHost() { return currentHost; }
    int getCurrentPort() { return currentPort; }

    void stop() {
        running.set(false);
        if (scheduler != null) {
            scheduler.shutdown();
            try {
                if (!scheduler.awaitTermination(3, TimeUnit.SECONDS)) {
                    scheduler.shutdownNow();
                }
            } catch (InterruptedException e) {
                scheduler.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
        // M1 fix: Also shutdown the reused discovery channel
        if (discoveryChannel != null) {
            discoveryChannel.shutdownNow();
            discoveryChannel = null;
        }
    }

    private void refreshLoop() {
        if (!running.get()) {
            return;
        }
        // Min interval guard
        long elapsed = System.currentTimeMillis() - lastDiscoverTimeMs.get();
        if (elapsed < MIN_DISCOVER_INTERVAL_MS && !immediateRefresh.compareAndSet(true, false)) {
            return;
        }
        immediateRefresh.set(false);
        try {
            discoverLeader();
        } catch (Exception e) {
            LOG.warn("Background leader refresh failed: {}", e.getMessage());
        }
    }
}
