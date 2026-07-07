package com.alibaba.tair.kvcm.client;

import com.alibaba.tair.kvcm.client.exception.KvcmException;
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
/**
 * Immutable holder for leader address (host + ports).
 * Ensures atomic reads of all fields.
 */
final class LeaderAddress {
    final String host;
    final int grpcPort;
    final int httpPort;

    LeaderAddress(String host, int grpcPort) {
        this(host, grpcPort, 0);
    }

    LeaderAddress(String host, int grpcPort, int httpPort) {
        this.host = host;
        this.grpcPort = grpcPort;
        this.httpPort = httpPort;
    }

    /** For backward compatibility: port() returns grpcPort. */
    int port() { return grpcPort; }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof LeaderAddress)) return false;
        LeaderAddress that = (LeaderAddress) o;
        return grpcPort == that.grpcPort && httpPort == that.httpPort && host.equals(that.host);
    }

    @Override
    public int hashCode() {
        return 31 * (31 * host.hashCode() + grpcPort) + httpPort;
    }
}

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

    // C3 fix: Use immutable holder for atomic host+port reads
    private volatile LeaderAddress currentAddress;
    // M1 fix: Make scheduler volatile for safe cross-thread publication
    private volatile ScheduledExecutorService scheduler;
    // C2 fix: Lock for channel creation to prevent check-then-act race
    private final Object channelLock = new Object();
    private ManagedChannel discoveryChannel; // guarded by channelLock

    LeaderDiscovery(String seedAddress, int grpcPort, String instanceId, int refreshIntervalSeconds) {
        this.seedAddress = seedAddress;
        this.grpcPort = grpcPort;
        this.instanceId = instanceId;
        this.refreshIntervalSeconds = refreshIntervalSeconds;
        this.currentAddress = new LeaderAddress(seedAddress, grpcPort);
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
        ScheduledExecutorService s = scheduler;
        if (s != null) {
            // Schedule immediate refresh
            s.schedule(this::refreshLoop, 0, TimeUnit.MILLISECONDS);
        }
    }

    /**
     * Attempt to discover the leader and update current address.
     *
     * @return true if leader was discovered and address updated
     */
    boolean discoverLeader() {
        ManagedChannel channel;
        // C2 fix: Lock channel creation to prevent check-then-act race
        synchronized (channelLock) {
            if (discoveryChannel == null || discoveryChannel.isShutdown()) {
                discoveryChannel = ManagedChannelBuilder.forAddress(seedAddress, grpcPort)
                        .usePlaintext()
                        .build();
            }
            channel = discoveryChannel;
        }

        try {
            MetaServiceGrpc.MetaServiceBlockingStub stub = MetaServiceGrpc.newBlockingStub(channel);

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

            // C3 fix: Atomic update via immutable holder
            LeaderAddress newAddress = new LeaderAddress(
                    endpoint.getHost(), endpoint.getMetaRpcPort(), endpoint.getMetaHttpPort());
            LeaderAddress old = currentAddress;
            if (!newAddress.equals(old)) {
                LOG.info("Leader discovered: switching from {}:{} to {}:{}",
                        old.host, old.grpcPort, newAddress.host, newAddress.grpcPort);
                currentAddress = newAddress;
            }
            return true;
        } catch (Exception e) {
            LOG.warn("Leader discovery from {} failed: {}", seedAddress, e.getMessage());
            // Channel may be broken, force rebuild next time
            synchronized (channelLock) {
                if (discoveryChannel != null) {
                    discoveryChannel.shutdownNow();
                    discoveryChannel = null;
                }
            }
            return false;
        } finally {
            lastDiscoverTimeMs.set(System.currentTimeMillis());
        }
    }

    LeaderAddress getCurrentAddress() { return currentAddress; }
    String getCurrentHost() { return currentAddress.host; }
    int getCurrentPort() { return currentAddress.grpcPort; }

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
        synchronized (channelLock) {
            if (discoveryChannel != null) {
                discoveryChannel.shutdownNow();
                discoveryChannel = null;
            }
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
