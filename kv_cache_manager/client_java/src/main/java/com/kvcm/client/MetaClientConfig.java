package com.kvcm.client;

/**
 * Configuration for KVCM MetaClient.
 * Use {@link Builder} to construct instances.
 */
public final class MetaClientConfig {

    private final String seedAddress;
    private final int grpcPort;
    private final Integer httpPort; // null = HTTP disabled
    private final String instanceId;
    private final int callTimeoutMs;
    private final boolean autoDiscoverLeader;
    private final int leaderRetryCount;
    private final int leaderRefreshIntervalSeconds;

    private MetaClientConfig(Builder builder) {
        this.seedAddress = builder.seedAddress;
        this.grpcPort = builder.grpcPort;
        this.httpPort = builder.httpPort;
        this.instanceId = builder.instanceId;
        this.callTimeoutMs = builder.callTimeoutMs;
        this.autoDiscoverLeader = builder.autoDiscoverLeader;
        this.leaderRetryCount = builder.leaderRetryCount;
        this.leaderRefreshIntervalSeconds = builder.leaderRefreshIntervalSeconds;
    }

    public String getSeedAddress() { return seedAddress; }
    public int getGrpcPort() { return grpcPort; }
    public Integer getHttpPort() { return httpPort; }
    public boolean isHttpEnabled() { return httpPort != null; }
    public String getInstanceId() { return instanceId; }
    public int getCallTimeoutMs() { return callTimeoutMs; }
    public boolean isAutoDiscoverLeader() { return autoDiscoverLeader; }
    public int getLeaderRetryCount() { return leaderRetryCount; }
    public int getLeaderRefreshIntervalSeconds() { return leaderRefreshIntervalSeconds; }

    public static Builder builder(String seedAddress) {
        return new Builder(seedAddress);
    }

    public static final class Builder {
        private final String seedAddress;
        private int grpcPort = 6381;
        private Integer httpPort = null;
        private String instanceId = "";
        private int callTimeoutMs = 5000;
        private boolean autoDiscoverLeader = true;
        private int leaderRetryCount = 1;
        private int leaderRefreshIntervalSeconds = 30;

        private Builder(String seedAddress) {
            if (seedAddress == null || seedAddress.isEmpty()) {
                throw new IllegalArgumentException("seedAddress is required");
            }
            this.seedAddress = seedAddress;
        }

        public Builder grpcPort(int grpcPort) { this.grpcPort = grpcPort; return this; }
        public Builder httpPort(int httpPort) { this.httpPort = httpPort; return this; }
        public Builder instanceId(String instanceId) { this.instanceId = instanceId; return this; }
        public Builder callTimeoutMs(int callTimeoutMs) { this.callTimeoutMs = callTimeoutMs; return this; }
        public Builder autoDiscoverLeader(boolean autoDiscoverLeader) { this.autoDiscoverLeader = autoDiscoverLeader; return this; }
        public Builder leaderRetryCount(int leaderRetryCount) { this.leaderRetryCount = leaderRetryCount; return this; }
        public Builder leaderRefreshIntervalSeconds(int leaderRefreshIntervalSeconds) { this.leaderRefreshIntervalSeconds = leaderRefreshIntervalSeconds; return this; }

        public MetaClientConfig build() {
            // M3 fix: Validate numeric parameters
            if (grpcPort <= 0 || grpcPort > 65535) {
                throw new IllegalArgumentException("grpcPort must be between 1 and 65535, got: " + grpcPort);
            }
            if (httpPort != null && (httpPort <= 0 || httpPort > 65535)) {
                throw new IllegalArgumentException("httpPort must be between 1 and 65535, got: " + httpPort);
            }
            if (callTimeoutMs <= 0) {
                throw new IllegalArgumentException("callTimeoutMs must be > 0, got: " + callTimeoutMs);
            }
            if (leaderRetryCount < 0) {
                throw new IllegalArgumentException("leaderRetryCount must be >= 0, got: " + leaderRetryCount);
            }
            if (leaderRefreshIntervalSeconds <= 0) {
                throw new IllegalArgumentException("leaderRefreshIntervalSeconds must be > 0, got: " + leaderRefreshIntervalSeconds);
            }
            return new MetaClientConfig(this);
        }
    }
}
