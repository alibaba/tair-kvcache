package com.alibaba.tair.kvcm.client;

import java.util.Objects;

/**
 * A resolved service endpoint returned by {@link ServiceDiscovery}.
 */
public final class ServiceEndpoint {

    private final String host;
    private final int port;
    private final int weight;
    private final boolean healthy;

    public ServiceEndpoint(String host, int port) {
        this(host, port, 100, true);
    }

    public ServiceEndpoint(String host, int port, int weight, boolean healthy) {
        if (host == null || host.isEmpty()) {
            throw new IllegalArgumentException("host must not be null or empty");
        }
        if (port <= 0 || port > 65535) {
            throw new IllegalArgumentException("port must be between 1 and 65535, got: " + port);
        }
        this.host = host;
        this.port = port;
        this.weight = weight;
        this.healthy = healthy;
    }

    public String getHost() { return host; }
    public int getPort() { return port; }
    public int getWeight() { return weight; }
    public boolean isHealthy() { return healthy; }

    /** Returns "host:port" format suitable for connection strings. */
    public String getAddress() { return host + ":" + port; }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof ServiceEndpoint)) return false;
        ServiceEndpoint that = (ServiceEndpoint) o;
        return port == that.port && weight == that.weight
                && healthy == that.healthy && host.equals(that.host);
    }

    @Override
    public int hashCode() {
        return Objects.hash(host, port, weight, healthy);
    }

    @Override
    public String toString() {
        return "ServiceEndpoint{host='" + host + "', port=" + port
                + ", weight=" + weight + ", healthy=" + healthy + "}";
    }
}
