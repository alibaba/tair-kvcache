package com.alibaba.tair.kvcm.client;

/**
 * Factory for creating {@link MetaClient} instances.
 * <p>
 * Recommended usage:
 * <pre>{@code
 * MetaClientConfig config = MetaClientConfig.builder("10.0.0.1")
 *     .grpcPort(6381)
 *     .instanceId("my-instance")
 *     .build();
 *
 * try (MetaClient client = MetaClientFactory.create(config)) {
 *     GetCacheLocationResponse resp = client.getCacheLocation(
 *         GetCacheLocationRequest.newBuilder()
 *             .setInstanceId("my-instance")
 *             .addBlockKeys(12345L)
 *             .build());
 *     // process response...
 * }
 * }</pre>
 */
public final class MetaClientFactory {

    private MetaClientFactory() {}

    /**
     * Create a MetaClient with auto-failover (gRPC primary, HTTP fallback, leader discovery).
     * This is the recommended entry point for production use.
     */
    public static MetaClient create(MetaClientConfig config) {
        return new AutoFailoverClient(config);
    }

    /**
     * Create a plain gRPC-only client (no failover, no leader discovery).
     * Useful for testing or simple single-node deployments.
     */
    public static MetaClient createGrpc(MetaClientConfig config) {
        return new GrpcMetaClient(config);
    }

    /**
     * Create a plain HTTP-only client (no failover, no leader discovery).
     * Requires httpPort to be set in config.
     */
    public static MetaClient createHttp(MetaClientConfig config) {
        if (!config.isHttpEnabled()) {
            throw new IllegalArgumentException("httpPort must be set for HTTP client");
        }
        return new HttpMetaClient(config);
    }
}
