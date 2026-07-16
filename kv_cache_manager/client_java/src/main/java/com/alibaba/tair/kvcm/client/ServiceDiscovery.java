package com.alibaba.tair.kvcm.client;

import java.util.List;

/**
 * Service discovery abstraction for resolving KVCM manager endpoints.
 * <p>
 * Implementations may resolve endpoints from static lists, Spectrum gateway,
 * VIPServer, or any other service registry.
 * <p>
 * Implementations must be thread-safe.
 */
public interface ServiceDiscovery extends AutoCloseable {

    /**
     * Return all available endpoints. Empty list if unavailable.
     */
    List<ServiceEndpoint> getAllEndpoints();

    /**
     * Return a single endpoint via load-balancing. {@code null} if unavailable.
     */
    ServiceEndpoint getOneEndpoint();

    /**
     * Force refresh of the endpoint list. Returns {@code true} if successful.
     */
    boolean refresh();

    /**
     * Return implementation type name (e.g. "Static", "Spectrum").
     */
    String getType();

    /**
     * Release resources. No-op by default.
     */
    @Override
    default void close() {}
}
