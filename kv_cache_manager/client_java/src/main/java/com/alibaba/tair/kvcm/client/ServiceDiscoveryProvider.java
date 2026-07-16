package com.alibaba.tair.kvcm.client;

import java.util.Map;

/**
 * SPI interface for {@link ServiceDiscovery} implementations.
 * <p>
 * External implementations (e.g. custom schemes) register via
 * {@code META-INF/services/com.alibaba.tair.kvcm.client.ServiceDiscoveryProvider}.
 */
public interface ServiceDiscoveryProvider {

    /**
     * Return the URL scheme this provider handles (e.g. "spectrum").
     */
    String getScheme();

    /**
     * Create a {@link ServiceDiscovery} instance from the parsed URL components.
     *
     * @param body   the URL body (e.g. virtual service ID)
     * @param params query parameters (e.g. cache_time, timeout)
     * @return a configured ServiceDiscovery instance
     */
    ServiceDiscovery create(String body, Map<String, String> params);
}
