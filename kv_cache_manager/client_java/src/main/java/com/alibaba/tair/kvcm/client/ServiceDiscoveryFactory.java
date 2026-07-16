package com.alibaba.tair.kvcm.client;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.ServiceLoader;

/**
 * Factory for creating {@link ServiceDiscovery} instances from URL strings.
 * <p>
 * Supported URL formats:
 * <ul>
 *   <li>{@code static://ip:port[,ip:port]...} — built-in static list</li>
 *   <li>{@code spectrum://vsid[?params]} — loaded via SPI (third-party provider)</li>
 * </ul>
 * <p>
 * Plain {@code host:port} (without scheme) is treated as a bare host and
 * must be wrapped with a port before reaching this factory.
 */
public final class ServiceDiscoveryFactory {

    private static final Logger LOG = LoggerFactory.getLogger(ServiceDiscoveryFactory.class);
    private static final String SCHEME_STATIC = "static";

    private ServiceDiscoveryFactory() {}

    /**
     * Create a {@link ServiceDiscovery} from a URL string.
     *
     * @param url URL in {@code scheme://body[?params]} format
     * @return a configured ServiceDiscovery instance
     * @throws IllegalArgumentException if the URL is null, empty, or has an unsupported scheme
     */
    public static ServiceDiscovery create(String url) {
        if (url == null || url.isEmpty()) {
            throw new IllegalArgumentException("service discovery URL must not be null or empty");
        }

        int sepIdx = url.indexOf("://");
        if (sepIdx <= 0) {
            throw new IllegalArgumentException("invalid service discovery URL, missing scheme: " + url);
        }

        String scheme = url.substring(0, sepIdx);
        String rest = url.substring(sepIdx + 3);
        if (rest.isEmpty()) {
            throw new IllegalArgumentException("invalid service discovery URL, empty body: " + url);
        }

        // Split body and query params
        String body;
        Map<String, String> params;
        int qIdx = rest.indexOf('?');
        if (qIdx >= 0) {
            body = rest.substring(0, qIdx);
            params = parseQueryString(rest.substring(qIdx + 1));
        } else {
            body = rest;
            params = Collections.emptyMap();
        }

        if (body.isEmpty()) {
            throw new IllegalArgumentException("invalid service discovery URL, empty body: " + url);
        }

        // Built-in: static scheme
        if (SCHEME_STATIC.equalsIgnoreCase(scheme)) {
            return new StaticServiceDiscovery(body);
        }

        // SPI-loaded: look up providers
        for (ServiceDiscoveryProvider provider : ServiceLoader.load(ServiceDiscoveryProvider.class)) {
            if (scheme.equalsIgnoreCase(provider.getScheme())) {
                try {
                    return provider.create(body, params);
                } catch (Exception e) {
                    throw new IllegalArgumentException(
                            "failed to create " + scheme + " service discovery: " + e.getMessage(), e);
                }
            }
        }

        throw new IllegalArgumentException("unsupported service discovery scheme: " + scheme
                + " (available: static). Register a ServiceDiscoveryProvider via META-INF/services for custom schemes.");
    }

    /**
     * Attempt to create a {@link ServiceDiscovery}. Returns {@code null} on failure
     * instead of throwing. Useful for optional discovery.
     */
    public static ServiceDiscovery tryCreate(String url) {
        try {
            return create(url);
        } catch (IllegalArgumentException e) {
            LOG.warn("Failed to create service discovery for URL '{}': {}", url, e.getMessage());
            return null;
        }
    }

    static Map<String, String> parseQueryString(String query) {
        if (query == null || query.isEmpty()) {
            return Collections.emptyMap();
        }
        Map<String, String> params = new HashMap<>();
        for (String kv : query.split("&")) {
            if (kv.isEmpty()) {
                continue;
            }
            int eqIdx = kv.indexOf('=');
            if (eqIdx > 0) {
                params.put(kv.substring(0, eqIdx), kv.substring(eqIdx + 1));
            } else {
                params.put(kv, "");
            }
        }
        return params;
    }
}
