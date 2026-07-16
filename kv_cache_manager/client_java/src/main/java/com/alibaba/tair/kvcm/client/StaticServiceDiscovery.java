package com.alibaba.tair.kvcm.client;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Fixed ip:port list service discovery with thread-safe round-robin selection.
 * <p>
 * Parses {@code static://ip:port[,ip:port]...} URL body.
 */
public class StaticServiceDiscovery implements ServiceDiscovery {

    private final List<ServiceEndpoint> endpoints;
    private final AtomicInteger index = new AtomicInteger(0);

    public StaticServiceDiscovery(String hostList) {
        if (hostList == null || hostList.isEmpty()) {
            throw new IllegalArgumentException("host list must not be null or empty");
        }
        this.endpoints = Collections.unmodifiableList(parseHostList(hostList));
        if (this.endpoints.isEmpty()) {
            throw new IllegalArgumentException("no valid endpoints parsed from: " + hostList);
        }
    }

    StaticServiceDiscovery(List<ServiceEndpoint> endpoints) {
        if (endpoints == null || endpoints.isEmpty()) {
            throw new IllegalArgumentException("endpoints must not be null or empty");
        }
        this.endpoints = Collections.unmodifiableList(new ArrayList<>(endpoints));
    }

    @Override
    public List<ServiceEndpoint> getAllEndpoints() {
        return new ArrayList<>(endpoints);
    }

    @Override
    public ServiceEndpoint getOneEndpoint() {
        if (endpoints.isEmpty()) {
            return null;
        }
        int i = index.getAndIncrement();
        // Handle wrap-around safely even if AtomicInteger overflows
        return endpoints.get(Math.abs(i % endpoints.size()));
    }

    @Override
    public boolean refresh() {
        return true;
    }

    @Override
    public String getType() {
        return "Static";
    }

    static List<ServiceEndpoint> parseHostList(String hostList) {
        List<ServiceEndpoint> result = new ArrayList<>();
        for (String token : hostList.split(",")) {
            token = token.trim();
            if (token.isEmpty()) {
                continue;
            }
            int colonIdx = token.lastIndexOf(':');
            if (colonIdx <= 0 || colonIdx == token.length() - 1) {
                throw new IllegalArgumentException("invalid endpoint format, expected ip:port: " + token);
            }
            String host = token.substring(0, colonIdx);
            String portStr = token.substring(colonIdx + 1);
            if (host.isEmpty()) {
                throw new IllegalArgumentException("host must not be empty: " + token);
            }
            int port;
            try {
                port = Integer.parseInt(portStr);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("port not numeric: " + token, e);
            }
            if (port <= 0 || port > 65535) {
                throw new IllegalArgumentException("port out of range: " + token);
            }
            result.add(new ServiceEndpoint(host, port));
        }
        return result;
    }
}
