package com.kvcm.client;

import com.kvcm.client.exception.KvcmException;
import com.kvcm.client.exception.ServerNotLeaderException;
import kv_cache_manager.proto.meta.MetaServiceOuterClass.*;

/**
 * Utility to check response status from MetaService RPCs.
 * Throws {@link KvcmException} on non-OK responses.
 */
final class ResponseChecker {

    private ResponseChecker() {}

    static void check(CommonResponse response) {
        checkStatus(response.getHeader());
    }

    static void check(RegisterInstanceResponse response) {
        checkStatus(response.getHeader());
    }

    static void check(GetInstanceInfoResponse response) {
        checkStatus(response.getHeader());
    }

    static void check(GetCacheLocationResponse response) {
        checkStatus(response.getHeader());
    }

    static void check(GetCacheLocationsByBackendResponse response) {
        checkStatus(response.getHeader());
    }

    static void check(GetCacheLocationLenResponse response) {
        checkStatus(response.getHeader());
    }

    static void check(GetCacheMetaResponse response) {
        checkStatus(response.getHeader());
    }

    static void check(StartWriteCacheResponse response) {
        checkStatus(response.getHeader());
    }

    static void check(ReportEventResponse response) {
        checkStatus(response.getHeader());
    }

    static void check(GetClusterInfoResponse response) {
        checkStatus(response.getHeader());
    }

    static void checkStatus(CommonResponseHeader header) {
        if (header == null || !header.hasStatus()) {
            throw new KvcmException(ErrorCode.INTERNAL_ERROR, "Response missing header or status");
        }
        Status status = header.getStatus();
        if (status.getCode() == ErrorCode.OK) {
            return;
        }
        if (status.getCode() == ErrorCode.SERVER_NOT_LEADER) {
            throw new ServerNotLeaderException(status.getMessage());
        }
        throw new KvcmException(status.getCode(), status.getMessage());
    }
}
