package com.kvcm.client.exception;

import kv_cache_manager.proto.meta.MetaServiceOuterClass.ErrorCode;

/**
 * Thrown when the server responds with SERVER_NOT_LEADER.
 * The failover client catches this to trigger leader re-discovery and retry.
 */
public class ServerNotLeaderException extends KvcmException {

    public ServerNotLeaderException(String message) {
        super(ErrorCode.SERVER_NOT_LEADER, message);
    }
}
