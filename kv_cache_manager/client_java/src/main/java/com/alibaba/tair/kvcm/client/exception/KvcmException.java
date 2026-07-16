package com.alibaba.tair.kvcm.client.exception;

import kv_cache_manager.proto.meta.MetaServiceOuterClass.ErrorCode;

/**
 * Exception wrapping a KVCM service error.
 */
public class KvcmException extends RuntimeException {

    private final ErrorCode errorCode;

    public KvcmException(ErrorCode errorCode, String message) {
        super(message);
        this.errorCode = errorCode;
    }

    public KvcmException(ErrorCode errorCode, String message, Throwable cause) {
        super(message, cause);
        this.errorCode = errorCode;
    }

    public ErrorCode getErrorCode() {
        return errorCode;
    }
}
