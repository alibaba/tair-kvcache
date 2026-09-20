{
    global:
        KVCMStagedChecksumRuntimeApiVersion;
        # Keep the public C++ SDK ABI visible while preventing statically linked
        # implementation dependencies (notably gRPC and protobuf) from
        # interposing on an embedding application's copies.
        _ZN16kv_cache_manager*;
        _ZNK16kv_cache_manager*;
        _ZTIN16kv_cache_manager*;
        _ZTSN16kv_cache_manager*;
        _ZTVN16kv_cache_manager*;
        _ZTTN16kv_cache_manager*;
    local:
        *;
};
