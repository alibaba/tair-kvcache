package com.kvcm.client;

import kv_cache_manager.proto.meta.MetaServiceOuterClass.*;

/**
 * KVCM MetaService client interface.
 * Covers all 12 MetaService RPCs (control plane only, no data plane).
 */
public interface MetaClient extends AutoCloseable {

    // --- Instance management ---

    RegisterInstanceResponse registerInstance(RegisterInstanceRequest request);

    GetInstanceInfoResponse getInstanceInfo(GetInstanceInfoRequest request);

    // --- CacheAware queries ---

    GetCacheLocationResponse getCacheLocation(GetCacheLocationRequest request);

    GetCacheLocationsByBackendResponse getCacheLocationsByBackend(GetCacheLocationsByBackendRequest request);

    GetCacheLocationLenResponse getCacheLocationLen(GetCacheLocationLenRequest request);

    GetCacheMetaResponse getCacheMeta(GetCacheMetaRequest request);

    // --- Write flow ---

    StartWriteCacheResponse startWriteCache(StartWriteCacheRequest request);

    CommonResponse finishWriteCache(FinishWriteCacheRequest request);

    // --- Delete / trim ---

    CommonResponse removeCache(RemoveCacheRequest request);

    CommonResponse trimCache(TrimCacheRequest request);

    // --- Reporting ---

    ReportEventResponse reportEvent(ReportEventRequest request);

    // --- Cluster info ---

    GetClusterInfoResponse getClusterInfo(GetClusterInfoRequest request);
}
