#ifndef KVCM_KV_META_OBJECT_NATIVE_API_H
#define KVCM_KV_META_OBJECT_NATIVE_API_H

#include <stddef.h>
#include <stdint.h>

// The consumer keeps this capsule alive for the entire call, including while
// the GIL is released. The capsule owns a reference to the existing client;
// normal explicit Close still closes that client and drains admitted calls.
#define KVCM_OBJECT_NATIVE_API_CAPSULE_V1 "kv_cache_manager.ObjectNativeApi.v1"
#define KVCM_OBJECT_NATIVE_API_VERSION_1 1U

typedef struct KvcmObjectNativeSaveRequestV1 {
    const char *key;
    size_t key_size;
    const char *trace_id;
    size_t trace_id_size;
    void *buffer;
    uint64_t buffer_size;
    uint8_t memory_type; // 0 = CPU, 1 = GPU.
} KvcmObjectNativeSaveRequestV1;

typedef struct KvcmObjectNativeResultV1 {
    int32_t code; // ClientErrorCode from the SDK
    uint8_t data_transfer_may_be_inflight;
} KvcmObjectNativeResultV1;

#ifdef __cplusplus
extern "C" {
#endif

// No callback may throw across the ABI or call Python. The buffer must remain
// alive through the call and longer when its result reports possible DMA.
typedef KvcmObjectNativeResultV1 KvcmObjectNativeSaveV1(void *context, const KvcmObjectNativeSaveRequestV1 *request);

#ifdef __cplusplus
}
#endif

typedef struct KvcmObjectNativeApiV1 {
    uint32_t version;
    uint32_t struct_size;
    void *context;
    KvcmObjectNativeSaveV1 *save;
} KvcmObjectNativeApiV1;

#endif
