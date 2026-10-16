# DebugService diagnostic API

Enable the debug service only in a test environment with
`--env kvcm.service.enable_debug_service=true`. The service is disabled by default.
Its HTTP listener uses the MetaService HTTP port plus 3000; its gRPC service
shares the main RPC listener unless a separate admin RPC listener is configured.

## GetCacheMetaDetail

```bash
curl -X POST http://localhost:9382/api/getCacheMetaDetail \
  -H "Content-Type: application/json" \
  -d '{
    "trace_id": "debug_meta_1",
    "instance_id": "test_instance",
    "block_keys": [123, 456],
    "block_mask": {"offset": 0},
    "detail_level": 1
  }'
```

The equivalent gRPC method is `kv_cache_manager.proto.debug.DebugService/GetCacheMetaDetail`.
The request and response messages are defined in `meta_service.proto`, but the RPC
is exposed only by DebugService. Ordinary MetaService clients have no route for it.

The response has one `items` entry per unmasked key, in request order. It includes
raw properties and all locations without storage selection, data-file checks, or
lazy pruning. `prev_block_key` also appears in `properties` intentionally.
Missing keys have an `OK` item status and a `CLS_NOT_FOUND` location. Read or
deserialization failures appear in the corresponding item status; successful
keys remain available. If all lookups fail, the header is non-OK and the item
errors are still returned.
