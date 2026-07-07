package com.alibaba.tair.kvcm.client;

import com.alibaba.tair.kvcm.client.exception.KvcmException;
import com.google.protobuf.util.JsonFormat;
import kv_cache_manager.proto.meta.MetaServiceOuterClass.*;
import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import okhttp3.mockwebserver.RecordedRequest;
import org.junit.jupiter.api.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for HttpMetaClient using OkHttp MockWebServer.
 */
class HttpMetaClientTest {

    private MockWebServer mockServer;
    private HttpMetaClient client;
    private JsonFormat.Printer printer;

    @BeforeEach
    void setUp() throws Exception {
        mockServer = new MockWebServer();
        mockServer.start();
        String host = mockServer.getHostName();
        int port = mockServer.getPort();
        client = new HttpMetaClient(host, port, 5000);
        printer = JsonFormat.printer().omittingInsignificantWhitespace()
                .includingDefaultValueFields().preservingProtoFieldNames();
    }

    @AfterEach
    void tearDown() throws Exception {
        client.close();
        mockServer.shutdown();
    }

    @Test
    void testGetCacheLocation_correctEndpointAndBody() throws Exception {
        // Enqueue OK response
        GetCacheLocationResponse response = GetCacheLocationResponse.newBuilder()
                .setHeader(CommonResponseHeader.newBuilder()
                        .setStatus(Status.newBuilder().setCode(ErrorCode.OK).build())
                        .build())
                .addLocations(CacheLocation.newBuilder()
                        .setType(StorageType.ST_3FS)
                        .addLocationSpecs(LocationSpec.newBuilder()
                                .setName("tp0")
                                .setUri("3fs://cluster/path?offset=0&size=1024")
                                .build())
                        .build())
                .build();
        mockServer.enqueue(new MockResponse()
                .setBody(printer.print(response))
                .setHeader("Content-Type", "application/json"));

        // Make request
        GetCacheLocationRequest request = GetCacheLocationRequest.newBuilder()
                .setTraceId("http-test")
                .setInstanceId("inst1")
                .addBlockKeys(12345L)
                .addTokenIds(1L)
                .build();
        GetCacheLocationResponse result = client.getCacheLocation(request);

        // Verify request
        RecordedRequest recorded = mockServer.takeRequest();
        assertEquals("POST", recorded.getMethod());
        assertEquals("/api/getCacheLocation", recorded.getPath());
        String body = recorded.getBody().readUtf8();
        assertTrue(body.contains("\"trace_id\":\"http-test\""), "snake_case trace_id");
        assertTrue(body.contains("\"instance_id\":\"inst1\""), "snake_case instance_id");
        assertTrue(body.contains("\"block_keys\":[\"12345\"]"), "int64 as string");

        // Verify response
        assertNotNull(result);
        assertEquals(1, result.getLocationsCount());
        assertEquals(StorageType.ST_3FS, result.getLocations(0).getType());
    }

    @Test
    void testServerError_throwsKvcmException() {
        CommonResponse errorResp = CommonResponse.newBuilder()
                .setHeader(CommonResponseHeader.newBuilder()
                        .setStatus(Status.newBuilder()
                                .setCode(ErrorCode.INSTANCE_NOT_EXIST)
                                .setMessage("instance not found")
                                .build())
                        .build())
                .build();
        try {
            mockServer.enqueue(new MockResponse()
                    .setBody("{\"header\":{\"status\":{\"code\":\"INSTANCE_NOT_EXIST\",\"message\":\"instance not found\"}}}"));
            client.getInstanceInfo(GetInstanceInfoRequest.newBuilder()
                    .setTraceId("err-test")
                    .setInstanceId("nonexistent")
                    .build());
            fail("Should have thrown");
        } catch (KvcmException e) {
            assertEquals(ErrorCode.INSTANCE_NOT_EXIST, e.getErrorCode());
        }
    }

    @Test
    void testHttp500_throwsKvcmException() {
        mockServer.enqueue(new MockResponse().setResponseCode(500).setBody("Internal Server Error"));
        assertThrows(KvcmException.class, () ->
                client.getClusterInfo(GetClusterInfoRequest.newBuilder()
                        .setTraceId("500-test")
                        .setInstanceId("inst1")
                        .build()));
    }

    @Test
    void testInt64Deserialization() throws Exception {
        // Server returns int64 as JSON strings (protobuf spec)
        GetCacheLocationLenResponse resp = GetCacheLocationLenResponse.newBuilder()
                .setHeader(CommonResponseHeader.newBuilder()
                        .setStatus(Status.newBuilder().setCode(ErrorCode.OK).build())
                        .build())
                .setCacheLocationLen(9876543210L)
                .build();
        mockServer.enqueue(new MockResponse()
                .setBody(printer.print(resp))
                .setHeader("Content-Type", "application/json"));

        GetCacheLocationLenResponse result = client.getCacheLocationLen(
                GetCacheLocationLenRequest.newBuilder()
                        .setTraceId("int64-test")
                        .setInstanceId("inst1")
                        .build());
        assertEquals(9876543210L, result.getCacheLocationLen());
    }

    @Test
    void testAllEndpoints_correctPaths() throws Exception {
        String okJson = "{\"header\":{\"status\":{\"code\":\"OK\"},\"request_id\":\"\",\"tracer_result\":\"\"}}";

        String[] endpoints = {
                "/api/registerInstance", "/api/getInstanceInfo",
                "/api/getCacheLocation", "/api/getCacheLocationsByBackend",
                "/api/getCacheLocationLen", "/api/getCacheMeta",
                "/api/startWriteCache", "/api/finishWriteCache",
                "/api/removeCache", "/api/trimCache",
                "/api/reportEvent", "/api/getClusterInfo"
        };

        for (String endpoint : endpoints) {
            mockServer.enqueue(new MockResponse().setBody(okJson).setHeader("Content-Type", "application/json"));
        }

        // Call all 12 endpoints
        client.registerInstance(RegisterInstanceRequest.newBuilder().setTraceId("t").build());
        client.getInstanceInfo(GetInstanceInfoRequest.newBuilder().setTraceId("t").build());
        client.getCacheLocation(GetCacheLocationRequest.newBuilder().setTraceId("t").build());
        client.getCacheLocationsByBackend(GetCacheLocationsByBackendRequest.newBuilder().setTraceId("t").build());
        client.getCacheLocationLen(GetCacheLocationLenRequest.newBuilder().setTraceId("t").build());
        client.getCacheMeta(GetCacheMetaRequest.newBuilder().setTraceId("t").build());
        client.startWriteCache(StartWriteCacheRequest.newBuilder().setTraceId("t").build());
        client.finishWriteCache(FinishWriteCacheRequest.newBuilder().setTraceId("t").build());
        client.removeCache(RemoveCacheRequest.newBuilder().setTraceId("t").build());
        client.trimCache(TrimCacheRequest.newBuilder().setTraceId("t").build());
        client.reportEvent(ReportEventRequest.newBuilder().setTraceId("t").build());
        client.getClusterInfo(GetClusterInfoRequest.newBuilder().setTraceId("t").build());

        // Verify all paths were hit
        for (String endpoint : endpoints) {
            RecordedRequest req = mockServer.takeRequest();
            assertEquals(endpoint, req.getPath(), "Expected path " + endpoint);
        }
    }
}
