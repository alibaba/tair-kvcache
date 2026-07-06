package com.kvcm.client;

import com.google.protobuf.InvalidProtocolBufferException;
import com.google.protobuf.Message;
import com.google.protobuf.util.JsonFormat;
import com.kvcm.client.exception.KvcmException;
import kv_cache_manager.proto.meta.MetaServiceOuterClass.*;
import okhttp3.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

/**
 * HTTP fallback MetaClient implementation.
 * Uses OkHttp for transport and protobuf-java-util JsonFormat for serialization.
 * This ensures JSON output is byte-compatible with the server's own protobuf JSON format.
 */
public class HttpMetaClient implements MetaClient {

    private static final Logger LOG = LoggerFactory.getLogger(HttpMetaClient.class);
    private static final MediaType JSON = MediaType.get("application/json; charset=utf-8");

    private final OkHttpClient httpClient;
    private final String baseUrl;
    private final JsonFormat.Printer printer;
    private final JsonFormat.Parser parser;

    public HttpMetaClient(String host, int port, int callTimeoutMs) {
        this.baseUrl = "http://" + host + ":" + port;
        this.httpClient = new OkHttpClient.Builder()
                .connectTimeout(callTimeoutMs, TimeUnit.MILLISECONDS)
                .readTimeout(callTimeoutMs, TimeUnit.MILLISECONDS)
                .writeTimeout(callTimeoutMs, TimeUnit.MILLISECONDS)
                .callTimeout(callTimeoutMs, TimeUnit.MILLISECONDS)
                .build();
        this.printer = JsonFormat.printer()
                .omittingInsignificantWhitespace()
                .includingDefaultValueFields()
                .preservingProtoFieldNames();
        this.parser = JsonFormat.parser().ignoringUnknownFields();
    }

    HttpMetaClient(MetaClientConfig config) {
        this(config.getSeedAddress(), config.getHttpPort(), config.getCallTimeoutMs());
    }

    // --- Instance management ---

    @Override
    public RegisterInstanceResponse registerInstance(RegisterInstanceRequest request) {
        return call("/api/registerInstance", request, RegisterInstanceResponse.newBuilder());
    }

    @Override
    public GetInstanceInfoResponse getInstanceInfo(GetInstanceInfoRequest request) {
        return call("/api/getInstanceInfo", request, GetInstanceInfoResponse.newBuilder());
    }

    // --- CacheAware queries ---

    @Override
    public GetCacheLocationResponse getCacheLocation(GetCacheLocationRequest request) {
        return call("/api/getCacheLocation", request, GetCacheLocationResponse.newBuilder());
    }

    @Override
    public GetCacheLocationsByBackendResponse getCacheLocationsByBackend(GetCacheLocationsByBackendRequest request) {
        return call("/api/getCacheLocationsByBackend", request, GetCacheLocationsByBackendResponse.newBuilder());
    }

    @Override
    public GetCacheLocationLenResponse getCacheLocationLen(GetCacheLocationLenRequest request) {
        return call("/api/getCacheLocationLen", request, GetCacheLocationLenResponse.newBuilder());
    }

    @Override
    public GetCacheMetaResponse getCacheMeta(GetCacheMetaRequest request) {
        return call("/api/getCacheMeta", request, GetCacheMetaResponse.newBuilder());
    }

    // --- Write flow ---

    @Override
    public StartWriteCacheResponse startWriteCache(StartWriteCacheRequest request) {
        return call("/api/startWriteCache", request, StartWriteCacheResponse.newBuilder());
    }

    @Override
    public CommonResponse finishWriteCache(FinishWriteCacheRequest request) {
        return call("/api/finishWriteCache", request, CommonResponse.newBuilder());
    }

    // --- Delete / trim ---

    @Override
    public CommonResponse removeCache(RemoveCacheRequest request) {
        return call("/api/removeCache", request, CommonResponse.newBuilder());
    }

    @Override
    public CommonResponse trimCache(TrimCacheRequest request) {
        return call("/api/trimCache", request, CommonResponse.newBuilder());
    }

    // --- Reporting ---

    @Override
    public ReportEventResponse reportEvent(ReportEventRequest request) {
        return call("/api/reportEvent", request, ReportEventResponse.newBuilder());
    }

    // --- Cluster info ---

    @Override
    public GetClusterInfoResponse getClusterInfo(GetClusterInfoRequest request) {
        return call("/api/getClusterInfo", request, GetClusterInfoResponse.newBuilder());
    }

    // --- Lifecycle ---

    @Override
    public void close() {
        httpClient.dispatcher().executorService().shutdown();
        httpClient.connectionPool().evictAll();
    }

    // --- Internal ---

    private <T extends Message> T call(String endpoint, Message request, Message.Builder responseBuilder) {
        String requestJson;
        try {
            requestJson = printer.print(request);
        } catch (InvalidProtocolBufferException e) {
            throw new KvcmException(ErrorCode.INVALID_ARGUMENT,
                    "Failed to serialize request to JSON: " + e.getMessage(), e);
        }

        RequestBody body = RequestBody.create(requestJson, JSON);
        Request httpRequest = new Request.Builder()
                .url(baseUrl + endpoint)
                .post(body)
                .build();

        try (Response httpResponse = httpClient.newCall(httpRequest).execute()) {
            if (!httpResponse.isSuccessful()) {
                throw new KvcmException(ErrorCode.IO_ERROR,
                        "HTTP " + httpResponse.code() + " from " + endpoint);
            }
            String responseJson = httpResponse.body() != null ? httpResponse.body().string() : "{}";
            parser.merge(responseJson, responseBuilder);
        } catch (InvalidProtocolBufferException e) {
            throw new KvcmException(ErrorCode.IO_ERROR,
                    "Failed to parse response JSON from " + endpoint + ": " + e.getMessage(), e);
        } catch (IOException e) {
            throw new KvcmException(ErrorCode.IO_ERROR,
                    "HTTP request to " + endpoint + " failed: " + e.getMessage(), e);
        }

        @SuppressWarnings("unchecked")
        T response = (T) responseBuilder.build();
        checkHeaderViaDescriptor(response);
        return response;
    }

    /**
     * Extract and check CommonResponseHeader from any response message via proto descriptors.
     * All MetaService responses have `header` at field number 1.
     */
    private void checkHeaderViaDescriptor(Message response) {
        com.google.protobuf.Descriptors.FieldDescriptor headerField =
                response.getDescriptorForType().findFieldByName("header");
        if (headerField == null) {
            return;
        }
        CommonResponseHeader header = (CommonResponseHeader) response.getField(headerField);
        ResponseChecker.checkStatus(header);
    }
}
