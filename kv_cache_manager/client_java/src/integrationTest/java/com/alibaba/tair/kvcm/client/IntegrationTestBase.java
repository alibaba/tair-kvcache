package com.alibaba.tair.kvcm.client;

import kv_cache_manager.proto.meta.MetaServiceOuterClass.*;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.TestInfo;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Base class for integration tests that require a running KVCM server.
 * <p>
 * Provides:
 * - Server lifecycle management (start/stop)
 * - Client factory methods for gRPC and HTTP
 * - Test isolation with unique instance IDs
 * - Helper methods for common operations (register, write, finish)
 */
public abstract class IntegrationTestBase {

    private static final Logger LOG = LoggerFactory.getLogger(IntegrationTestBase.class);

    protected static KvcmServerManager server;
    protected static MetaClient grpcClient;
    protected static MetaClient httpClient;

    protected String instanceId;

    @BeforeAll
    static void startServer() throws Exception {
        server = new KvcmServerManager();
        server.start();

        // Create gRPC client
        grpcClient = new GrpcMetaClient("localhost", server.getRpcPort(), 5000);

        // Create HTTP client
        httpClient = new HttpMetaClient("localhost", server.getHttpPort(), 5000);

        LOG.info("Integration test server started on ports: gRPC={}, HTTP={}",
                server.getRpcPort(), server.getHttpPort());
    }

    @AfterAll
    static void stopServer() throws Exception {
        if (grpcClient != null) {
            grpcClient.close();
        }
        if (server != null) {
            server.stop();
        }
        LOG.info("Integration test server stopped");
    }

    @BeforeEach
    void setUpTestIsolation(TestInfo testInfo) {
        // Generate unique instance ID for each test
        String methodName = testInfo.getTestMethod()
                .map(m -> m.getName())
                .orElse("unknown");
        instanceId = "test_" + methodName + "_" + System.currentTimeMillis();
    }

    /**
     * Gets the gRPC client for testing.
     */
    protected MetaClient getGrpcClient() {
        return grpcClient;
    }

    /**
     * Gets the HTTP client for testing.
     * Note: Not all tests use HTTP, so this may be null in some cases.
     */
    protected MetaClient getHttpClient() {
        return httpClient;
    }

    /**
     * Helper to register an instance with default configuration.
     *
     * @param instanceId the instance ID to register
     * @return the registration response
     */
    protected RegisterInstanceResponse registerInstance(String instanceId) {
        return registerInstance(instanceId, 128);
    }

    /**
     * Helper to register an instance with custom block size.
     *
     * @param instanceId the instance ID to register
     * @param blockSize the block size
     * @return the registration response
     */
    protected RegisterInstanceResponse registerInstance(String instanceId, int blockSize) {
        ModelDeployment deployment = ModelDeployment.newBuilder()
                .setModelName("test_model")
                .setDtype("FP8")
                .setUseMla(false)
                .setTpSize(1)
                .setDpSize(1)
                .setPpSize(1)
                .build();

        RegisterInstanceRequest request = RegisterInstanceRequest.newBuilder()
                .setTraceId("test-trace")
                .setInstanceGroup("default")
                .setInstanceId(instanceId)
                .setBlockSize(blockSize)
                .setModelDeployment(deployment)
                .addLocationSpecInfos(LocationSpecInfo.newBuilder()
                        .setName("tp0")
                        .setSize(1024)
                        .build())
                .build();

        return grpcClient.registerInstance(request);
    }

    /**
     * Helper to start writing cache entries.
     *
     * @param instanceId the instance ID
     * @param blockKeys the block keys to write
     * @return the write session ID
     */
    protected String startWriteCache(String instanceId, long... blockKeys) {
        StartWriteCacheRequest.Builder builder = StartWriteCacheRequest.newBuilder()
                .setTraceId("test-trace")
                .setInstanceId(instanceId)
                .setWriteTimeoutSeconds(30);

        for (long key : blockKeys) {
            builder.addBlockKeys(key);
        }

        StartWriteCacheResponse response = grpcClient.startWriteCache(builder.build());
        return response.getWriteSessionId();
    }

    /**
     * Helper to finish writing cache entries (all blocks successful).
     *
     * @param instanceId the instance ID
     * @param writeSessionId the write session ID
     * @param blockCount the number of blocks written
     */
    protected void finishWriteCache(String instanceId, String writeSessionId, int blockCount) {
        BoolMasksType.Builder boolMasks = BoolMasksType.newBuilder();
        for (int i = 0; i < blockCount; i++) {
            boolMasks.addValues(true);
        }

        BlockMask blockMask = BlockMask.newBuilder()
                .setBoolMasks(boolMasks)
                .build();

        FinishWriteCacheRequest request = FinishWriteCacheRequest.newBuilder()
                .setTraceId("test-trace")
                .setInstanceId(instanceId)
                .setWriteSessionId(writeSessionId)
                .setSuccessBlocks(blockMask)
                .build();

        grpcClient.finishWriteCache(request);
    }

    /**
     * Helper to finish writing cache entries with custom block mask offset.
     *
     * @param instanceId the instance ID
     * @param writeSessionId the write session ID
     * @param offset the block mask offset
     */
    protected void finishWriteCacheWithOffset(String instanceId, String writeSessionId, int offset) {
        BlockMask blockMask = BlockMask.newBuilder()
                .setOffset(offset)
                .build();

        FinishWriteCacheRequest request = FinishWriteCacheRequest.newBuilder()
                .setTraceId("test-trace")
                .setInstanceId(instanceId)
                .setWriteSessionId(writeSessionId)
                .setSuccessBlocks(blockMask)
                .build();

        grpcClient.finishWriteCache(request);
    }
}
