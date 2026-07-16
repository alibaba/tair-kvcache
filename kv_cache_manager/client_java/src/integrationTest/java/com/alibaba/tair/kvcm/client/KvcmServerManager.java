package com.alibaba.tair.kvcm.client;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import kv_cache_manager.proto.meta.MetaServiceGrpc;
import kv_cache_manager.proto.meta.MetaServiceOuterClass.GetClusterInfoRequest;
import kv_cache_manager.proto.meta.MetaServiceOuterClass.GetClusterInfoResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.net.ServerSocket;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.concurrent.TimeUnit;

/**
 * Manages a KVCM server process for integration testing.
 * <p>
 * This class handles:
 * - Port allocation using ServerSocket(0)
 * - Server startup via ProcessBuilder with daemon mode
 * - Health check by polling getClusterInfo RPC
 * - Graceful shutdown with SIGTERM/SIGKILL
 * - Work directory management (stdout/stderr files)
 */
public class KvcmServerManager {

    private static final Logger LOG = LoggerFactory.getLogger(KvcmServerManager.class);
    private static final int HEALTH_CHECK_TIMEOUT_SECONDS = 30;
    private static final int HEALTH_CHECK_POLL_INTERVAL_MS = 100;
    private static final int SHUTDOWN_GRACE_PERIOD_SECONDS = 5;

    private final String binaryPath;
    private final Path workDir;

    private int rpcPort;
    private int httpPort;
    private int adminRpcPort;
    private int adminHttpPort;

    private Process serverProcess;
    private volatile boolean running = false;

    /**
     * Creates a new server manager.
     *
     * @throws IllegalStateException if KVCM_BIN environment variable is not set or invalid
     */
    public KvcmServerManager() {
        this.binaryPath = validateAndGetBinaryPath();
        this.workDir = createWorkDirectory();
    }

    /**
     * Starts the KVCM server and waits for it to become healthy.
     *
     * @throws IOException if server startup fails
     * @throws InterruptedException if interrupted while waiting for health check
     */
    public void start() throws IOException, InterruptedException {
        allocatePorts();

        ProcessBuilder pb = new ProcessBuilder(
                binaryPath,
                "--env", "kvcm.service.rpc_port=" + rpcPort,
                "--env", "kvcm.service.http_port=" + httpPort,
                "--env", "kvcm.service.admin_rpc_port=" + adminRpcPort,
                "--env", "kvcm.service.admin_http_port=" + adminHttpPort,
                "--env", "kvcm.logger.log_level=5",
                "-d"
        );

        pb.directory(workDir.toFile());
        pb.redirectOutput(workDir.resolve("stdout.log").toFile());
        pb.redirectError(workDir.resolve("stderr.log").toFile());

        LOG.info("Starting KVCM server with ports: rpc={}, http={}, admin_rpc={}, admin_http={}",
                rpcPort, httpPort, adminRpcPort, adminHttpPort);

        serverProcess = pb.start();
        running = true;

        // In daemon mode (-d), the parent process forks and exits quickly.
        // We can't rely on serverProcess.isAlive() to check if the daemon is running.
        // Instead, we proceed directly to the health check which will verify the server is actually running.

        waitForHealthy();
        LOG.info("KVCM server started successfully");
    }

    /**
     * Stops the KVCM server gracefully (SIGTERM) or forcefully (SIGKILL).
     */
    public void stop() {
        if (!running) {
            return;
        }

        LOG.info("Stopping KVCM server...");
        running = false;

        // In daemon mode, the Process object doesn't represent the actual daemon.
        // Find the daemon PID by matching the rpc port in the command line.
        try {
            String pid = findDaemonPid();
            if (pid != null) {
                // Try graceful shutdown first (SIGTERM)
                Process killProc = new ProcessBuilder("kill", "-TERM", pid).start();
                killProc.waitFor();

                // Wait for graceful exit
                long startTime = System.currentTimeMillis();
                boolean exited = false;
                while (System.currentTimeMillis() - startTime < SHUTDOWN_GRACE_PERIOD_SECONDS * 1000L) {
                    if (findDaemonPid() == null) {
                        exited = true;
                        break;
                    }
                    Thread.sleep(100);
                }

                if (!exited) {
                    LOG.warn("Server did not exit gracefully, forcing shutdown...");
                    new ProcessBuilder("kill", "-KILL", pid).start().waitFor();
                }
            }
        } catch (Exception e) {
            LOG.warn("Error stopping server: {}", e.getMessage());
        }

        // Also try to destroy the Process object (in case it's still alive)
        if (serverProcess != null) {
            serverProcess.destroyForcibly();
        }

        cleanupWorkDirectory();
        LOG.info("KVCM server stopped");
    }

    private String findDaemonPid() {
        try {
            Process pgrep = new ProcessBuilder("pgrep", "-f", "kv_cache_manager_bin.*rpc_port=" + rpcPort)
                    .start();
            byte[] output = pgrep.getInputStream().readAllBytes();
            pgrep.waitFor();
            String pidStr = new String(output).trim();
            if (!pidStr.isEmpty()) {
                // pgrep may return multiple PIDs, take the first one
                return pidStr.split("\n")[0];
            }
        } catch (Exception e) {
            // Ignore
        }
        return null;
    }

    public int getRpcPort() {
        return rpcPort;
    }

    public int getHttpPort() {
        return httpPort;
    }

    public int getAdminRpcPort() {
        return adminRpcPort;
    }

    public int getAdminHttpPort() {
        return adminHttpPort;
    }

    public Path getWorkDir() {
        return workDir;
    }

    // --- Private helper methods ---

    private static String validateAndGetBinaryPath() {
        String path = System.getenv("KVCM_BIN");

        if (path == null || path.trim().isEmpty()) {
            throw new IllegalStateException(
                    "KVCM_BIN environment variable not set. " +
                    "Please set it to the path of kv_cache_manager_bin binary.");
        }

        File binaryFile = new File(path);

        if (!binaryFile.exists()) {
            throw new IllegalStateException(
                    "Server binary not found at: " + path + "\n" +
                    "Please ensure KVCM_BIN points to a valid kv_cache_manager_bin binary.");
        }

        if (!binaryFile.canExecute()) {
            throw new IllegalStateException(
                    "Server binary is not executable: " + path + "\n" +
                    "Please ensure the file has execute permissions.");
        }

        LOG.info("Using KVCM binary: {}", path);
        return binaryFile.getAbsoluteFile().getAbsolutePath();
    }

    private static Path createWorkDirectory() {
        try {
            Path tempDir = Files.createTempDirectory("kvcm-integration-test-");
            LOG.info("Created work directory: {}", tempDir);
            return tempDir;
        } catch (IOException e) {
            throw new RuntimeException("Failed to create work directory", e);
        }
    }

    private void allocatePorts() throws IOException {
        rpcPort = allocatePort();
        httpPort = allocatePort();
        adminRpcPort = allocatePort();
        adminHttpPort = allocatePort();
    }

    private static int allocatePort() throws IOException {
        try (ServerSocket socket = new ServerSocket(0)) {
            socket.setReuseAddress(true);
            return socket.getLocalPort();
        }
    }

    private void waitForHealthy() throws IOException, InterruptedException {
        long startTime = System.currentTimeMillis();
        long timeoutMillis = HEALTH_CHECK_TIMEOUT_SECONDS * 1000L;

        ManagedChannel channel = null;
        MetaServiceGrpc.MetaServiceBlockingStub stub = null;

        try {
            channel = ManagedChannelBuilder.forAddress("localhost", rpcPort)
                    .usePlaintext()
                    .build();
            stub = MetaServiceGrpc.newBlockingStub(channel);

            while (System.currentTimeMillis() - startTime < timeoutMillis) {
                try {
                    GetClusterInfoResponse response = stub
                            .withDeadlineAfter(1, TimeUnit.SECONDS)
                            .getClusterInfo(GetClusterInfoRequest.newBuilder()
                                    .setTraceId("health-check")
                                    .build());

                    if (response.hasHeader() && response.getHeader().hasStatus() &&
                            response.getHeader().getStatus().getCode() == kv_cache_manager.proto.meta.MetaServiceOuterClass.ErrorCode.OK) {
                        LOG.info("Server health check passed after {} ms",
                                System.currentTimeMillis() - startTime);
                        return;
                    }
                } catch (Exception e) {
                    // Server not ready yet, continue polling
                }

                Thread.sleep(HEALTH_CHECK_POLL_INTERVAL_MS);
            }

            throw new IOException(
                    "Server failed to start within " + HEALTH_CHECK_TIMEOUT_SECONDS + " seconds. " +
                    "Check " + workDir.resolve("stderr.log") + " for details.");

        } finally {
            if (channel != null) {
                channel.shutdownNow();
                channel.awaitTermination(1, TimeUnit.SECONDS);
            }
        }
    }

    private void cleanupWorkDirectory() {
        try {
            Files.walk(workDir)
                    .sorted(Comparator.reverseOrder())
                    .map(Path::toFile)
                    .forEach(File::delete);
            LOG.info("Cleaned up work directory: {}", workDir);
        } catch (IOException e) {
            LOG.warn("Failed to cleanup work directory: {}", e.getMessage());
        }
    }
}
