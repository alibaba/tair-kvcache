# Process extensions

The server keeps its existing `main → CommandLine::Run → Server` entry point.
`ProcessExtension` supports optional process-scoped services through the existing
`stub_source` build boundary. The default open-source factory returns a no-op;
building and running the open-source server requires no other implementation.

The interface is in `kv_cache_manager/service/process_extension.h`. An extension
implements `Start(context)` and `Stop() noexcept`. The context contains the
configuration filename and command-line overrides. Extension-specific parsing,
dependencies and configuration semantics belong to the selected implementation,
not to `ServerConfig` or the storage manager. An embedding application can also
provide an extension to the `CommandLine` constructor.

Lifecycle order:

1. Parse the normal arguments/configuration, daemonize if requested, initialize logging.
2. Create and start the extension.
3. Initialize, start and wait for the original Server.
4. Stop, join and destroy the Server, including active election callbacks.
5. Stop and destroy the extension, then destroy logging.

Failed or throwing extension startup prevents Server initialization and still
calls `Stop`. Server initialization/start failure uses the same cleanup order.
`Stop` must drain outstanding work, tolerate partial startup and repeated calls,
and must not throw. The host serializes lifecycle calls. Extensions must not
daemonize or install process signal handlers. Shard leadership changes and
individual storage lifetimes do not start or stop the process extension.

Termination signal handlers only record a request; the normal Run thread
shuts down and joins the Server before resource destruction. A
termination request received during extension startup is retained until Server
startup completes. This does not cancel an extension blocked inside `Start`;
implementations should bound their own initialization operations.

The `stub_source/bazel/server_build.bzl` file supplies optional final server
linker features/options. Both lists are empty for the open-source build.
It does not replace the main function or the existing server product targets.

Regression targets: `ProcessExtensionTest`, `CommandLineTest`,
`ServerConfigTest`, and `ServerEventPublisherTest` under
`//kv_cache_manager/service/test`.
