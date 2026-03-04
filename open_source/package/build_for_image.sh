#!/bin/bash

bazelisk build //package:kv_cache_manager_server --config=client_with_cuda
# bazel-bin/package/kv_cache_manager_server.tar.gz