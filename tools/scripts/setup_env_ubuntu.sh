#!/usr/bin/env bash

set -euo pipefail

BAZEL_VERSION="${BAZEL_VERSION:-6.4.0}"
BAZEL_BASE_URL="${BAZEL_BASE_URL:-https://mirrors.huaweicloud.com/bazel}"
BAZEL_SHA256="${BAZEL_SHA256:-}"
INSTALL_PREFIX="${INSTALL_PREFIX:-/usr/local/bin}"
RUN_BUILD="${RUN_BUILD:-0}"
INSTALL_BUILDIFIER="${INSTALL_BUILDIFIER:-0}"
BUILDIFIER_VERSION="${BUILDIFIER_VERSION:-8.2.1}"
BUILDIFIER_SHA256="${BUILDIFIER_SHA256:-}"
CURL_MAX_TIME="${CURL_MAX_TIME:-600}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

usage() {
    cat <<USAGE
Usage: tools/scripts/setup_env_ubuntu.sh [--build] [--no-build]

Installs the Ubuntu packages and Bazel version needed to build Tair KVCache
Manager in this container.

Environment overrides:
  BAZEL_VERSION         Bazel version to install. Default: ${BAZEL_VERSION}
  BAZEL_BASE_URL        Bazel download base URL. Default: ${BAZEL_BASE_URL}
  BAZEL_SHA256          Expected Bazel binary SHA256 for custom versions.
  INSTALL_PREFIX        Tool install directory. Default: ${INSTALL_PREFIX}
  INSTALL_BUILDIFIER=1  Also install buildifier from GitHub releases.
  BUILDIFIER_SHA256     Expected buildifier SHA256 for custom versions.
  CURL_MAX_TIME         Max seconds per download. Default: ${CURL_MAX_TIME}
  RUN_BUILD=1           Run bazelisk build //kv_cache_manager:main after setup.
USAGE
}

run_root() {
    if [[ "$(id -u)" == "0" ]]; then
        "$@"
    else
        sudo "$@"
    fi
}

run_root_env() {
    if [[ "$(id -u)" == "0" ]]; then
        env "$@"
    else
        sudo env "$@"
    fi
}

parse_args() {
    for arg in "$@"; do
        case "${arg}" in
            --build)
                RUN_BUILD=1
                ;;
            --no-build)
                RUN_BUILD=0
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                echo "Unknown argument: ${arg}" >&2
                usage >&2
                exit 2
                ;;
        esac
    done
}

install_apt_packages() {
    if ! command -v apt-get >/dev/null 2>&1; then
        echo "apt-get is required by this setup script." >&2
        exit 1
    fi

    local packages=(
        build-essential
        ca-certificates
        clang-format
        cpio
        curl
        git
        iproute2
        libaio-dev
        libibverbs-dev
        libicu-dev
        libnuma-dev
        librdmacm-dev
        openssh-client
        patchelf
        pigz
        procps
        python3
        python3-autopep8
        python3-dev
        python3-packaging
        python3-pip
        rpm2cpio
        tar
        unzip
        vim
        wget
        zip
    )

    run_root apt-get update
    run_root_env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${packages[@]}"
}

current_bazel_version() {
    local bazel_cmd=""
    if command -v bazelisk >/dev/null 2>&1; then
        bazel_cmd="bazelisk"
    elif command -v bazel >/dev/null 2>&1; then
        bazel_cmd="bazel"
    else
        return 0
    fi

    "${bazel_cmd}" version 2>/dev/null | awk -F': ' '/Build label:/ {print $2; exit}'
}

bazel_arch() {
    case "$(uname -m)" in
        x86_64|amd64)
            echo "x86_64"
            ;;
        aarch64|arm64)
            echo "arm64"
            ;;
        *)
            echo "Unsupported architecture: $(uname -m)" >&2
            exit 1
            ;;
    esac
}

default_bazel_sha256() {
    local arch="$1"
    case "${BAZEL_VERSION}:${arch}" in
        6.4.0:x86_64)
            echo "79e4f370efa6e31717b486af5d9efd95864d0ef13da138582224ac9b2a1bad86"
            ;;
        6.4.0:arm64)
            echo "1df1147765ad4aa23d7f12045b8e8d21b47db40525de69c877ac49234bf2d22d"
            ;;
        *)
            return 1
            ;;
    esac
}

default_buildifier_sha256() {
    local arch="$1"
    case "${BUILDIFIER_VERSION}:${arch}" in
        8.2.1:amd64)
            echo "6ceb7b0ab7cf66fceccc56a027d21d9cc557a7f34af37d2101edb56b92fcfa1a"
            ;;
        8.2.1:arm64)
            echo "3baa1cf7eb41d51f462fdd1fff3a6a4d81d757275d05b2dd5f48671284e9a1a5"
            ;;
        *)
            return 1
            ;;
    esac
}

download_checked() {
    local url="$1"
    local output="$2"
    local expected_sha256="$3"

    curl -fL --retry 3 --connect-timeout 20 --max-time "${CURL_MAX_TIME}" "${url}" -o "${output}"
    printf "%s  %s\n" "${expected_sha256}" "${output}" | sha256sum -c -
}

install_bazel() {
    local current
    current="$(current_bazel_version || true)"
    if [[ "${current}" == "${BAZEL_VERSION}" ]]; then
        echo "Bazel ${BAZEL_VERSION} already installed."
        return
    fi

    local arch
    arch="$(bazel_arch)"
    local url="${BAZEL_BASE_URL%/}/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-linux-${arch}"
    local expected_sha256="${BAZEL_SHA256}"
    if [[ -z "${expected_sha256}" ]]; then
        if ! expected_sha256="$(default_bazel_sha256 "${arch}")"; then
            echo "No built-in Bazel SHA256 for ${BAZEL_VERSION} on ${arch}; set BAZEL_SHA256." >&2
            exit 1
        fi
    fi

    local tmp
    tmp="$(mktemp)"
    trap 'rm -f "${tmp}"' RETURN

    echo "Installing Bazel ${BAZEL_VERSION} from ${url}"
    download_checked "${url}" "${tmp}" "${expected_sha256}"
    run_root install -m 0755 "${tmp}" "${INSTALL_PREFIX}/bazel"
    run_root ln -sf "${INSTALL_PREFIX}/bazel" "${INSTALL_PREFIX}/bazelisk"
    rm -f "${tmp}"
    trap - RETURN
}

install_buildifier() {
    if [[ "${INSTALL_BUILDIFIER}" != "1" ]]; then
        return
    fi

    local arch
    case "$(uname -m)" in
        x86_64|amd64)
            arch="amd64"
            ;;
        aarch64|arm64)
            arch="arm64"
            ;;
        *)
            echo "Unsupported architecture for buildifier: $(uname -m)" >&2
            exit 1
            ;;
    esac

    local url="https://github.com/bazelbuild/buildtools/releases/download/v${BUILDIFIER_VERSION}/buildifier-linux-${arch}"
    local expected_sha256="${BUILDIFIER_SHA256}"
    if [[ -z "${expected_sha256}" ]]; then
        if ! expected_sha256="$(default_buildifier_sha256 "${arch}")"; then
            echo "No built-in buildifier SHA256 for ${BUILDIFIER_VERSION} on ${arch}; set BUILDIFIER_SHA256." >&2
            exit 1
        fi
    fi

    local tmp
    tmp="$(mktemp)"
    trap 'rm -f "${tmp}"' RETURN

    echo "Installing buildifier ${BUILDIFIER_VERSION} from ${url}"
    download_checked "${url}" "${tmp}" "${expected_sha256}"
    run_root install -m 0755 "${tmp}" "${INSTALL_PREFIX}/buildifier"
    rm -f "${tmp}"
    trap - RETURN
}

verify_build() {
    if [[ "${RUN_BUILD}" != "1" ]]; then
        return
    fi

    cd "${REPO_ROOT}"
    bazelisk build //kv_cache_manager:main
}

main() {
    parse_args "$@"
    install_apt_packages
    install_bazel
    install_buildifier
    verify_build
}

main "$@"
