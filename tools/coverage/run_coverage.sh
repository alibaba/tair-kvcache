#!/usr/bin/env bash

set -euo pipefail

if [[ "${COVERAGE_XTRACE:-0}" == "1" ]]; then
  set -x
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_WORKSPACE="$(cd "${SCRIPT_DIR}/../.." && pwd)"

WORKSPACE="${DEFAULT_WORKSPACE}"
BASE_REF="origin/main"
HEAD_REF="HEAD"
OUTPUT_DIR="coverage"
INCLUDE_PREFIX="kv_cache_manager/"
JOBS="8"
LOCAL_TEST_JOBS="8"
TEST_TIMEOUT="900"
TEST_OUTPUT="errors"
FETCH_MAIN="0"
GENERATE_HTML="1"
TARGETS=()

usage() {
  cat <<'EOF'
Usage: tools/coverage/run_coverage.sh [options] -- <bazel targets...>

Options:
  --workspace PATH        Repository workspace. Defaults to the repo root.
  --base-ref REF          Base ref for incremental coverage. Defaults to origin/main.
  --head-ref REF          Head ref for incremental coverage. Defaults to HEAD.
  --output-dir DIR        Output directory. Defaults to coverage.
  --include-prefix PATH   Include files under this path in summaries. Defaults to kv_cache_manager/.
  --jobs N                Bazel build jobs. Defaults to 8.
  --local-test-jobs N     Bazel local test jobs. Defaults to 8.
  --test-timeout SECONDS  Bazel test timeout. Defaults to 900.
  --test-output MODE      Bazel test output mode. Defaults to errors.
  --fetch-main            Run "git fetch origin main --no-tags --prune" before coverage.
  --no-html               Skip genhtml output.
  -h, --help              Show this help.

Environment:
  KVCM_REAL_GCOV          Real gcov binary. Auto-detected when unset.
  GCOV                    gcov wrapper. Defaults to tools/coverage/gcov_json_isolated.sh.
  COVERAGE_GCOV_OPTIONS   Extra gcov options. Defaults to -b for branch coverage.
  COVERAGE_XTRACE=1       Enable shell xtrace.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace)
      WORKSPACE="$2"
      shift 2
      ;;
    --base-ref)
      BASE_REF="$2"
      shift 2
      ;;
    --head-ref)
      HEAD_REF="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --include-prefix)
      INCLUDE_PREFIX="$2"
      shift 2
      ;;
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    --local-test-jobs)
      LOCAL_TEST_JOBS="$2"
      shift 2
      ;;
    --test-timeout)
      TEST_TIMEOUT="$2"
      shift 2
      ;;
    --test-output)
      TEST_OUTPUT="$2"
      shift 2
      ;;
    --fetch-main)
      FETCH_MAIN="1"
      shift
      ;;
    --no-html)
      GENERATE_HTML="0"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      TARGETS=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ${#TARGETS[@]} -eq 0 ]]; then
  TARGETS=(//kv_cache_manager/... //integration_test/...)
fi

cd "${WORKSPACE}"

if [[ -z "${BASE_REF}" || "${BASE_REF}" =~ ^0+$ ]]; then
  BASE_REF="origin/main"
fi

if [[ -n "${GITHUB_WORKSPACE:-}" ]]; then
  git config --global --add safe.directory "${GITHUB_WORKSPACE}"
fi

if [[ "${FETCH_MAIN}" == "1" ]]; then
  git fetch origin main --no-tags --prune
fi

if [[ -z "${KVCM_REAL_GCOV:-}" ]]; then
  GCC_MAJOR="$(gcc -dumpversion | cut -d. -f1)"
  KVCM_REAL_GCOV="$(command -v "gcov-${GCC_MAJOR}" || command -v gcov)"
fi

GCOV="${GCOV:-${WORKSPACE}/tools/coverage/gcov_json_isolated.sh}"
COVERAGE_GCOV_OPTIONS="${COVERAGE_GCOV_OPTIONS:--b}"

echo "gcc: $(gcc --version | head -1)"
echo "gcov: $(${KVCM_REAL_GCOV} --version | head -1)"
echo "gcov wrapper: ${GCOV}"
echo "coverage base: ${BASE_REF}"
echo "coverage head: ${HEAD_REF}"

rm -rf "${OUTPUT_DIR}" bazel-testlogs

bazelisk coverage \
  --config=debug \
  --config=ci_fast \
  --combined_report=lcov \
  --instrumentation_filter="^//kv_cache_manager" \
  --action_env=GCOV="${GCOV}" \
  --test_env=GCOV="${GCOV}" \
  --action_env=KVCM_REAL_GCOV="${KVCM_REAL_GCOV}" \
  --test_env=KVCM_REAL_GCOV="${KVCM_REAL_GCOV}" \
  --test_env=COVERAGE_GCOV_OPTIONS="${COVERAGE_GCOV_OPTIONS}" \
  --test_timeout="${TEST_TIMEOUT}" \
  --jobs="${JOBS}" \
  --local_test_jobs="${LOCAL_TEST_JOBS}" \
  --cache_test_results=no \
  --test_output="${TEST_OUTPUT}" \
  "${TARGETS[@]}"

mkdir -p "${OUTPUT_DIR}"
LCOV_INFO="${OUTPUT_DIR}/lcov.info"
cp bazel-out/_coverage/_coverage_report.dat "${LCOV_INFO}"
sed -E -i \
  -e 's/^(DA:[0-9]+),-[0-9]+/\1,0/' \
  -e 's/^(BRDA:[^,]+,[^,]+,[^,]+),-[0-9]+/\1,0/' \
  "${LCOV_INFO}"

python3 "${SCRIPT_DIR}/coverage_report.py" \
  --lcov "${LCOV_INFO}" \
  --workspace "${WORKSPACE}" \
  --base-ref "${BASE_REF}" \
  --head-ref "${HEAD_REF}" \
  --include-prefix "${INCLUDE_PREFIX}" \
  --output-dir "${OUTPUT_DIR}"

if [[ "${GENERATE_HTML}" == "1" ]]; then
  if ! command -v genhtml >/dev/null 2>&1; then
    echo "genhtml not found; install lcov or rerun with --no-html" >&2
    exit 1
  fi
  GENHTML_ARGS=(
    "${LCOV_INFO}"
    --output-directory "${OUTPUT_DIR}/html"
    --title "tair-kvcache coverage"
    --legend
    --show-details
    --ignore-errors source
  )
  if grep -q '^BRDA:' "${LCOV_INFO}"; then
    GENHTML_ARGS+=(--branch-coverage)
  fi
  genhtml "${GENHTML_ARGS[@]}"
fi
