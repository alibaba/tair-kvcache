#!/usr/bin/env bash

set -u

real_gcov="${KVCM_REAL_GCOV:-}"
if [[ -z "${real_gcov}" ]]; then
  echo "KVCM_REAL_GCOV is not set" >&2
  exit 1
fi

if [[ -z "${COVERAGE_DIR:-}" ]]; then
  exec "${real_gcov}" "$@"
fi

caller_pwd="${PWD}"
tmpdir="$(mktemp -d "${COVERAGE_DIR}/gcov-json.XXXXXX")"
status=0

(
  cd "${tmpdir}" || exit 1
  "${real_gcov}" "$@"
) || status=$?

gcda=""
for arg in "$@"; do
  if [[ "${arg}" == *.gcda ]]; then
    gcda="${arg}"
  fi
done

if [[ -n "${gcda}" && "${gcda}" == "${COVERAGE_DIR}/"* ]]; then
  relative_gcda="${gcda#${COVERAGE_DIR}/}"
  dest_dir="${COVERAGE_DIR}/$(dirname "${relative_gcda}")"
  mkdir -p "${dest_dir}"
else
  dest_dir="${caller_pwd}"
fi

shopt -s nullglob
json_files=("${tmpdir}"/*.gcov.json.gz)
if ((${#json_files[@]} > 0)); then
  mv -- "${json_files[@]}" "${dest_dir}/"
fi

text_files=("${tmpdir}"/*.gcov)
if ((${#text_files[@]} > 0)); then
  mv -- "${text_files[@]}" "${caller_pwd}/"
fi
shopt -u nullglob

rm -rf "${tmpdir}"
exit "${status}"
