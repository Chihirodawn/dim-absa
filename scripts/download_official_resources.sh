#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
resource_root="${project_root}/resources/DimABSA2026"
proxy_url="${PROXY_URL:-}"

if [[ -e "${resource_root}" ]]; then
  echo "Refusing to overwrite existing resources: ${resource_root}" >&2
  exit 1
fi

if [[ -n "${proxy_url}" ]]; then
  HTTPS_PROXY="${proxy_url}" HTTP_PROXY="${proxy_url}" ALL_PROXY="${proxy_url}" \
    git clone --depth 1 https://github.com/DimABSA/DimABSA2026.git "${resource_root}"
else
  git clone --depth 1 https://github.com/DimABSA/DimABSA2026.git "${resource_root}"
fi
git -C "${resource_root}" rev-parse HEAD
