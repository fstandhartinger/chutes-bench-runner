#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}/.."

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required to regenerate requirements.lock" >&2
    exit 1
fi

uv pip compile \
    --python-version 3.11 \
    --python-platform linux \
    --generate-hashes \
    --upgrade \
    --custom-compile-command './scripts/compile_requirements.sh' \
    --output-file requirements.lock \
    requirements.txt

sha256sum requirements.txt requirements.lock > requirements.lock.sha256
