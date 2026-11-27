#!/usr/bin/env bash
# Copyright (c) 2026-2027 zh
# 内容：使用 uv 锁文件同步指定 PyTorch 后端的项目环境。
# 用法：bash setup.sh [--torch-backend auto|cpu|cu130] [-- uv sync 参数]

set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
torch_backend="auto"
sync_args=()

while (($#)); do
  case "$1" in
    --torch-backend)
      torch_backend="$2"
      shift 2
      ;;
    --torch-backend=*)
      torch_backend="${1#*=}"
      shift
      ;;
    -h|--help)
      cat <<'HELP'
Usage: bash setup.sh [--torch-backend auto|cpu|cu130] [-- UV_SYNC_ARGS...]

Run uv sync --locked --extra BACKEND in this project.
Auto selects cu130 for CUDA Toolkit 13+ on PATH, otherwise cpu.
Install uv first: https://docs.astral.sh/uv/getting-started/installation/
Configure mirrors and proxies with uv's environment variables.
HELP
      exit 0
      ;;
    --)
      shift
      sync_args+=("$@")
      break
      ;;
    *)
      sync_args+=("$1")
      shift
      ;;
  esac
done

if [[ "${torch_backend}" == "auto" ]]; then
  torch_backend="cpu"
  if command -v nvcc >/dev/null 2>&1; then
    nvcc_output="$(nvcc --version)"
    [[ "${nvcc_output}" =~ release[[:space:]]+([0-9]+)\.([0-9]+) ]]
    if ((10#${BASH_REMATCH[1]} >= 13)); then
      torch_backend="cu130"
    fi
  fi
fi

cd -- "${project_root}"
exec uv sync --locked --extra "${torch_backend}" "${sync_args[@]}"
