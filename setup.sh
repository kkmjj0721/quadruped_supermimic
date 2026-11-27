#!/usr/bin/env bash
# Copyright (c) 2026-2027 zh
# 内容：测速并选择 Python、PyPI 和 PyTorch 镜像，同步锁定环境。
# 用法：bash setup.sh [--torch-backend auto|cpu|cu130] [镜像／测速参数] [-- uv sync 参数]

set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
torch_backend="auto"
download_args=()
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
    --probe-only|--skip-probes)
      download_args+=("$1")
      shift
      ;;
    --python-mirror|--pypi-mirror|--torch-mirror|--probe-timeout|--probe-bytes|--download-timeout)
      download_args+=("$1" "$2")
      shift 2
      ;;
    --python-mirror=*|--pypi-mirror=*|--torch-mirror=*|--probe-timeout=*|--probe-bytes=*|--download-timeout=*)
      download_args+=("$1")
      shift
      ;;
    -h|--help)
      cat <<'HELP'
Usage: bash setup.sh [--torch-backend auto|cpu|cu130] [OPTIONS] [-- UV_SYNC_ARGS...]

Select download mirrors and synchronize the locked project environment.
  --probe-only            Probe channels without installing project dependencies.
  --skip-probes           Use the original locked download URLs.
  --python-mirror ROOT    Python archive root (repeatable).
  --pypi-mirror ROOT      PyPI artifact root including /packages (repeatable).
  --torch-mirror ROOT     Torch artifact root preceding /cpu or /cu130 (repeatable).
  --probe-timeout SEC     Probe budget per sample (default: 8).
  --probe-bytes BYTES     Bytes per sample (default: 524288).
  --download-timeout SEC  Seconds per installation attempt (default: 1800).
Auto selects cu130 for CUDA Toolkit 13+ on PATH, otherwise cpu.
Install uv first: https://docs.astral.sh/uv/getting-started/installation/
Custom mirror lists replace the built-in candidates for that category.
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
exec python3 -B tools/setup_env.py --torch-backend "${torch_backend}" "${download_args[@]}" -- "${sync_args[@]}"
