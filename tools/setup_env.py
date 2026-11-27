# Copyright (c) 2026-2027 zh
"""
内容：
    选择 Python、PyPI 和 PyTorch 镜像，通过持久下载项目安装锁定环境。

用法：
    bash setup.sh --torch-backend cu130
    python3 tools/setup_env.py --torch-backend cu130 --probe-only
"""

from __future__ import annotations

import argparse
from itertools import product
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import unquote, urlsplit

from setup_downloads import (
    DownloadSource, PYPI_MIRRORS, TORCH_MIRRORS,
    ensure_python, print_probes, probe_sources, python_download_sources,
)


# ---------- 锁定文件与镜像 ----------

def read_toml(text: str) -> dict:
    import tomllib

    return tomllib.loads(text)


def select_wheel(lock: dict, name: str, backend: str | None = None) -> dict:
    package = next(package for package in lock["package"]
                   if package["name"] == name and (backend is None or package["version"].endswith("+" + backend)))
    return next(wheel for wheel in package["wheels"]
                if "cp311-cp311-" in unquote(wheel["url"]) and "linux" in wheel["url"] and "x86_64" in wheel["url"])


def candidates(roots: list[str] | None, defaults: tuple) -> list[tuple[str, str]]:
    values = defaults if roots is None else [(f"custom-{index}", root) for index, root in enumerate(roots, 1)]
    return [(name, root.rstrip("/")) for name, root in values]


def package_sources(lock: dict, backend: str, pypi_roots: list[str] | None, torch_roots: list[str] | None):
    suffix = select_wheel(lock, "numpy")["url"].split("/packages/", 1)[1]
    pypi = [DownloadSource(name, root, (root + "/" + suffix,)) for name, root in candidates(pypi_roots, PYPI_MIRRORS)]
    filenames = [urlsplit(select_wheel(lock, name, backend)["url"]).path.rsplit("/", 1)[-1]
                 for name in ("torch", "torchvision")]
    torch = [DownloadSource(name, root, tuple(root + "/" + backend + "/" + filename for filename in filenames))
             for name, root in candidates(torch_roots, TORCH_MIRRORS)]
    return pypi, torch


def rewrite_lock(text: str, backend: str, pypi_root: str, torch_root: str) -> str:
    """只替换下载URL，依赖解析结果及文件校验信息直接沿用原锁。"""
    for package in read_toml(text)["package"]:
        artifacts = list(package["wheels"]) if "wheels" in package else []
        if "sdist" in package:
            artifacts.append(package["sdist"])
        for artifact in artifacts:
            original = artifact["url"]
            if package["name"] in {"torch", "torchvision"}:
                if not package["version"].endswith("+" + backend):
                    continue
                replacement = torch_root + "/" + backend + "/" + urlsplit(original).path.rsplit("/", 1)[-1]
            elif "/packages/" in original:
                replacement = pypi_root + "/" + original.split("/packages/", 1)[1]
            else:
                continue
            text = text.replace('url = "' + original + '"', 'url = "' + replacement + '"')
    return text


# ---------- 环境安装 ----------

def run_sync(command: list[str], *, cwd: Path, env: dict, timeout: float) -> int:
    """每轮只启动一个uv进程，超时由subprocess结束后再切换镜像。"""
    try:
        return subprocess.run(command, cwd=cwd, env=env, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        print(f"Download attempt exceeded {timeout:g} seconds; trying the next source.", flush=True)
        return 124


def direct_command(uv: str, project: Path, python_version: str, backend: str, sync_args: list[str]) -> list[str]:
    return [uv, "sync", "--project", str(project), "--python", python_version, "--locked", "--extra", backend, *sync_args]


def synchronize(project: Path, args, python_version: str, lock_text: str, pypi: list, torch: list) -> int:
    """复用一个可检查的下载项目；每次直接写入选定源，不改原项目文件。"""
    download_project = project / ".artifacts" / "setup-downloads" / args.torch_backend
    download_project.mkdir(parents=True, exist_ok=True)
    for filename in ("pyproject.toml", ".python-version", "README.md"):
        (download_project / filename).write_bytes((project / filename).read_bytes())
    env = os.environ.copy()
    env["UV_PROJECT_ENVIRONMENT"] = str(project / ".venv")
    pairs = sorted(product(enumerate(pypi), enumerate(torch)), key=lambda pair: pair[0][0] + pair[1][0])
    code = 1
    for attempt, ((_, pypi_result), (_, torch_result)) in enumerate(pairs, 1):
        pypi_root, torch_root = pypi_result.source.root, torch_result.source.root
        print(f"Synchronizing source combination {attempt}/{len(pairs)}: PyPI={pypi_root}; Torch={torch_root}", flush=True)
        (download_project / "uv.lock").write_text(rewrite_lock(lock_text, args.torch_backend, pypi_root, torch_root), encoding="utf-8")
        command = [args.uv, "sync", "--project", str(download_project), "--python", python_version,
                   "--frozen", "--extra", args.torch_backend, *args.sync_args]
        code = run_sync(command, cwd=project, env=env, timeout=args.download_timeout)
        if code == 0:
            return 0
        print(f"Synchronization failed with exit code {code}; trying the next source combination.", flush=True)
    return code


# ---------- 入口 ----------

def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Select download mirrors and install the locked project environment.")
    result.add_argument("--uv", default="uv", help=argparse.SUPPRESS)
    result.add_argument("--torch-backend", choices=("cpu", "cu130"), required=True)
    modes = result.add_mutually_exclusive_group()
    modes.add_argument("--skip-probes", action="store_true", help="Use original locked URLs without probing.")
    modes.add_argument("--probe-only", action="store_true", help="Probe mirrors without installing project dependencies.")
    result.add_argument("--python-mirror", action="append", help="Python standalone mirror root (repeatable).")
    result.add_argument("--pypi-mirror", action="append", help="PyPI artifact root including /packages (repeatable).")
    result.add_argument("--torch-mirror", action="append", help="Torch artifact root preceding /cpu or /cu130 (repeatable).")
    result.add_argument("--probe-timeout", type=float, default=8., help="Probe budget per sample, in seconds (default: 8).")
    result.add_argument("--probe-bytes", type=int, default=524288, help="Bytes read per sample (default: 524288).")
    result.add_argument("--download-timeout", type=float, default=1800., help="Seconds per installation attempt (default: 1800).")
    result.add_argument("--python-ready", help=argparse.SUPPRESS)
    result.add_argument("sync_args", nargs=argparse.REMAINDER, help="uv sync arguments after --.")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.sync_args[:1] == ["--"]:
        args.sync_args = args.sync_args[1:]
    project = Path(__file__).resolve().parents[1]
    python_version = (project / ".python-version").read_text().strip()
    env = os.environ.copy()
    env["UV_PROJECT_ENVIRONMENT"] = str(project / ".venv")
    command = direct_command(args.uv, project, python_version, args.torch_backend, args.sync_args)
    offline = ("--offline" in args.sync_args or
               "UV_OFFLINE" in os.environ and os.environ["UV_OFFLINE"].lower() in {"1", "true"})
    if args.probe_only and offline:
        print("Offline mode: no network probes or installations were run.", flush=True)
        return 0
    if not args.probe_only and (args.skip_probes or offline or any(flag in args.sync_args for flag in ("--check", "--dry-run"))):
        return run_sync(command, cwd=project, env=env, timeout=args.download_timeout)
    if not args.probe_only and not args.sync_args:
        installed = subprocess.run([*command, "--check", "--offline"], cwd=project, env=env, capture_output=True)
        if installed.returncode == 0:
            print("The environment already matches uv.lock; skipping download probes.", flush=True)
            return 0
    python_path = args.python_ready if args.python_ready is not None else ensure_python(
        args.uv, python_version, mirrors=args.python_mirror, offline=args.probe_only,
        timeout=args.probe_timeout, probe_bytes=args.probe_bytes, install_timeout=args.download_timeout)
    if Path(sys.executable).resolve() != Path(python_path).resolve():
        os.execv(python_path, [python_path, "-B", str(Path(__file__).resolve()), "--python-ready", python_path, *sys.argv[1:]])
    lock_text = (project / "uv.lock").read_text()
    subprocess.run([args.uv, "lock", "--project", str(project), "--check", "--python", python_version],
                   cwd=project, env=env, check=True)
    if args.probe_only:
        python_results = probe_sources(python_download_sources(args.uv, python_version, args.python_mirror),
                                      timeout=args.probe_timeout, probe_bytes=args.probe_bytes)
        print_probes("Python archive channels", python_results)
    pypi_sources, torch_sources = package_sources(read_toml(lock_text), args.torch_backend, args.pypi_mirror, args.torch_mirror)
    pypi_results = probe_sources(pypi_sources, timeout=args.probe_timeout, probe_bytes=args.probe_bytes)
    torch_results = probe_sources(torch_sources, timeout=args.probe_timeout, probe_bytes=args.probe_bytes)
    print_probes("PyPI wheel channels", pypi_results)
    print_probes("Torch and torchvision wheel channels", torch_results)
    pypi = [result for result in pypi_results if result.available]
    torch = [result for result in torch_results if result.available]
    if args.probe_only:
        return 0 if pypi and torch and any(result.available for result in python_results) else 1
    return synchronize(project, args, python_version, lock_text, pypi, torch)


if __name__ == "__main__":
    raise SystemExit(main())
