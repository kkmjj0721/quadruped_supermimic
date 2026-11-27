# Copyright (c) 2026-2027 zh
"""
内容：
    测速安装文件镜像，按速度选择下载通道并准备固定版本的 uv Python。

用法：
    bash setup.sh --probe-only
    python3 tools/setup_downloads.py '["https://example.com/package.whl"]' 8 524288
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PYTHON_MIRRORS = (
    ("Astral", "https://releases.astral.sh/github/python-build-standalone/releases/download"),
    ("GitHub", "https://github.com/astral-sh/python-build-standalone/releases/download"),
    ("NJU", "https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone"),
    ("USTC", "https://mirrors.ustc.edu.cn/github-release/astral-sh/python-build-standalone"),
)
PYPI_MIRRORS = (
    ("PyPI", "https://files.pythonhosted.org/packages"),
    ("Tsinghua", "https://pypi.tuna.tsinghua.edu.cn/packages"),
    ("Aliyun", "https://mirrors.aliyun.com/pypi/packages"),
    ("Huawei", "https://repo.huaweicloud.com/repository/pypi/packages"),
)
TORCH_MIRRORS = (
    ("PyTorch R2", "https://download-r2.pytorch.org/whl"),
    ("PyTorch", "https://download.pytorch.org/whl"),
    ("Aliyun", "https://mirrors.aliyun.com/pytorch-wheels"),
    ("NJU", "https://mirror.nju.edu.cn/pytorch/whl"),
)


@dataclass(frozen=True)
class DownloadSource:
    name: str
    root: str
    urls: tuple[str, ...]


@dataclass(frozen=True)
class ProbeResult:
    source: DownloadSource
    elapsed: float
    bytes_read: int
    error: str | None = None

    @property
    def available(self) -> bool:
        return self.error is None and self.bytes_read > 0

    @property
    def speed(self) -> float:
        return self.bytes_read / self.elapsed if self.available else 0.0


# ---------- 前缀测速 ----------

def _probe_urls(urls: list[str], timeout: float, probe_bytes: int) -> dict:
    """子进程读取每个文件的有限前缀；父进程限制包含连接与响应头的总时长。"""
    started = time.monotonic()
    size = 0
    try:
        for url in urls:
            request = urllib.request.Request(url, headers={"Range": f"bytes=0-{probe_bytes - 1}"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read(probe_bytes)
                if not payload:
                    return {"elapsed": time.monotonic() - started, "bytes_read": size,
                            "error": f"Empty response: {url}"}
                size += len(payload)
    except (OSError, urllib.error.URLError) as exc:
        return {"elapsed": time.monotonic() - started, "bytes_read": size,
                "error": f"{type(exc).__name__}: {exc}"}
    return {"elapsed": time.monotonic() - started, "bytes_read": size, "error": None}


def probe_sources(
    sources: Iterable[DownloadSource], *, timeout: float = 8.0, probe_bytes: int = 524288, workers: int = 4,
) -> list[ProbeResult]:
    """并发测速；每源总墙钟上限为timeout乘文件数，排名排除解释器启动耗时。"""
    def probe(source: DownloadSource) -> ProbeResult:
        started = time.monotonic()
        command = [sys.executable, "-B", str(Path(__file__).resolve()), json.dumps(source.urls),
                   str(timeout), str(probe_bytes)]
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, text=True, check=True,
                                    timeout=timeout * len(source.urls))
        except subprocess.TimeoutExpired as exc:
            return ProbeResult(source, time.monotonic() - started, 0, str(exc))
        return ProbeResult(source, **json.loads(result.stdout))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(probe, sources))
    return sorted(results, key=lambda result: (not result.available, -result.speed))


def print_probes(title: str, results: Iterable[ProbeResult]) -> None:
    """输出各源的实际前缀吞吐与失败原因。"""
    print(title, flush=True)
    for index, result in enumerate(results, 1):
        status = (f"{result.speed / 1048576:.2f} MiB/s, {result.elapsed:.2f}s, {result.bytes_read} bytes"
                  if result.available else f"unavailable ({result.error})")
        print(f"  {index}. {result.source.name}: {result.source.root} - {status}", flush=True)


# ---------- Python 准备 ----------

def python_download_sources(uv: str, version: str, mirrors: list[str] | None = None) -> list[DownloadSource]:
    """使用uv提供的当前平台真实发行包地址构造镜像候选。"""
    result = subprocess.run(
        [uv, "python", "list", version, "--only-downloads", "--show-urls", "--output-format", "json"],
        stdout=subprocess.PIPE, text=True, check=True,
    )
    downloads = json.loads(result.stdout)
    download = next(item for item in downloads if item["implementation"] == "cpython"
                    and item["variant"] == "default"
                    and item["version"].split(".")[:len(version.split("."))] == version.split("."))
    suffix = download["url"].split("/releases/download/", 1)[1]
    if mirrors is not None:
        roots = [(f"Custom Python mirror {index}", root) for index, root in enumerate(mirrors, 1)]
    elif "UV_PYTHON_INSTALL_MIRROR" in os.environ:
        roots = [("Configured Python mirror", os.environ["UV_PYTHON_INSTALL_MIRROR"])]
    else:
        roots = PYTHON_MIRRORS
    return [DownloadSource(name, root.rstrip("/"), (f"{root.rstrip('/')}/{suffix}",)) for name, root in roots]


def ensure_python(
    uv: str, version: str, *, mirrors: list[str] | None = None, offline: bool = False,
    timeout: float = 8.0, probe_bytes: int = 524288, install_timeout: float = 600.0,
) -> str:
    """复用已安装解释器，或按测速排名安装；安装失败与超时继续下一源。"""
    find_command = [uv, "python", "find", version, "--managed-python", "--no-python-downloads", "--no-project", "--system"]
    found = subprocess.run(find_command, stdout=subprocess.PIPE, text=True, timeout=timeout)
    if found.returncode == 0:
        return found.stdout.strip()
    if offline or ("UV_OFFLINE" in os.environ and os.environ["UV_OFFLINE"].lower() in {"1", "true", "yes", "on"}):
        found.check_returncode()
    results = probe_sources(python_download_sources(uv, version, mirrors), timeout=timeout, probe_bytes=probe_bytes)
    print_probes(f"Python {version} download channels", results)
    available = [result for result in results if result.available]
    failures = []
    for index, result in enumerate(available, 1):
        print(f"Installing Python {version} ({index}/{len(available)}) from {result.source.root}", flush=True)
        try:
            subprocess.run([uv, "python", "install", version, "--mirror", result.source.root],
                           timeout=install_timeout, check=True)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            failures.append(exc)
            print(f"Python installation failed: {exc}; trying the next source.", flush=True)
        else:
            found = subprocess.run(find_command, stdout=subprocess.PIPE, text=True, timeout=timeout, check=True)
            return found.stdout.strip()
    raise failures[-1]


if __name__ == "__main__":
    print(json.dumps(_probe_urls(json.loads(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]))))
