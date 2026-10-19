"""Startup timings and stack traces without changing model execution."""

from __future__ import annotations

import faulthandler
import logging
import math
import os
import platform
import sys
import time
from threading import Event, Thread, TIMEOUT_MAX
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

logger = logging.getLogger(__name__)


def memory_summary() -> str:
    """Best-effort process, container and already-initialized CUDA metrics."""
    metrics = [f"pid={os.getpid()}"]
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                metrics.append(f"rss_kib={line.split()[1]}")
                break
    except (OSError, ValueError, IndexError):
        pass
    for label, paths in (
        ("cgroup_used_bytes", ("memory.current", "memory/memory.usage_in_bytes")),
        ("cgroup_limit_bytes", ("memory.max", "memory/memory.limit_in_bytes")),
        ("cgroup_events", ("memory.events", "memory/memory.failcnt")),
    ):
        for relative in paths:
            try:
                value = (Path("/sys/fs/cgroup") / relative).read_text().strip()
            except OSError:
                continue
            metrics.append(f"{label}={value.replace(chr(10), ',')}")
            break
    # Never initialize CUDA just to measure it (CPU offload may not have used it).
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_initialized():
            free, total = torch.cuda.mem_get_info()
            metrics.extend(
                (
                    f"cuda_free_bytes={free}",
                    f"cuda_total_bytes={total}",
                    f"cuda_allocated_bytes={torch.cuda.memory_allocated()}",
                    f"cuda_reserved_bytes={torch.cuda.memory_reserved()}",
                )
            )
    except Exception:
        # Diagnostics must not hide the original error, including CUDA errors.
        pass
    return " ".join(metrics)


@contextmanager
def log_stage(name: str):
    start = time.monotonic()
    logger.info("stage=%s begin %s", name, memory_summary())
    try:
        yield
    except BaseException:
        logger.exception(
            "stage=%s failed elapsed_s=%.2f %s",
            name,
            time.monotonic() - start,
            memory_summary(),
        )
        raise
    else:
        logger.info(
            "stage=%s done elapsed_s=%.2f %s",
            name,
            time.monotonic() - start,
            memory_summary(),
        )


@contextmanager
def startup_diagnostics():
    raw = os.environ.get("QWEN_STARTUP_TRACE_INTERVAL", "60")
    try:
        interval = float(raw)
        if not math.isfinite(interval) or interval < 0:
            raise ValueError
    except ValueError:
        logger.warning("Invalid QWEN_STARTUP_TRACE_INTERVAL=%r; using 60 seconds", raw)
        interval = 60.0
    cancelled = Event()
    watchdog = None
    if interval:
        try:
            # The native faulthandler timer walks frames without the GIL and
            # can race with frame destruction on Python 3.12. Call the direct
            # dump API from a Python thread, which holds the GIL while dumping.
            if sys.stderr is None:
                raise RuntimeError("stderr is unavailable")
            descriptor = sys.stderr.fileno()
            if interval > TIMEOUT_MAX:
                raise OverflowError("Stack dump interval exceeds thread timeout")

            def dump_stacks():
                while not cancelled.wait(interval):
                    try:
                        os.write(descriptor, f"Timeout (startup, {interval:g}s)!\n".encode())
                        faulthandler.dump_traceback(file=descriptor, all_threads=True)
                    except (OSError, ValueError, RuntimeError):
                        logger.warning("Could not dump startup stacks", exc_info=True)
                        return

            watchdog = Thread(target=dump_stacks, name="qwen-startup-stacks", daemon=True)
            logger.info(
                "Startup stack dump every %gs until loading completes; dumps do not stop the worker",
                interval,
            )
            watchdog.start()
        except (OSError, ValueError, RuntimeError, OverflowError):
            watchdog = None
            logger.warning("Could not enable startup stack dumps", exc_info=True)
    try:
        with log_stage("startup"):
            yield
    finally:
        cancelled.set()
        if watchdog is not None:
            watchdog.join()


def log_runtime_info(torch) -> None:
    packages = []
    for name in ("diffusers", "transformers", "peft", "accelerate", "runpod"):
        try:
            packages.append(f"{name}={version(name)}")
        except PackageNotFoundError:
            packages.append(f"{name}=unavailable")
    logger.info(
        "Runtime python=%s torch=%s cuda_build=%s torch_threads=%s "
        "cpu_count=%s %s",
        platform.python_version(),
        torch.__version__,
        torch.version.cuda,
        torch.get_num_threads(),
        os.cpu_count(),
        " ".join(packages),
    )


def log_cuda_info(torch) -> None:
    available = torch.cuda.is_available()
    logger.info("CUDA available=%s compiled_arches=%s", available, torch.cuda.get_arch_list())
    if available:
        for index in range(torch.cuda.device_count()):
            device = torch.cuda.get_device_properties(index)
            logger.info(
                "GPU index=%s name=%s capability=sm_%s%s total_bytes=%s",
                index,
                device.name,
                device.major,
                device.minor,
                device.total_memory,
            )
